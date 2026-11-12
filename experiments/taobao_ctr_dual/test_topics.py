"""Resource ownership, incremental topic gain, merge controls and cancellation."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from common import read,write,sha
import controller as base
from topics import TopicPool,load_references,validate_gpu_budget


def descriptor(name):return {'title':name,'question':'Question '+name,'hypothesis':'Hypothesis '+name}


class TopicsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        model=self.root/'reference.py';model.write_text('class Model:\n    pass\n')
        self.reference={'id':'baseline-din','baseline_kind':'din','status':'success','source_sha256':sha(model),
            'config':{'kind':'din'},'metrics':{'data_identity':'d','validation':{'auc':.6,'logloss':.3},'parameters':10},
            'wall_seconds':1,'model_path':str(model),'output':str(self.root/'reference'),'seed':42,'round':-1}
        self.protocol={'identity':'p','data_identity':'d','train_seed':42,'estimated_unit_seconds':.01,
                       'epochs':3,'batch_size':4096,'history_limit':128}
        self.pool=TopicPool(self.root,[0,1],[descriptor('A'),descriptor('B')],[self.reference],self.protocol,time.time()+1000)
    def tearDown(self):self.tmp.cleanup()

    def trial(self,gpu,auc,name='good'):
        return {**copy.deepcopy(self.reference),'id':f'g{gpu}-{name}','baseline_kind':None,
                'topic_id':self.pool.states[gpu]['active_topic_id'],'round':0,'role':'candidate',
                'metrics':{'data_identity':'d','validation':{'auc':auc,'logloss':.2},'parameters':10}}

    def positive_topics(self):
        for gpu,auc in [(0,.62),(1,.63)]:
            self.pool.apply_decision(gpu,{'action':'continue','reason':'retain','evidence_refs':['baseline-din']},self.reference)
            self.pool.states[gpu]['trials'].append(self.trial(gpu,auc))
            self.pool.states[gpu]['round']=1

    def decision(self,action='merge'):
        return {'action':action,'reason':'source gains and complementary mechanisms; compare costs','evidence_refs':['g0-good','g1-good'],
                'topic':descriptor('combined'),'source_topic_ids':['g0-t000','g1-t000'],
                'merge_control_refs':['g0-good','g1-good']}

    def test_each_gpu_requires_one_distinct_topic(self):
        for gpus,topics in [([0,0],[descriptor('A'),descriptor('B')]),([0,1],[descriptor('A')]),
                            ([0,1],[descriptor('A'),descriptor('A')])]:
            with self.assertRaises(ValueError):TopicPool(self.root,gpus,topics,[self.reference],self.protocol,time.time()+100)

    def test_gpu_lanes_respect_declared_resource_budget(self):
        validate_gpu_budget([0,1],{'gpus':{'auto':[0,1],'enhanced':[2,3]}})
        validate_gpu_budget([0],{'gpus':[0,1]})
        with self.assertRaises(ValueError):validate_gpu_budget([0,2],{'gpus':{'auto':[0,1],'enhanced':[2,3]}})

    def test_merge_requires_both_incremental_gains_and_both_controls(self):
        self.positive_topics();snapshot=self.pool.decision_snapshot(0);decision=self.decision()
        self.assertEqual(self.pool.validate_decision(0,{'topic_decision':decision},snapshot)['action'],'merge')
        bad=copy.deepcopy(decision);bad['merge_control_refs']=['g0-good']
        with self.assertRaisesRegex(ValueError,'both frozen'):self.pool.validate_decision(0,{'topic_decision':bad},snapshot)
        bad=copy.deepcopy(snapshot)
        next(t for t in bad['topic_portfolio']['board']['topics'] if t['id']=='g1-t000')['incremental_gain']=0
        with self.assertRaisesRegex(ValueError,'increments'):self.pool.validate_decision(0,{'topic_decision':decision},bad)
        bad=copy.deepcopy(decision);bad['source_topic_ids']=['g0-t000','invented']
        with self.assertRaises(ValueError):self.pool.validate_decision(0,{'topic_decision':bad},snapshot)

    def test_merge_uses_stronger_source_and_keeps_donor_gpu_topic(self):
        self.positive_topics();decision=self.decision();strong=self.pool.states[1]['trials'][0]
        merged=self.pool.apply_decision(0,decision,strong)
        self.assertEqual(merged,'g0-t001');self.assertEqual(self.pool.states[1]['active_topic_id'],'g1-t000')
        self.pool.states[0]['trials'].append(self.trial(0,.625,'merged'))
        topic=next(t for t in self.pool.board()['topics'] if t['id']==merged)
        self.assertAlmostEqual(topic['incremental_gain'],-.005)
        self.assertGreater(topic['observed_gain_vs_reference'],0)
        snap=self.pool.decision_snapshot(0);bad=self.decision();bad['source_topic_ids']=[merged,'g1-t000']
        bad['merge_control_refs']=['g0-merged','g1-good']
        with self.assertRaisesRegex(ValueError,'increments'):self.pool.validate_decision(0,{'topic_decision':bad},snap)

    def test_recovery_from_a_weak_parent_is_not_merge_eligible_below_shared_reference(self):
        self.positive_topics();snapshot=self.pool.decision_snapshot(0)
        topic=next(t for t in snapshot['topic_portfolio']['board']['topics'] if t['id']=='g0-t000')
        topic['incremental_gain']=.04;topic['observed_gain_vs_reference']=-.01
        with self.assertRaisesRegex(ValueError,'shared reference'):
            self.pool.validate_decision(0,{'topic_decision':self.decision()},snapshot)

    def test_switch_preserves_old_work_and_rejects_duplicate_active_question(self):
        self.positive_topics();before=copy.deepcopy(self.pool.states[0]['trials'])
        change={'action':'switch','reason':'limited further payoff','evidence_refs':['g0-good'],'topic':descriptor('C')}
        self.pool.apply_decision(0,change,self.pool.states[0]['trials'][0])
        self.assertEqual(self.pool.states[0]['trials'],before)
        self.assertEqual(self.pool.states[0]['topics'][0]['status'],'parked')
        self.assertEqual(self.pool.states[0]['topics'][1]['entry_trial_id'],'g0-good')
        duplicate={**change,'topic':descriptor('B')}
        with self.assertRaisesRegex(ValueError,'already active'):self.pool.apply_decision(0,duplicate)

    def proposal(self,parent='baseline-din'):
        study={'id':'fixture','protocol_ref':'p','evidence_refs':[parent],'question':'Fixture question','hypothesis':'Fixture hypothesis',
            'nearest_work':[{'source_ref':'fixture-primary','verification':'discovered','difference':'fixture'}],
            'inherited_components':[],'proposed_difference':'fixture mechanism','predictions':['positive'],
            'falsifiers':['negative'],'alternative_explanations':['capacity'],
            'contrasts':[{'parent_ref':parent,'intervention':'fixture','control':'fixture','held_fixed':[],
                         'remaining_confounds':[],'decision_if_supported':'retain','decision_if_not_supported':'switch'}],
            'cost_plan':{'estimate_ref':'fixture','all_variants_counted':True,'attribution_reserve':'fixture',
                         'assessment_reserve':'fixture','stop_rule':'fixture'}}
        return {'study':study,'parent_trial':parent,
                'variants':[{'name':name,'file':name+'.py','role':role,'config':{}} for name,role in [('candidate','candidate'),('control','control')]]}

    def mock_researcher(self,directory,*args,**kwargs):
        write(directory/'proposal.json',self.proposal())
        for name in ['candidate','control']:(directory/(name+'.py')).write_text('class Model:\n    pass\n')
        return {'status':'success','wall_seconds':.001,'usage':[]}

    def test_lanes_run_in_parallel_but_each_gpu_is_serial(self):
        lock=threading.Lock();active={0:0,1:0};peak=0;calls=[]
        def train(**job):
            nonlocal peak
            with lock:
                active[job['gpu']]+=1;self.assertEqual(active[job['gpu']],1)
                peak=max(peak,sum(active.values()));calls.append(job['gpu'])
            time.sleep(.03)
            job['out'].mkdir(parents=True);shutil.copyfile(job['source'],job['out']/'model.py')
            result={'status':'success','wall_seconds':.03,'seed':job['seed'],'gpu':job['gpu'],
                'config':job['config'],'source_sha256':sha(job['source']),'output':str(job['out']),
                'metrics':{'data_identity':'d','validation':{'auc':.61,'logloss':.2},'parameters':10}}
            with lock:active[job['gpu']]-=1
            return result
        with (patch.object(base,'researcher',side_effect=self.mock_researcher),patch.object(base,'train_job',side_effect=train),
              patch.object(base,'prepare_template'),patch.object(base,'final_assessment') as assess):
            self.pool.run(max_rounds=1)
        self.assertEqual(sorted(calls),[0,0,1,1]);self.assertGreaterEqual(peak,2)
        ids=[t['id'] for s in self.pool.states.values() for t in s['trials']]
        self.assertEqual(len(set(ids)),4);assess.assert_not_called()
        self.assertFalse(read(self.root/'status.json')['assessment_launched'])

    def test_stop_during_proposal_never_trains_or_assesses(self):
        def researcher(directory,*args,**kwargs):
            write(self.root/'STOP',{'reason':'user request'});return {'status':'success','wall_seconds':.01,'usage':[]}
        with (patch.object(base,'researcher',side_effect=researcher),patch.object(base,'prepare_template'),
              patch.object(base,'train_job') as train,patch.object(base,'final_assessment') as assess):
            self.pool.run(max_rounds=1)
        train.assert_not_called();assess.assert_not_called()
        self.assertEqual(read(self.root/'status.json')['phase'],'terminated_by_user')

    def test_merge_round_executes_on_own_gpu_and_measures_against_stronger_source(self):
        self.positive_topics();self.pool.apply_decision(0,self.decision(),self.pool.states[1]['trials'][0])
        proposal=self.proposal('g0-good')
        proposal['study']['evidence_refs']=['g0-good','g1-good']
        proposal['study']['contrasts'][0]['control']='g0-good and g1-good'
        calls=[]
        def researcher(directory,*args,**kwargs):
            write(directory/'proposal.json',proposal)
            for name in ['candidate','control']:(directory/(name+'.py')).write_text('class Model:\n    pass\n')
            return {'status':'success','wall_seconds':.01,'usage':[]}
        def train(**job):
            calls.append(job['gpu']);job['out'].mkdir(parents=True)
            shutil.copyfile(job['source'],job['out']/'model.py')
            return {'status':'success','wall_seconds':1,'config':{},'source_sha256':sha(job['source']),
                'output':str(job['out']),'seed':42,'metrics':{'validation':{'auc':.625,'logloss':.2},'parameters':10}}
        with patch.object(base,'researcher',side_effect=researcher),patch.object(base,'train_job',side_effect=train):
            self.pool.execute_round(0)
        self.assertEqual(calls,[0,0]);self.assertEqual(self.pool.states[1]['active_topic_id'],'g1-t000')
        merged=next(t for t in self.pool.board()['topics'] if t['id']=='g0-t001')
        self.assertEqual(merged['entry_trial_id'],'g1-good');self.assertAlmostEqual(merged['incremental_gain'],-.005)

    def test_merge_metadata_cannot_replace_actual_study_controls(self):
        self.positive_topics();self.pool.apply_decision(0,self.decision(),self.pool.states[1]['trials'][0])
        proposal=self.proposal('g0-good')
        def researcher(directory,*args,**kwargs):
            write(directory/'proposal.json',proposal)
            for name in ['candidate','control']:(directory/(name+'.py')).write_text('class Model:\n    pass\n')
            return {'status':'success','wall_seconds':.01,'usage':[]}
        with (patch.object(base,'researcher',side_effect=researcher),patch.object(base,'train_job') as train,
              self.assertRaisesRegex(ValueError,'both frozen')):
            self.pool.execute_round(0)
        train.assert_not_called();self.assertEqual(len(self.pool.states[0]['topics']),2)

    def test_stopped_campaign_cannot_be_reopened(self):
        write(self.root/'STOP',{'reason':'user request'})
        with patch.object(base,'prepare_template') as prepare,self.assertRaises(ValueError):self.pool.run()
        prepare.assert_not_called()

    def test_three_failed_research_calls_preserve_receipts_and_never_assess(self):
        def researcher(directory,*args,**kwargs):
            write(directory/'agent-receipt.json',{'status':'timeout','wall_seconds':2,'usage':[]})
            raise RuntimeError('fixture timeout')
        with (patch.object(base,'researcher',side_effect=researcher),patch.object(base,'prepare_template'),
              patch.object(base,'train_job') as train,patch.object(base,'final_assessment') as assess):
            self.pool.run(max_rounds=3)
        train.assert_not_called();assess.assert_not_called()
        for lane in self.pool.states.values():
            self.assertEqual(lane['phase'],'failed_before_assessment')
            self.assertEqual(len(lane['agent_calls']),3)
        self.assertEqual(read(self.root/'status.json')['phase'],'topic_research_with_failures')

    def test_nested_topic_phase_uses_original_auto_context_and_only_local_sources(self):
        self.positive_topics();state=self.pool.call_state(0)
        directory=self.root/'topics/gpu-0/rounds/r0000/propose';directory.mkdir(parents=True)
        write(self.root/'auto-validation-only.json',{'enabled':True})
        with (patch.object(base,'knowledge_resources',return_value=[]) as resources,
              patch.object(base,'agent_run',return_value={'status':'success'}) as agent):
            base.researcher(directory,'propose','auto',state,self.protocol,time.time()+30,[],campaign_root=self.root)
        self.assertEqual(resources.call_args.args[0],self.root)
        runtime=agent.call_args.args[2]
        self.assertNotIn('topic_portfolio',runtime['context'])
        self.assertEqual(agent.call_args.args[1]['research_topic']['id'],'g0-t000')
        self.assertNotIn('g1-good',runtime['context']['allowed_observation_ids'])
        self.assertIn('validation only',runtime['context']['final_assessment_scope'])
        self.assertFalse((directory/'history_g1_good.py').exists())
        self.assertNotIn('g1-good',{t['id'] for t in resources.call_args.args[1]['trials']})

    def test_local_context_matches_legacy_auto_and_is_unchanged_by_peer_progress(self):
        self.positive_topics();state=self.pool.call_state(0)
        project=Path(__file__).absolute().parents[2]
        source=project/'artifacts/taobao-ctr-dual-20261104/source-at-memory-index-repair-v3/controller.py'
        spec=importlib.util.spec_from_file_location('legacy_auto_context',source)
        legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
        expected=legacy.phase_contract('propose','auto',state,self.protocol,1000)
        actual=base.phase_contract('propose','auto',state,self.protocol,1000)
        self.assertEqual(actual,expected)
        peer=self.trial(1,.99,'peer-new');peer['metrics']['candidate_diagnostics']={'secret':'peer-only'*1000}
        self.pool.states[1]['trials'].append(peer)
        self.assertEqual(self.pool.call_state(0),state)

    def test_research_keeps_original_auto_best_reference_when_new_variant_regresses(self):
        self.pool.states[0]['trials'].append(self.trial(0,.59,'regression'))
        self.assertEqual(self.pool.call_state(0)['incumbent'],'baseline-din')
        self.pool.states[0]['trials'].clear();self.positive_topics()
        self.pool.apply_decision(0,self.decision(),self.pool.states[1]['trials'][0])
        self.pool.states[0]['trials'].append(self.trial(0,.625,'merged'))
        self.assertEqual(self.pool.call_state(0)['incumbent'],'g1-good')

    def test_light_decision_has_summaries_but_no_models_papers_or_full_history(self):
        self.positive_topics()
        self.pool.states[1]['trials'][0]['metrics']['candidate_diagnostics']={'secret':'peer-only'*1000}
        def agent(directory,*args,**kwargs):
            write(directory/'topic-decision.json',{'action':'continue','reason':'retain this question',
                  'evidence_refs':['g0-good']})
            return {'status':'success','wall_seconds':.01,'usage':[]}
        with patch.object(base,'agent_run',side_effect=agent) as call:
            self.pool.decide(0,0)
        args=call.call_args.args;runtime=args[2];workspace=args[3]
        self.assertEqual(runtime['phase'],'topic_decision')
        self.assertIn('g1-good',runtime['context']['allowed_evidence_refs'])
        self.assertNotIn('peer-only',json.dumps(runtime))
        for key in ['trajectory_index','history','study_schema','candidate_api']:self.assertNotIn(key,runtime['context'])
        self.assertEqual(workspace['model_files'],[]);self.assertEqual(args[4],[])
        self.assertEqual(call.call_args.kwargs['resources'],[])
        self.assertFalse(list(args[0].glob('*.py')))
        self.assertNotIn('g1-good',{t['id'] for t in self.pool.call_state(0)['trials']})

    def test_switch_without_import_does_not_inherit_old_topic_context(self):
        self.positive_topics()
        self.pool.apply_decision(0,{'action':'switch','reason':'new question','evidence_refs':['g0-good'],
                                  'topic':descriptor('C')},self.reference)
        state=self.pool.call_state(0)
        self.assertEqual({t['id'] for t in state['trials']},{'baseline-din'})
        self.assertEqual(state['incumbent'],'baseline-din')
        self.assertEqual(len(self.pool.states[0]['trials']),1)

    def test_stop_during_light_decision_does_not_apply_or_train(self):
        self.positive_topics()
        def agent(directory,*args,**kwargs):
            write(self.root/'STOP',{'reason':'user stop'})
            return {'status':'success','wall_seconds':.01,'usage':[]}
        with patch.object(base,'agent_run',side_effect=agent),patch.object(base,'train_job') as train:
            self.pool.decide(0,0)
        train.assert_not_called();self.assertEqual(self.pool.states[0]['active_topic_id'],'g0-t000')

    def test_decision_errors_do_not_enter_auto_research_error_feedback(self):
        self.pool.states[0]['last_decision_error']={'error':'scheduler-only failure'}
        state=self.pool.call_state(0)
        runtime=base.phase_contract('propose','auto',state,self.protocol,1000)
        self.assertIsNone(runtime['context']['previous_runtime_error'])
        self.assertNotIn('scheduler-only',json.dumps(runtime))
        self.assertEqual(self.pool.decision_snapshot(0)['light_context']['last_decision_error']['error'],'scheduler-only failure')

    def test_peer_sources_enter_research_only_after_an_explicit_import_decision(self):
        self.positive_topics()
        self.assertNotIn('g1-good',{t['id'] for t in self.pool.call_state(0)['trials']})
        decision={'action':'switch','reason':'borrow the selected source for a new question',
            'evidence_refs':['g1-good'],'start_trial_ref':'g1-good','topic':descriptor('C')}
        def agent(directory,*args,**kwargs):
            write(directory/'topic-decision.json',decision)
            return {'status':'success','wall_seconds':.01,'usage':[]}
        with patch.object(base,'agent_run',side_effect=agent):self.pool.decide(0,0)
        state=self.pool.call_state(0);ids={t['id'] for t in state['trials']}
        self.assertIn('g1-good',ids);self.assertNotIn('g0-good',ids)
        self.assertEqual(state['incumbent'],'g1-good')

    def test_decision_interval_does_not_change_the_internal_auto_proposal_loop(self):
        phases=[]
        def agent(directory,*args,**kwargs):
            phases.append(args[1]['phase'])
            write(directory/'topic-decision.json',{'action':'continue','reason':'continue based on observed results',
                  'evidence_refs':['baseline-din']})
            return {'status':'success','wall_seconds':.01,'usage':[]}
        def train(**job):
            job['out'].mkdir(parents=True);shutil.copyfile(job['source'],job['out']/'model.py')
            return {'status':'success','wall_seconds':.01,'seed':42,'config':{},'source_sha256':sha(job['source']),
                'output':str(job['out']),'metrics':{'validation':{'auc':.61,'logloss':.2},'parameters':10}}
        with (patch.object(base,'researcher',side_effect=self.mock_researcher) as research,
              patch.object(base,'agent_run',side_effect=agent),patch.object(base,'train_job',side_effect=train),
              patch.object(base,'prepare_template')):
            self.pool.run(max_rounds=7)
        self.assertEqual(research.call_count,14);self.assertEqual(phases,['topic_decision','topic_decision'])
        for args,kwargs in research.call_args_list:
            self.assertNotIn('topic_portfolio',args[3]);self.assertEqual(args[1],'propose')

    def test_shared_reference_recipe_is_checked(self):
        project=Path(__file__).absolute().parents[2];campaign=project/'artifacts/taobao-ctr-dual-20261104'
        protocol=read(campaign/'protocol.json')
        self.assertEqual(len(load_references(campaign/'auto/state.json',protocol)),3)
        changed={**protocol,'epochs':protocol['epochs']+1}
        with self.assertRaisesRegex(ValueError,'recipe mismatch'):load_references(campaign/'auto/state.json',changed)

    def test_decision_interval_counts_completed_rounds_without_counting_proposal_failures(self):
        decisions=[]
        def execute(gpu):
            if self.pool.states[gpu]['round']==1:raise ValueError('proposal failure fixture')
            self.pool.states[gpu]['phase']='propose'
            return True
        def decide(gpu,index):
            decisions.append((gpu,index,self.pool.states[gpu]['completed_research_rounds']))
        with (patch.object(self.pool,'execute_round',side_effect=execute),
              patch.object(self.pool,'decide',side_effect=decide),patch.object(base,'prepare_template'),
              patch.object(base,'train_job') as train):
            self.pool.run(max_rounds=8)
        self.assertEqual(sorted(decisions),[(0,6,6),(1,6,6)])
        self.assertEqual([s['completed_research_rounds'] for s in self.pool.states.values()],[7,7])
        train.assert_not_called()


if __name__=='__main__':unittest.main()
