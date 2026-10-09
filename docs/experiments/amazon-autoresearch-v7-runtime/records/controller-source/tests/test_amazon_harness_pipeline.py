import copy
import unittest
from unittest.mock import patch
import test_amazon_harness_online as fixture
from rec_harness import harness_prefix,harness_pipeline,amazon_prefix
from rec_harness.core import Artifacts,ContractError,digest,read_json,write_json
from rec_harness.cost_audit import freeze_cost_prefix
from rec_harness.store import Store
from rec_harness.validation_preflight import evaluation_ids


class AmazonHarnessPipelineTests(unittest.TestCase):
    def setUp(self):
        fixture.AmazonHarnessOnlineTests.setUp(self)
        self.spec['baseline_policy']={}
        write_json(self.root/'online-spec.json',self.spec)
        from rec_harness.amazon_harness_online import freeze_suite
        self.online=freeze_suite(self.root/'online-spec.json',self.root/'online')
        self.source=self.root/'source';study=self.root/'source-study'
        config=read_json(self.configs[0]);cfg=read_json(self.root/'development/study.json')
        cfg.update(development_seeds=[1],confirmation_seeds=[2]);write_json(study/'study.json',cfg)
        write_json(study/'study-lock.json',{'config_hash':digest(cfg)})
        config.update(study=str(study),study_config_hash=digest(cfg),protocol_hash=cfg['protocol_hash'])
        write_json(self.source/'amazon-auto.json',config);write_json(self.source/'auto-lock.json',{'config_hash':digest(config)})
        (self.source/'model.py').write_text('original = True\n')
        objects=Artifacts(self.source/'artifacts/objects')
        self.context={'bindings':{'base_model_id':'base','train_protocol_hash':cfg['protocol_hash']}}
        trace={'trace_id':'historical','role':'evidence','visible_state':self.context,'input_hash':digest(self.context),
            'visible_source_snapshot':objects.snapshot(self.source,['model.py']),'submitted_event':'event',
            'role_output':{'must_not_send':'FUTURE_ANSWER'},'host_feedback':{'must_not_send':'FUTURE_OUTCOME'}}
        self.traces={'decision_traces':[trace],'source':{'task_id':'old','event_head':'fixture','cutoff_seq':2}}
        write_json(self.root/'prefix-spec.json',{'campaign':'source','roles':['evidence'],'budget':4,'call_cost':1,
            'agent_backend':{'model':'fixture'},'identity':self.online['identity']})
        with patch('rec_harness.amazon_prefix.collect_traces',return_value=self.traces):
            self.prefix=harness_prefix.freeze_suite(self.root/'prefix-spec.json',self.root/'prefix')
        write_json(self.candidate/'policy.json',{'prompts':{'evidence':'Inspect competing explanations.'}})
        write_json(self.root/'pipeline-spec.json',{'prefix':'prefix/suite.json','online':'online/suite.json'})
        self.path=self.root/'pipeline/suite.json'
        harness_pipeline.freeze_suite(self.root/'pipeline-spec.json',self.path.parent)
        self.response={'gap_id':'gap','model_version':'base','task_context':'task','protocol_hash':cfg['protocol_hash'],
            'observation_refs':[],'suspected_capability':'unknown','affected_components':[],
            'alternative_explanations':['noise','model'],'predicted_observations':['difference'],
            'cheapest_discriminating_test':'measurement','status':'unknown'}

    def test_amazon_prefix_keeps_only_visible_input_and_reuses_calls(self):
        self.assertNotIn('FUTURE_',str(self.prefix));calls=[];outer=self
        class Agent:
            def run(self,role,context,directory):
                calls.append(role);outer.assertEqual(context,outer.context)
                outer.assertEqual((directory/'model/model.py').read_text(),'original = True\n')
                return copy.deepcopy(outer.response)
        kwargs={'agent_factory':lambda *_:Agent()};output=self.root/'prefix-result.json'
        result=harness_prefix.run_validation(self.root/'prefix/suite.json',self.candidate,output,**kwargs)
        self.assertTrue(result['passed']);self.assertEqual(result['total_cost'],2)
        self.assertEqual(harness_prefix.run_validation(self.root/'prefix/suite.json',self.candidate,output,**kwargs),result)
        self.assertEqual(calls,['evidence','evidence'])
        with self.assertRaisesRegex(ContractError,'incumbent identity'):
            amazon_prefix.check_response(self.prefix['cases'][0],self.context,{**self.response,'model_version':'future'})

    def test_unavailable_retrieval_and_source_mutation_fail_before_calls(self):
        write_json(self.candidate/'policy.json',{'retrieval_limit':10})
        with self.assertRaisesRegex(ContractError,'historical memory retrieval'):
            harness_prefix.run_validation(self.root/'prefix/suite.json',self.candidate,self.root/'result.json')
        config=read_json(self.source/'amazon-auto.json');config['study_config_hash']='changed'
        write_json(self.source/'amazon-auto.json',config)
        with self.assertRaisesRegex(ContractError,'controller changed'):
            harness_prefix.freeze_suite(self.root/'prefix-spec.json',self.root/'new-prefix')

    def test_rejected_and_missing_source_traces_are_not_replayed(self):
        self.traces['decision_traces'][0]['role_admission']='rejected'
        with patch('rec_harness.amazon_prefix.collect_traces',return_value=self.traces):
            with self.assertRaisesRegex(ContractError,'no admitted'):
                harness_prefix.freeze_suite(self.root/'prefix-spec.json',self.root/'rejected-prefix')
        self.traces['decision_traces'][0].pop('role_admission')
        self.traces['decision_traces'][0]['visible_source_snapshot']=None
        with patch('rec_harness.amazon_prefix.collect_traces',return_value=self.traces):
            with self.assertRaisesRegex(ContractError,'pre-call source'):
                harness_prefix.freeze_suite(self.root/'prefix-spec.json',self.root/'missing-prefix')

    def test_development_decisions_require_actual_hash_and_gate(self):
        case={'role':'lead'};context={'bindings':{},'development':{'hash':'dev','summary':{'eligible_for_confirmation':False}}}
        decision={'selection':'discard','conclusion_status':'provisional','rationale':'insufficient evidence',
            'evidence_refs':['dev'],'unresolved_explanations':[]}
        amazon_prefix.check_response(case,context,decision)
        for changed in [{'evidence_refs':['later']},{'conclusion_status':'confirmed'},{'selection':'keep'}]:
            with self.assertRaises(ContractError):amazon_prefix.check_response(case,context,{**decision,**changed})
        context['development']['summary']['eligible_for_confirmation']=True
        amazon_prefix.check_response(case,context,{**decision,'selection':'keep'})

    def test_pipeline_routes_ids_and_rejects_overlapping_training_conditions(self):
        self.assertEqual(harness_pipeline.load_suite(self.path)['identity'],self.online['identity'])
        self.assertEqual(evaluation_ids('rec_harness.harness_pipeline',self.path,'development'),{'development'})
        self.assertEqual(evaluation_ids('rec_harness.harness_pipeline',self.path,'confirmation'),{'confirmation'})
        self.assertEqual(evaluation_ids('rec_harness.harness_pipeline',self.path,'prefix_audit'),{'historical'})
        self.prefix['source_conditions']=amazon_prefix.conditions(self.online['cases'][0]['study'])
        with self.assertRaisesRegex(ContractError,'overlap prefix source'):
            harness_pipeline.check_compatibility(self.prefix,self.online)

    def test_costs_convert_gpu_and_retain_abandoned_charges(self):
        directory=self.root/'evolution/candidates/candidate/validation'
        gpu=directory/'development/amazon-harness-online/case/baseline/study/memory'
        abandoned=directory.parent/'validation-attempts/prior/study/memory'
        for folder in [gpu,abandoned]:
            write_json(folder.parent/'study.json',{'budget_gpu_seconds':100})
            write_json(folder.parent/'study-lock.json',{'config_hash':digest({'budget_gpu_seconds':100})})
        prefix=directory/'prefix_audit/prefix-replay/memory';parent=self.root/'outer/memory'
        for folder,limit,charge in [(gpu,100,60),(abandoned,100,20),(prefix,4,2),(parent,10,1)]:
            store=Store(folder)
            try:
                store.configure_budget(limit);store.reserve('work',charge)
                if folder!=abandoned:store.settle('work',charge,'success')
                if folder==parent:write_json(self.root/'cost.json',freeze_cost_prefix(store,limit))
            finally:store.close()
        result=harness_pipeline.costs(directory,'development',self.root/'cost.json',cost_weights=self.online['cost_weights'])
        self.assertAlmostEqual(result['total_cost'],3.8)
        self.assertEqual(sum(row['axis']=='gpu_seconds' for row in result['children']),2)
        self.assertEqual(harness_pipeline.costs(directory,'development',self.root/'cost.json',cost_weights=self.online['cost_weights']),result)
        with self.assertRaisesRegex(ContractError,'lacks frozen cost weight'):
            harness_pipeline.costs(directory,'development',self.root/'cost.json',cost_weights={'role_calls':1})

    def test_pipeline_routes_online_and_requires_actual_audited_result(self):
        directory=self.root/'validation';implementation=Artifacts(self.root/'objects').snapshot(self.candidate,['policy.json'])
        suite=harness_pipeline.load_suite(self.path)
        previous={'identity':suite['identity'],'pipeline_hash':suite['hash']}
        write_json(directory/'prefix_audit/accepted.json',{'report':previous,'report_hash':digest(previous),'implementation_hash':digest(implementation)})
        report={'identity':suite['identity'],'total_cost':12.2,'online':True,'passed':True}
        def run(path,candidate,output,stage):write_json(output,report);return report
        with patch('rec_harness.amazon_harness_online.run_validation',side_effect=run) as run_online, \
             patch('rec_harness.amazon_harness_online.audit_report',return_value=report), \
             patch('rec_harness.harness_pipeline.costs',return_value={'total_cost':16}) as audit:
            result=harness_pipeline.run_validation(self.path,'development',self.candidate,directory/'development/result.json',self.root/'cost.json')
            self.assertEqual(run_online.call_args.args[-1],'development')
            self.assertEqual(audit.call_args.kwargs['cost_weights'],self.online['cost_weights'])
            self.assertEqual(result['stage_cost'],12.2);self.assertEqual(result['total_cost'],16)
        with patch('rec_harness.amazon_harness_online.run_validation',side_effect=run), \
             patch('rec_harness.amazon_harness_online.audit_report',return_value={**report,'online':False}):
            with self.assertRaisesRegex(ContractError,'real Amazon transports'):
                harness_pipeline.run_validation(self.path,'development',self.candidate,directory/'development/result.json',self.root/'cost.json')


if __name__=='__main__':unittest.main()
