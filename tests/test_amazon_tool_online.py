import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.amazon_tool_online import freeze_suite,load_suite,create_study,episode,run_validation,GROUPS
from rec_harness.core import ContractError,digest,file_hash,read_json,write_json
from rec_harness.store import Store


class AmazonToolOnlineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        data=self.root/'data';data.mkdir()
        for split in ['train','valid']:(data/('A_'+split+'.txt')).write_text('fixture '+split)
        model=self.root/'model.py';model.write_text('# fixed model\n')
        worker=self.root/'worker.py';worker.write_text('# fixed worker\n')
        self.configs=[]
        for index,stage in enumerate(['development','confirmation']):
            study=self.root/stage;study.mkdir()
            cfg={'study_id':stage,'datasets':['A'],'development_seeds':[100+index*10],
                 'confirmation_seeds':[101+index*10],
                 'model_versions':{'base':{'path':str(model),'sha256':file_hash(model)}},
                 'worker':str(worker),'worker_sha256':file_hash(worker),'python':'/usr/bin/python3',
                 'data':str(data),'data_hashes':{p.name:file_hash(p) for p in data.iterdir()},
                 'protocol':{'metric':'HR@10'},'protocol_hash':digest({'metric':'HR@10'}),
                 'budget_gpu_seconds':100,'job_timeout_seconds':10,'gpu_ids':[0],
                 'trusted_baselines_only':True,'hidden_evaluation':'not_available_to_worker'}
            write_json(study/'study.json',cfg);write_json(study/'study-lock.json',{'config_hash':digest(cfg)})
            config={'adapter':'amazon-nextitnet-v1','study':str(study),'task_id':'old',
                    'base_model_id':'base','max_rounds':1,'role_call_limit':6,'diagnostic_budget':2,
                    'rescore_recipe':{'runtime_dirs':['/usr'],'runtime_files':['/bin/true']},'uid_registry':'unused','agent_backend':{'kind':'jailed_codex',
                    'model':'fixture','template':'unused','runtime_root':'unused-runtime'}}
            path=self.root/(stage+'.json');write_json(path,config);self.configs.append(path)
        tool=self.root/'tool';tool.mkdir();(tool/'diagnostic.py').write_text('def measure(request): return {}\n')
        self.spec={'cost_weights':{'role_calls':1,'gpu_seconds':0.01,'diagnostic_units':1},
            'existing_tool':'tool','tool_files':['diagnostic.py'],'tool_runtime':{'cost':1,'timeout':10,
            'execution':{'recipe':{},'python':'/usr/bin/python3','uid_registry':'uids'}},
            'cases':[{'id':stage,'stage':stage,'config':str(config),
                      'execution_recipe':{'runtime_dirs':['/usr'],'runtime_files':['/bin/true']}}
                     for stage,config in zip(['development','confirmation'],self.configs)]}

    def freeze(self):
        write_json(self.root/'spec.json',self.spec)
        freeze_suite(self.root/'spec.json',self.root/'suite')
        return self.root/'suite/suite.json'

    def test_fresh_studies_have_same_inputs_and_independent_ledgers(self):
        suite,objects=load_suite(self.freeze());case=suite['cases'][0];roots=[]
        for group in GROUPS:
            root=self.root/group/'study';execution=create_study(case,root,objects);roots.append(root)
            config=read_json(root/'study.json')
            self.assertEqual(config['protocol_hash'],case['study']['protocol_hash'])
            self.assertEqual(file_hash(config['model_versions']['base']['path']),case['models']['base']['sha256'])
            self.assertFalse((root/'jobs').exists())
            store=Store(root/'memory')
            try:
                self.assertEqual(store.budget()['charged_or_reserved'],0)
                self.assertEqual([e['kind'] for e in store.events()],['study_registered','amazon_execution_registered'])
                self.assertEqual(store.events()[-1]['payload']['execution_id'],execution)
                store.reserve('only-this-group',1);store.settle('only-this-group',1,'success')
            finally:store.close()
        self.assertEqual(len({read_json(root/'study.json')['model_versions']['base']['path'] for root in roots}),3)
        with self.assertRaises(FileExistsError):create_study(case,roots[0],objects)

    def test_reused_condition_and_inherited_memory_rejected(self):
        source=self.root/'confirmation/study.json';cfg=read_json(source);cfg['development_seeds']=[100]
        write_json(source,cfg);write_json(source.parent/'study-lock.json',{'config_hash':digest(cfg)})
        with self.assertRaisesRegex(ContractError,'reuse a training condition'):self.freeze()
        self.assertFalse((self.root/'suite/suite.json').exists())
        cfg['development_seeds']=[110];write_json(source,cfg);write_json(source.parent/'study-lock.json',{'config_hash':digest(cfg)})
        config=read_json(self.configs[0]);config['development_history']=['old-readout.json'];write_json(self.configs[0],config)
        # Use a new output after the intentionally retained failed freeze artifacts.
        self.spec['cases'][0]['config']=str(self.configs[0]);write_json(self.root/'spec.json',self.spec)
        with self.assertRaisesRegex(ContractError,'fresh reference'):freeze_suite(self.root/'spec.json',self.root/'suite2')

    def test_private_data_and_code_change_rejected_before_execution(self):
        path=self.freeze()
        with patch('rec_harness.amazon_tool_online.code_identity',return_value={}):
            with self.assertRaisesRegex(ContractError,'implementation changed'):load_suite(path)
        (self.root/'data/A_valid.txt').write_text('changed')
        with self.assertRaisesRegex(ContractError,'private data identity'):load_suite(path)

    def test_three_group_scores_and_separate_cost_axes(self):
        path=self.freeze();calls=[]
        def run(case,root,objects,tool,**kwargs):
            group=root.name;calls.append((case['id'],group,tool))
            score={'no_tool':0.4,'existing_tool':0.5,'candidate_tool':0.6}[group]
            return {'score':score,'protocol_hash':'p','evaluation_conditions':[{'fixture':'same'}],'diagnostic_calls':int(tool is not None),'diagnostic_failures':0,
                    'costs':{'role_calls':{'charged_or_reserved':6},'gpu_seconds':{'charged_or_reserved':10},
                             'diagnostic_units':{'charged_or_reserved':int(tool is not None)}}}
        with patch('rec_harness.amazon_tool_online.episode',side_effect=run):
            report=run_validation(path,self.root/'tool',self.root/'dev/report.json','utility',agent_factory=lambda *_:object())
            self.assertAlmostEqual(report['gain'],0.1);self.assertEqual(report['regression'],0)
            self.assertAlmostEqual(report['total_cost'],20.3);self.assertFalse(report['online'])
            self.assertEqual(report['evaluation_mode'],'end_to_end_candidate_generation')
            self.assertEqual([c[1] for c in calls],list(GROUPS));self.assertIsNone(calls[0][2])
            self.assertEqual({c[0] for c in calls},{'development'})
            run_validation(path,self.root/'tool',self.root/'confirm/report.json','confirmation',agent_factory=lambda *_:object())
            self.assertEqual({c[0] for c in calls[3:]},{'confirmation'})
            with self.assertRaisesRegex(ContractError,'binding changed'):
                run_validation(path,self.root/'tool',self.root/'dev/report.json','confirmation',agent_factory=lambda *_:object())

    def test_unknown_group_never_restarts_and_completed_group_rechecks_evidence(self):
        suite,objects=load_suite(self.freeze());case=suite['cases'][0];root=self.root/'group';agent=object()
        with patch('rec_harness.amazon_tool_online.create_study',side_effect=KeyboardInterrupt()) as create:
            with self.assertRaises(KeyboardInterrupt):episode(case,root,objects,None,agent=agent)
            with self.assertRaisesRegex(ContractError,'outcome unknown'):episode(case,root,objects,None,agent=agent)
            self.assertEqual(create.call_count,1)
        saved={'score':0.5,'integrity':{'head':'frozen'}};write_json(root/'completed.json',saved)
        with patch('rec_harness.amazon_tool_online.read_result',return_value=saved),patch('rec_harness.amazon_tool_online.run_study',side_effect=AssertionError('no new jobs')):
            self.assertEqual(episode(case,root,objects,None,agent=agent),saved)
        with patch('rec_harness.amazon_tool_online.read_result',return_value={**saved,'score':0.7}):
            with self.assertRaisesRegex(ContractError,'group changed'):episode(case,root,objects,None,agent=agent)

    def test_episode_initializes_real_controller_and_runs_its_baseline_before_research(self):
        suite,objects=load_suite(self.freeze());case=suite['cases'][0];root=self.root/'episode';order=[]
        tool={**suite['tool_runtime'],'source':str(self.root/'tool'),'files':suite['tool_files']}
        saved={'score':0.5,'incumbent':'base'}
        with patch('rec_harness.amazon_tool_online.run_study',side_effect=lambda *a,**kw:order.append(('baseline',kw))), \
             patch('rec_harness.amazon_tool_online.AmazonAutoresearch._rescore',side_effect=lambda *a:order.append(('rescore',a))), \
             patch('rec_harness.amazon_tool_online.AmazonAutoresearch.run',side_effect=lambda:order.append(('research',None))), \
             patch('rec_harness.amazon_tool_online.read_result',return_value=saved):
            self.assertEqual(episode(case,root,objects,tool,agent=object()),saved)
        self.assertEqual([item[0] for item in order],['baseline','rescore','research'])
        self.assertEqual(order[0][1]['seeds'],case['study']['development_seeds'])
        config=read_json(root/'campaign/amazon-auto.json')
        self.assertTrue(config['validation_only'])
        self.assertEqual(config['study'],str(root/'study'))
        self.assertEqual(config['execution_id'],order[0][1]['execution_id'])

    def test_terminal_score_recomputed_from_receipts_and_committed_state(self):
        from rec_harness.amazon_tool_online import read_result
        from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
        suite,objects=load_suite(self.freeze());case=suite['cases'][0];root=self.root/'result'
        execution=create_study(case,root/'study',objects)
        config={**case['controller'],'task_id':'readout','study':str(root/'study'),'execution_id':execution}
        AmazonAutoresearch.init(root/'campaign',config)
        state={'round':1,'incumbent':'base','status':'ready'}
        store=Store(root/'campaign/memory')
        try:
            store.event('agent_submitted',{'task_id':'fixture','role':'evidence'})
            store.reserve('fixture',1);store.settle('fixture',1,'success')
            store.event('auto_round_completed',{'state':state,'result':{'selected':False}})
        finally:store.close()
        report={'state':{**state,'status':'stopped','stop_reason':'max_rounds'},'failed_rounds':0}
        write_json(root/'campaign/run-report.json',report)
        with patch.object(AmazonAutoresearch,'verify',return_value={'fixture':'audit'}), \
             patch('rec_harness.amazon_tool_online.progress_report',return_value={'rows':[{'development_score':0.5}]}), \
             patch('rec_harness.amazon_tool_online.audit_job',return_value=({'metrics':{'HR@10':0.5}},{'candidates_hash':'fixed','data_hashes':{'train':'t','validation':'v'}})):
            result=read_result(root)
            self.assertEqual(result['score'],0.5);self.assertEqual(result['costs']['role_calls']['charged_or_reserved'],1)
            with patch('rec_harness.amazon_tool_online.audit_job',return_value=({'metrics':{'HR@10':0.6}},{'candidates_hash':'fixed','data_hashes':{'train':'t','validation':'v'}})):
                with self.assertRaisesRegex(ContractError,'audited training receipts'):read_result(root)
            report['state']['incumbent']='invented';write_json(root/'campaign/run-report.json',report)
            with self.assertRaisesRegex(ContractError,'committed research'):read_result(root)

    def test_cross_group_evaluation_population_mismatch_rejected(self):
        path=self.freeze()
        def run(case,root,*args,**kwargs):
            return {'score':0.5,'protocol_hash':'p','evaluation_conditions':[{'candidates_hash':root.name}]}
        with patch('rec_harness.amazon_tool_online.episode',side_effect=run):
            with self.assertRaisesRegex(ContractError,'evaluation conditions differ'):
                run_validation(path,self.root/'tool',self.root/'dev/report.json','utility',agent_factory=lambda *_:object())
        self.assertFalse((self.root/'dev/report.json').exists())

    def test_explicit_new_seed_conditions_do_not_rewrite_source_study(self):
        source=self.root/'development/study.json';before=source.read_bytes()
        self.spec['cases'][0].update(development_seeds=[901],confirmation_seeds=[902,903])
        suite,_=load_suite(self.freeze())
        self.assertEqual(suite['cases'][0]['study']['development_seeds'],[901])
        self.assertEqual(suite['cases'][0]['study']['confirmation_seeds'],[902,903])
        self.assertEqual(source.read_bytes(),before)

    def test_builtin_preflight_checks_suite_identity_and_excluded_conditions(self):
        from rec_harness.validation_preflight import preflight
        path=self.freeze();suite=read_json(path)
        plan={'identity':suite['identity'],'development_ids':[],
              'stages':{'utility':{'command':['python3','-m','rec_harness.amazon_tool_online','run',
                 '--suite',str(path),'--stage','utility','--candidate','{candidate}','--output','{output}']}}}
        result=preflight(plan)
        self.assertEqual(result['checks'][0]['evaluation_ids'],['development'])
        plan['development_ids']=['development']
        with self.assertRaisesRegex(ContractError,'overlap development/history'):preflight(plan)
