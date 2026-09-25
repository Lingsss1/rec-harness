import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.store import Store

class RoleRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.runner=object.__new__(AmazonAutoresearch)
        self.runner.store=Store(self.root/'memory');self.addCleanup(self.runner.store.close)
        self.runner.store.configure_budget(12);self.folder=self.root/'round';self.folder.mkdir()
        self.calls=0
        outer=self
        class Agent:
            def run(self,role,context,target):
                outer.calls+=1
                write_json(target/'request.json',{'role':role})
                write_json(target/'resources.json',{'status':'success'})
                response={'selection':'discard','conclusion_status':'unknown','rationale':'test',
                          'evidence_refs':['readout'],'unresolved_explanations':[]}
                write_json(target/'response.json',response)
                return response
        self.runner.agent=Agent()

    def test_cached_role_and_source_are_bound_without_rebilling(self):
        source=self.root/'model.py';source.write_text('x=1')
        a=self.runner._role(self.folder,'lead',{},source)
        b=self.runner._role(self.folder,'lead',{},source)
        self.assertEqual(a,b);self.assertEqual(self.calls,1)
        self.assertEqual(self.runner.store.budget()['charged_or_reserved'],1)
        (self.folder/'lead/model/model.py').write_text('x=2')
        with self.assertRaisesRegex(ContractError,'cached model'):self.runner._role(self.folder,'lead',{},source)
        self.assertEqual(self.calls,1)

    def test_bad_role_retries_only_twice_even_after_resume(self):
        def reject(response,target):raise ContractError('bad reasoning references')
        for _ in range(2):
            with self.assertRaisesRegex(ContractError,'automatic retry'):
                self.runner._role(self.folder,'lead',{},validator=reject)
        self.assertEqual(self.calls,2)
        self.assertEqual(self.runner.store.budget()['charged_or_reserved'],2)

    def test_transport_receipt_recovers_after_controller_loss(self):
        original=write_json
        def interrupt(path,value):
            if Path(path).name=='accepted.json':raise KeyboardInterrupt()
            return original(path,value)
        with patch('rec_harness.adapters.amazon_autonomous.write_json',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):self.runner._role(self.folder,'lead',{})
        self.runner.store.recover()
        self.runner._role(self.folder,'lead',{})
        self.assertEqual(self.calls,1)
        self.assertEqual(self.runner.store.budget()['charged_or_reserved'],1)
        events=self.runner.store.events()
        self.assertEqual(sum(e['kind']=='agent_submitted' for e in events),1)
        self.assertEqual(sum(e['kind']=='role_completed' for e in events),1)

    def test_accepted_marker_before_completion_event_recovers_exactly_once(self):
        original=self.runner.store.settle
        def interrupt(*args,**kwargs):raise KeyboardInterrupt()
        with patch.object(self.runner.store,'settle',interrupt):
            with self.assertRaises(KeyboardInterrupt):self.runner._role(self.folder,'lead',{})
        self.runner.store.recover()
        self.runner._role(self.folder,'lead',{});self.runner._role(self.folder,'lead',{})
        self.assertEqual(self.calls,1)
        self.assertEqual(sum(e['kind']=='role_completed' for e in self.runner.store.events()),1)
        self.assertEqual(self.runner.store.budget()['charged_or_reserved'],1)

class AmazonPipelineRecoveryTests(unittest.TestCase):
    def test_single_entry_recovers_completed_training_and_finishes_six_roles(self):
        import json
        from contextlib import ExitStack
        from rec_harness.autonomous import autoresearch
        from rec_harness.agents import schema_for
        from rec_harness.core import digest,file_hash
        from test_amazon_plan import sample
        from test_model_source_contract import SOURCE
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root=Path(tmp);study=root/'study';study.mkdir();source=study/'base.py';source.write_text(SOURCE)
            cfg={'model_versions':{'legacy_incumbent':{'path':str(source),'sha256':file_hash(source)}},
                 'protocol_hash':'protocol','protocol':{'metric':'HR@10','minimum_effect':0.005},
                 'datasets':['tiny'],'development_seeds':[42],'confirmation_seeds':[17,83],'python':'python3'}
            write_json(study/'study.json',cfg);write_json(study/'study-lock.json',{'config_hash':digest(cfg)})
            ledger=Store(study/'memory');ledger.configure_budget(10000);ledger.close()
            config={'adapter':'amazon-nextitnet-v1','study':str(study),'task_id':'auto','max_rounds':1,
                    'role_call_limit':12,'rescore_recipe':{'fixture':True},'uid_registry':'fixture',
                    'agent_backend':{'template':'unused','runtime_root':'unused'}}
            write_json(root/'config.json',config);campaign=root/'campaign';calls=[];fits=[]
            class Agent:
                def run(self,role,context,target):
                    calls.append(role);response=sample(schema_for(role))
                    if role=='evidence':response.update(gap_id='gap',model_version='legacy_incumbent',protocol_hash='protocol')
                    if role=='hypothesis':
                        response['method']['method_id']='method'
                        response['proposal'].update({k:v for k,v in context['bindings'].items() if k!='candidate_model_version'})
                        response['proposal'].update(gap_ids=['gap'],method_ids=['method'],budget=9000)
                    if role=='experiment':
                        response['changed_files']=['model.py']
                        (target/'model/model.py').write_text(SOURCE.replace('return self.block(x)','return self.block(x) * 0.9'))
                    if role in ('evaluation','lead'):response.update(selection='discard',conclusion_status='unknown',evidence_refs=['report'])
                    if role=='knowledge':response['patch_json']=json.dumps({'base_version':context['graph']['graph_version'],'nodes':[],'edges':[]})
                    for name,value in [('request.json',{'role':role}),('schema.json',schema_for(role)),('context.json',context),('response.json',response),('resources.json',{'status':'success'})]:write_json(target/name,value)
                    return response
            def gate(python,source,model_id,directory,env,timeout):
                write_json(directory/'report.json',{'checks':{'parameter_gradients':{'w':{'finite':True}}}})
                return {'status':'success'}
            def phase(root,pid,partition):
                marker=Path(root)/'trained.json'
                if not marker.exists():fits.append(partition);write_json(marker,{'completed':True})
            def graph(root,pid):
                s=Store(Path(root)/'memory');g={'task_id':'graph-task','protocol_hash':'protocol','graph_version':s.version,'nodes':[],'edges':[]};s.close()
                write_json(Path(root)/'research-readouts'/(pid+'-graph.json'),g);return g
            report={'hash':'report','protocol_hash':'protocol','summary':{'eligible_for_confirmation':False}}
            stack.enter_context(patch('rec_harness.adapters.amazon_autonomous.audit_job',return_value=({'metrics':{'HR@10':0.5}},{})))
            stack.enter_context(patch('rec_harness.adapters.amazon_admission.run_gate',side_effect=gate))
            stack.enter_context(patch('rec_harness.adapters.amazon_admission.run_training_smoke',
                side_effect=lambda python,source,directory,recipe,artifacts,registry:{'status':'success','report_ref':artifacts.put(source['path'])}))
            stack.enter_context(patch('rec_harness.adapters.amazon_autonomous.run_phase',side_effect=phase))
            stack.enter_context(patch('rec_harness.adapters.amazon_autonomous.ingest_development',side_effect=graph))
            for module in ['amazon_autonomous','amazon_cycle']:
                stack.enter_context(patch('rec_harness.adapters.'+module+'.development_readout',return_value=report))
            with patch.object(AmazonAutoresearch,'_rescore',side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):autoresearch(campaign,root/'config.json',agent=Agent())
            self.assertEqual(fits,['development']);self.assertEqual(calls,['evidence','hypothesis','experiment'])
            with patch.object(AmazonAutoresearch,'_rescore',return_value=[]):
                result=autoresearch(campaign,agent=Agent())
                again=autoresearch(campaign,agent=Agent())
            self.assertEqual(fits,['development']);self.assertEqual(len(calls),6)
            self.assertEqual(result,again);self.assertEqual(result['state']['round'],1)
            self.assertEqual(result['role_calls']['charged_or_reserved'],6)
            self.assertFalse((campaign/'rounds/round-1/private-confirmation.json').exists())
            with patch('rec_harness.adapters.amazon_autonomous.JailedCodexAgent',side_effect=AssertionError('audit must not initialize service')):
                audit=AmazonAutoresearch(campaign,load_agent=False)
                try:
                    verified=audit.verify()
                    self.assertEqual(verified['verified_roles'],6);self.assertEqual(verified['verified_cycles'],1)
                    self.assertEqual(verified['roles_with_legacy_partial_event_binding'],0)
                    with audit.lock():
                        other=AmazonAutoresearch(campaign,load_agent=False)
                        try:
                            with self.assertRaisesRegex(ContractError,'already active'):other.verify()
                        finally:other.close()
                    context=read_json(campaign/'rounds/round-1/lead/context.json')
                    context['injected']='changed after completion';write_json(campaign/'rounds/round-1/lead/context.json',context)
                    marker=read_json(campaign/'rounds/round-1/lead/accepted.json');marker['context_hash']=digest(context)
                    write_json(campaign/'rounds/round-1/lead/accepted.json',marker)
                    with self.assertRaisesRegex(ContractError,'binding differs'):audit.verify()
                finally:audit.close()

class ContextPrivacyTests(unittest.TestCase):
    def test_confirmation_failures_do_not_enter_next_role_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runner=object.__new__(AmazonAutoresearch)
            runner.root=root;runner.study=root/'study';runner.config={'task_id':'auto'}
            ledger=Store(runner.study/'memory');ledger.configure_budget(100);ledger.close()
            prior=root/'rounds/round-1';write_json(prior/'failure.json',{'reason':'CONFIRMATION_SECRET'})
            write_json(prior/'private-confirmation-started.json',{'scope':'host'})
            write_json(root/'rounds/round-2/failure.json',{'reason':'invalid source syntax'})
            write_json(root/'rounds/round-2/development-readout.json',{'hash':'dev-hash'})
            write_json(root/'rounds/round-2/handoff.json',{'cards':{'mechanism':'previous gated model'}})
            write_json(root/'rounds/round-2/knowledge/response.json',{'patch_json':'development-only explanation'})
            write_json(root/'rounds/round-2/private-confirmation.json',{'result':'PRIVATE_METRICS'})
            cfg={'datasets':['tiny'],'development_seeds':[42],'confirmation_seeds':[17,83],
                 'protocol':{},'protocol_hash':'p'}
            with patch('rec_harness.adapters.amazon_autonomous.load_config',return_value=cfg), patch('rec_harness.adapters.amazon_autonomous.audit_job',return_value=({'metrics':{'HR@10':0.5}},{})):
                context=runner._context({'incumbent':'base'},root/'rounds/round-3')
            self.assertNotIn('CONFIRMATION_SECRET',str(context))
            self.assertIn('invalid source syntax',str(context))
            self.assertIn('previous gated model',str(context))
            self.assertIn('development-only explanation',str(context))
            self.assertNotIn('PRIVATE_METRICS',str(context))
