import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
from rec_harness.core import ContractError,read_json,write_json,digest
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
        from rec_harness.adapters.amazon_auto_audit import verify_rejected_roles
        from rec_harness.core import Artifacts
        self.assertEqual(verify_rejected_roles(self.runner.store,Artifacts(self.root/'artifacts/objects')),2)

    def test_transport_failure_stops_without_retry_including_resume(self):
        for status in ['crash','timeout',None]:
            with self.subTest(status=status):
                folder=self.folder/str(status);folder.mkdir()
                calls=[]
                def fail(role,context,target):
                    calls.append(role)
                    write_json(target/'request.json',{'role':role})
                    if status:write_json(target/'resources.json',{'status':status})
                    from rec_harness.generation_workspace import capture
                    write_json(target/'model/draft.json',{'unfinished':True})
                    capture(target,target)
                    raise ContractError('host or network unavailable')
                before=self.runner.store.budget()['charged_or_reserved']
                with patch.object(self.runner.agent,'run',side_effect=fail):
                    for _ in range(3):
                        with self.assertRaisesRegex(ContractError,'explicitly reopen'):
                            self.runner._role(folder,'lead',{})
                self.assertEqual(calls,['lead'])
                self.assertEqual(self.runner.store.budget()['charged_or_reserved'],before+1)
                self.assertFalse(read_json(folder/'lead/rejection.json')['retryable'])
                from rec_harness.adapters.amazon_auto_audit import verify_rejected_roles
                from rec_harness.core import Artifacts
                artifacts=Artifacts(self.root/'artifacts/objects')
                failures=[e for e in self.runner.store.events() if e['kind']=='role_rejected']
                self.assertEqual(verify_rejected_roles(self.runner.store,artifacts),len(failures))
                ref=failures[-1]['payload']['archive']['returned-workspace/model/draft.json']
                self.assertEqual(read_json(artifacts.verify(ref)),{'unfinished':True})
                # Editing the mutable role directory cannot rewrite its failure archive.
                write_json(folder/'lead/returned-workspace/model/draft.json',{'changed':True})
                self.assertEqual(read_json(artifacts.verify(ref)),{'unfinished':True})
        artifacts.verify(ref).write_text('corrupt')
        with self.assertRaisesRegex(ContractError,'corrupt artifact'):verify_rejected_roles(self.runner.store,artifacts)

    def test_existing_failed_transport_receipt_is_not_retried(self):
        write_json(self.folder/'lead/request.json',{'role':'lead'})
        write_json(self.folder/'lead/resources.json',{'status':'crash'})
        write_json(self.folder/'lead/rejection.json',{'reason':'older transport failure'})
        with self.assertRaisesRegex(ContractError,'explicitly reopen'):
            self.runner._role(self.folder,'lead',{})
        self.assertEqual(self.calls,0)
        self.assertEqual(self.runner.store.budget()['charged_or_reserved'],0)

    def test_transport_failure_stops_campaign_and_explicit_repair_retains_cost(self):
        from contextlib import nullcontext
        from rec_harness.core import Artifacts
        r=self.runner;r.root=self.root;r.artifacts=Artifacts(self.root/'artifacts/objects')
        r.study=self.root/'study';ledger=Store(r.study/'memory');ledger.configure_budget(10000);ledger.close()
        config={'model_versions':{}}
        write_json(r.study/'study.json',config)
        write_json(r.study/'study-lock.json',{'config_hash':digest(config)})
        r.config={'max_rounds':3,'role_call_limit':12};r.lock=lambda:nullcontext()
        write_json(r.root/'state.json',{'round':0,'incumbent':'base','status':'ready'})
        original=r.agent.run;calls=[]
        def transport(role,context,target):
            calls.append(role)
            if len(calls)==1:
                write_json(target/'request.json',{'role':role})
                write_json(target/'resources.json',{'status':'crash'})
                raise ContractError('service unavailable')
            return original(role,context,target)
        def run_round(state,folder):
            r._role(folder,'lead',{})
            return {'selected':False}
        with patch.object(r.agent,'run',side_effect=transport),patch.object(r,'_round',side_effect=run_round):
            result=r.run()
            self.assertEqual(result['state']['stop_reason'],'role_transport_failure')
            self.assertEqual(result['state']['round'],1)
            self.assertEqual(result['rounds'][0]['failure_kind'],'role_transport')
            before=r.store.verify();cost=r.store.budget()
            self.assertEqual(r.run(),result)
            self.assertEqual(r.store.verify(),before);self.assertEqual(r.store.budget(),cost)
            self.assertEqual(calls,['lead'])
            repair=r.reopen_source_failure('service repaired')
            self.assertEqual(repair['transport_roles'],['lead'])
            resumed=r.run()
            self.assertEqual(resumed['state']['round'],3)
            self.assertEqual(resumed['state']['stop_reason'],'max_rounds')
            self.assertEqual(len(calls),4)  # One failed call plus three completed rounds.
            self.assertEqual(r.store.budget()['charged_or_reserved'],4)
            archives=list((r.root/'rounds/round-1').glob('lead-transport-archive-*'))
            self.assertEqual(len(archives),1)
            self.assertFalse(read_json(archives[0]/'rejection.json')['retryable'])
            from rec_harness.adapters.amazon_cycle import failed_role_archives
            self.assertIn(archives[0].name,failed_role_archives(r.root/'rounds/round-1'))
            from rec_harness.adapters.amazon_auto_audit import verify_reopen_history
            self.assertEqual(verify_reopen_history(r),1)

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

    def test_host_repair_reopens_without_repeating_successful_transport(self):
        from contextlib import nullcontext
        from rec_harness.core import Artifacts
        r=self.runner;r.root=self.root;r.artifacts=Artifacts(self.root/'artifacts/objects')
        r.study=self.root/'study';ledger=Store(r.study/'memory');ledger.configure_budget(10000);ledger.close()
        config={'model_versions':{}}
        write_json(r.study/'study.json',config)
        write_json(r.study/'study-lock.json',{'config_hash':digest(config)})
        r.config={'max_rounds':1,'role_call_limit':12};r.lock=lambda:nullcontext()
        self.folder=self.root/'rounds/round-1';self.folder.mkdir(parents=True)
        def reject(*args):raise ContractError('host capability omission')
        with self.assertRaises(ContractError):r._role(self.folder,'lead',{},validator=reject)
        failed={'status':'failed','reason':'host capability omission'}
        state={'round':1,'incumbent':'base','status':'ready'}
        write_json(self.folder/'failure.json',failed);write_json(self.root/'state.json',state)
        r.store.event('auto_round_completed',{'result':failed,'state':state})
        record=r.reopen_source_failure('Allow declared module calls after review')
        before=r.store.verify();cost=r.store.budget()
        self.assertEqual(r.reopen_source_failure('repeat'),record);self.assertEqual(r.store.verify(),before)
        self.assertIn('failure.json',record['prior_artifacts']);self.assertTrue(record['controller_sources'])
        def finish(state,folder):
            r._role(folder,'lead',{},validator=lambda *args:None)
            return {'selected':False}
        with patch.object(r,'_round',side_effect=finish):result=r.run()
        self.assertEqual(self.calls,2);self.assertEqual(r.store.budget(),cost)
        self.assertEqual(result['state']['round'],1)
        before=r.store.verify();self.assertEqual(r.run(),result);self.assertEqual(r.store.verify(),before)
        with self.assertRaisesRegex(ContractError,'not a source failure'):r.reopen_source_failure('no')
        from rec_harness.adapters.amazon_auto_audit import verify_reopen_history
        self.assertEqual(verify_reopen_history(r),1)
        altered=r.store.events()
        recovery=next(e for e in altered if e['kind']=='auto_round_reopened')
        recovery['payload']['retained_role_cost']=0
        with patch.object(r.store,'events',return_value=altered),patch.object(r.store,'verify',return_value=before):
            with self.assertRaisesRegex(ContractError,'amount changed'):verify_reopen_history(r)
        # Alter the archived response reference while keeping it a valid CAS object.
        write_json(self.root/'other.json',{'different':'output'})
        altered=r.store.events();recovery=next(e for e in altered if e['kind']=='auto_round_reopened')
        recovery['payload']['prior_artifacts']['lead/response.json']=r.artifacts.put(self.root/'other.json')
        with patch.object(r.store,'events',return_value=altered),patch.object(r.store,'verify',return_value=before):
            with self.assertRaisesRegex(ContractError,'pre-repair archive'):verify_reopen_history(r)

    def test_reopen_rejects_post_admission_and_running_work(self):
        from contextlib import nullcontext
        r=self.runner;r.root=self.root;r.lock=lambda:nullcontext()
        folder=self.root/'rounds/round-1';failed={'status':'failed','reason':'failure'}
        write_json(folder/'failure.json',failed)
        r.store.event('auto_round_completed',{'result':failed,'state':{'round':1,'incumbent':'base'}})
        write_json(folder/'handoff.json',{})
        with self.assertRaisesRegex(ContractError,'pre-admission'):r.reopen_source_failure('repair')
        (folder/'handoff.json').unlink();r.store.reserve('live',1)
        with self.assertRaisesRegex(ContractError,'unsettled'):r.reopen_source_failure('repair')

class AmazonPipelineRecoveryTests(unittest.TestCase):
    def test_mismatched_control_is_rejected_before_any_role_or_training(self):
        for changed in ['execution','candidates','data']:
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);study=root/'study'
                ledger=Store(study/'memory');ledger.configure_budget(100);ledger.close()
                runner=object.__new__(AmazonAutoresearch)
                runner.root=root;runner.study=study
                runner.config={'task_id':'task','execution_id':'exec','control_model_ids':['control']}
                cfg={'datasets':['tiny'],'development_seeds':[42]}
                def audit(root,config,version,dataset,seed):
                    result={'execution_id':'exec','metrics':{'HR@10':.5}}
                    manifest={'candidates_hash':'same','data_hashes':{'train':'same'}}
                    if version=='control':
                        if changed=='execution':result['execution_id']='other'
                        elif changed=='candidates':manifest['candidates_hash']='other'
                        else:manifest['data_hashes']={'train':'other'}
                    return result,manifest
                with patch('rec_harness.adapters.amazon_autonomous.load_config',return_value=cfg), \
                     patch('rec_harness.adapters.amazon_autonomous.audit_job',side_effect=audit), \
                     patch.object(runner,'_role',side_effect=AssertionError('unexpected Codex call')), \
                     patch('rec_harness.adapters.amazon_autonomous.run_phase',side_effect=AssertionError('unexpected training')):
                    with self.assertRaisesRegex(ContractError,'development control'):
                        runner._round({'incumbent':'base'},root/'round-1')
                self.assertFalse((root/'round-1/input.json').exists())

    def test_single_entry_recovers_completed_training_and_finishes_six_roles(self):
        self.run_pipeline(False)

    def test_confirmation_uses_same_frozen_execution_and_resumes_without_calls(self):
        self.run_pipeline(True)

    def test_measurement_is_committed_to_cycle_and_checked_on_recovery(self):
        self.run_pipeline(False,measurement=True)

    def test_legacy_measurement_cycle_coverage_is_reported_without_rewriting(self):
        self.run_pipeline(False,measurement=True,legacy_measurement=True)

    def test_development_history_reaches_reviewers_without_extra_roles(self):
        self.run_pipeline(False,history=True)

    def test_confirmation_comparator_interruption_resumes_without_new_roles(self):
        self.run_pipeline(True,interrupt_control=True)

    def test_registered_mechanism_control_survives_confirmation_recovery(self):
        self.run_pipeline(True,interrupt_control=True,extra_control=True)

    def test_candidate_can_pass_all_registered_confirmation_controls(self):
        self.run_pipeline(True,extra_control=True,control_metric=.55)

    def test_generated_control_is_admitted_archived_and_recovers_without_extra_roles(self):
        self.run_pipeline(True,generated_control=True,control_metric=.55,interrupt_control=True)

    def test_transport_failure_archive_survives_cycle_completion_and_detects_tampering(self):
        self.run_pipeline(False,transport_archive=True)

    def run_pipeline(self,confirmation,measurement=False,legacy_measurement=False,history=False,interrupt_control=False,extra_control=False,control_metric=.6,generated_control=False,transport_archive=False):
        import json
        from contextlib import ExitStack
        from rec_harness.autonomous import autoresearch
        from rec_harness.agents import schema_for
        from rec_harness.core import digest,file_hash
        from test_amazon_plan import sample
        from test_model_source_contract import SOURCE
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            if legacy_measurement:
                original_patch=Store.patch
                def legacy_patch(store,patch,**kwargs):
                    for kind,payload in kwargs.get('accompanying_events',[]):
                        if kind=='amazon_research_cycle_completed':payload['roles'].pop('measurement',None)
                    return original_patch(store,patch,**kwargs)
                stack.enter_context(patch.object(Store,'patch',legacy_patch))
            root=Path(tmp);study=root/'study';study.mkdir();source=study/'base.py';source.write_text(SOURCE)
            cfg={'model_versions':{'legacy_incumbent':{'path':str(source),'sha256':file_hash(source)}},
                 'protocol_hash':'protocol','protocol':{'metric':'HR@10','minimum_effect':0.005},
                 'datasets':['tiny'],'development_seeds':[42],'confirmation_seeds':[17,83],'python':'python3'}
            if extra_control:
                control=study/'control.py';control.write_text(SOURCE.replace('return self.block(x)','return self.block(x) * 0.8'))
                cfg['model_versions']['mechanism_control']={'path':str(control),'sha256':file_hash(control)}
            write_json(study/'study.json',cfg);write_json(study/'study-lock.json',{'config_hash':digest(cfg)})
            ledger=Store(study/'memory');ledger.configure_budget(10000);ledger.close()
            config={'adapter':'amazon-nextitnet-v1','study':str(study),'task_id':'auto','max_rounds':1,
                    'execution_id':'execution-test',
                    'role_call_limit':12,'rescore_recipe':{'fixture':True},'uid_registry':'fixture',
                    'agent_backend':{'template':'unused','runtime_root':'unused'}}
            if extra_control:config['control_model_ids']=['mechanism_control']
            if generated_control:config['generated_control_count']=1
            expected_reused=['legacy_incumbent']+(['mechanism_control'] if extra_control else [])
            control_id='auto-round-1-control-1' if generated_control else 'mechanism_control'
            expected_controls=expected_reused+([control_id] if generated_control else [])
            if measurement:config['measurement_discovery']={'development_budget':10}
            if history:
                config['history_diagnostic']=True
                history_call=stack.enter_context(patch('rec_harness.adapters.amazon_history_context.development_history',return_value={'report_hash':'development-slices','rows':[]}))
                stack.enter_context(patch('rec_harness.adapters.amazon_baseline_history.baseline_history',return_value={'report_hash':'baseline-slices','rows':[]}))
                stack.enter_context(patch('rec_harness.adapters.amazon_baseline_history.verify_visible_history'))
                stack.enter_context(patch('rec_harness.adapters.amazon_history_graph.attach_history',side_effect=lambda c,f,g,p:g))
            write_json(root/'config.json',config);campaign=root/'campaign';calls=[];fits=[];control_fits=[]
            self_test=self
            class Agent:
                def run(self,role,context,target):
                    calls.append(role);response=sample(schema_for(role))
                    if role=='evidence':
                        response.update(gap_id='gap',model_version='legacy_incumbent',protocol_hash='protocol')
                        if transport_archive:
                            # Archive fixture only; failed-call accounting is exercised
                            # separately through actual transport failure and reopening.
                            failed=target.parent/'evidence-transport-archive-fixture';failed.mkdir()
                            write_json(failed/'context.json',{'task_id':'failed-fixture'})
                            write_json(failed/'request.json',{'role':'evidence'})
                            write_json(failed/'resources.json',{'status':'crash'})
                            write_json(failed/'process.json',{'returncode':1})
                            write_json(failed/'rejection.json',{'retryable':False})
                            (failed/'stdout.jsonl').write_text('transport interrupted\n')
                    if role=='measurement':
                        response.update(assessment='sufficient',requirements=[],evidence_refs=context['allowed_evidence_refs'],rationale='Existing development aggregates suffice.')
                        response['prerequisites']={key:{'status':'satisfied','rationale':'Audited baseline supplied.'} for key in ['invocation','implementation','data','applicability']}
                    if role=='hypothesis':
                        self_test.assertEqual(context['bindings']['controls'],expected_controls)
                        if extra_control:self_test.assertEqual(context['development_controls'][0]['model_version'],'mechanism_control')
                        response['method'].update(method_id='method',origin='agent_hypothesis',sources=[])
                        response['proposal'].update({k:v for k,v in context['bindings'].items() if k!='candidate_model_version'})
                        response['proposal'].update(gap_ids=['gap'],method_ids=['method'],budget=9000)
                        if context.get('insertion_contract_format'):
                            response['proposal']['insertion_contract']=json.dumps({'schema':'nextitnet-method-scope-v1','allowed_methods':['NextItNetModel.forward'],'rationale':'fixture scale at model forward'})
                    if role=='experiment':
                        response['changed_files']=['model.py']
                        (target/'model/model.py').write_text(SOURCE.replace('return self.block(x)','return self.block(x) * 0.9'))
                        if generated_control:
                            response['changed_files'].append('control-1.py')
                            (target/'model/control-1.py').write_text(SOURCE.replace('return self.block(x)','return self.block(x) * 0.7'))
                    if role in ('evaluation','lead'):
                        author=context['implementation_author_report']
                        self_test.assertEqual(author['status'],'identity_verified')
                        self_test.assertFalse(author['checks_verified_by_this_report'])
                        self_test.assertEqual(author['reported_limitations'],sample(schema_for('experiment'))['limitations'])
                        if history:self_test.assertEqual(context['baseline_history']['report_hash'],'baseline-slices')
                        self_test.assertEqual(context['implementation_evidence']['status'],'verified')
                        self_test.assertEqual(context['implementation_evidence']['model_version'],context['model_version'])
                        delta=context['implementation_evidence']['structural_changes']
                        self_test.assertEqual([row['method'] for row in delta['methods']],['NextItNetModel.forward'])
                        execution=context['candidate_execution_evidence']
                        self_test.assertEqual(len(execution),1)
                        self_test.assertEqual(execution[0]['model_version'],context['model_version'])
                        self_test.assertEqual(execution[0]['source_sha256'],context['implementation_evidence']['source_sha256'])
                        self_test.assertEqual(execution[0]['status'],'unavailable')  # Fixture has no protocol manifest.
                        self_test.assertEqual(execution[0]['training_contract']['status'],'unavailable')
                        if generated_control:
                            controls=context['generated_control_evidence']
                            self_test.assertEqual(len(controls),1)
                            control=controls[0]
                            self_test.assertEqual(control['model_version'],control_id)
                            self_test.assertEqual(control['implementation_evidence']['status'],'verified')
                            self_test.assertEqual(control['execution_evidence'][0]['training_contract']['status'],'unavailable')
                            self_test.assertEqual(control['execution_evidence'][0]['source_sha256'],control['implementation_evidence']['source_sha256'])
                            self_test.assertNotEqual(control['implementation_evidence']['source_sha256'],context['implementation_evidence']['source_sha256'])
                            self_test.assertNotEqual(control['implementation_evidence']['structural_changes']['candidate_ast_hash'],delta['candidate_ast_hash'])
                        if history:self_test.assertEqual(context['development_history']['report_hash'],'development-slices')
                        self_test.assertEqual(context['model_version'],context['bindings']['candidate_model_version'])
                        if role=='evaluation':self_test.assertIsNone(context['visible_model_source'])
                        else:
                            self_test.assertEqual(context['visible_model_source']['sha256'],file_hash(target/'model/model.py'))
                            self_test.assertEqual(context['visible_model_source']['model_version'],context['model_version'])
                        response.update(selection='keep' if confirmation else 'discard',conclusion_status='unknown',evidence_refs=[report['hash']])
                    if role=='knowledge':
                        self_test.assertEqual(context['implementation_author_report']['reported_limitations'],sample(schema_for('experiment'))['limitations'])
                        self_test.assertFalse(context['implementation_author_report']['checks_verified_by_this_report'])
                        response['patch_json']=json.dumps({'base_version':context['graph']['graph_version'],'nodes':[],'edges':[]})
                    for name,value in [('request.json',{'role':role}),('schema.json',schema_for(role)),('context.json',context),('response.json',response),('resources.json',{'status':'success'})]:write_json(target/name,value)
                    if role=='hypothesis':
                        write_json(target/'literature-search.json',{'search_observed':False,'qualification':'fixture; no source verification'})
                        (target/'stdout.jsonl').write_text('{"type":"fixture"}\n')
                        write_json(target/'process.json',{'status':'success','qualification':'fixture'})
                    return response
            def gate(python,source,model_id,directory,env,timeout):
                write_json(directory/'report.json',{'checks':{'parameter_gradients':{'w':{'finite':True}}}})
                return {'status':'success'}
            def phase(root,pid,partition,execution_id=None):
                self.assertEqual(execution_id,'execution-test')
                marker=Path(root)/(partition+'-trained.json')
                if not marker.exists():fits.append(partition);write_json(marker,{'completed':True})
            def controls(root,model_versions,seeds,execution_id=None):
                self.assertEqual(execution_id,'execution-test')
                self.assertEqual(model_versions,expected_reused);self.assertEqual(seeds,[17,83])
                self.assertEqual(calls[-1],'knowledge')
                self.assertTrue((campaign/'rounds/round-1/private-confirmation-started.json').exists())
                marker=Path(root)/'confirmation-controls.json'
                if not marker.exists():
                    control_fits.append(model_versions);write_json(marker,{'completed':True})
                    if interrupt_control:raise KeyboardInterrupt()
            def graph(root,pid,*,include_method_sources=False):
                self_test.assertTrue(include_method_sources)
                s=Store(Path(root)/'memory');g={'task_id':'graph-task','protocol_hash':'protocol','graph_version':s.version,'nodes':[],'edges':[]};s.close()
                write_json(Path(root)/'research-readouts'/(pid+'-graph.json'),g);return g
            report={'schema':'amazon-development-selection-v1','proposal_id':'fixture','plan_handoff_hash':'fixture',
                'source_hashes':{},'source_receipts':{},'selection_policy_provenance':'fixture',
                'protocol_hash':'protocol','summary':{'eligible_for_confirmation':confirmation},
                'rows':[{'model_version':version,'dataset':'tiny','seed':42,'metrics':{'HR@10':.6 if version=='auto-round-1-model' else .5}}
                        for version in dict.fromkeys(['auto-round-1-model',*expected_controls])]}
            report['hash']=digest(report)
            def audited_job(root,config,version,dataset,seed):
                value=.6 if (extra_control or generated_control) and seed in [17,83] and version!='legacy_incumbent' else .5
                if (extra_control or generated_control) and seed in [17,83] and version==control_id:value=control_metric
                return ({'metrics':{'HR@10':value},'execution_id':'execution-test'},
                        {'candidates_hash':'same','data_hashes':{}})
            stack.enter_context(patch('rec_harness.adapters.amazon_autonomous.audit_job',side_effect=audited_job))
            stack.enter_context(patch('rec_harness.adapters.amazon_autonomous.release_confirmation',return_value={}))
            resolver=stack.enter_context(patch('rec_harness.adapters.amazon_execution.resolve_execution',return_value={'execution_id':'execution-test'}))
            stack.enter_context(patch('rec_harness.adapters.amazon_admission.run_gate',side_effect=gate))
            stack.enter_context(patch('rec_harness.adapters.amazon_admission.run_training_smoke',
                side_effect=lambda python,source,directory,recipe,artifacts,registry:{'status':'success','report_ref':artifacts.put(source['path'])}))
            stack.enter_context(patch('rec_harness.adapters.amazon_autonomous.run_phase',side_effect=phase))
            control_run=stack.enter_context(patch('rec_harness.adapters.amazon_autonomous.run_study',side_effect=controls))
            stack.enter_context(patch('rec_harness.adapters.amazon_autonomous.ingest_development',side_effect=graph))
            for module in ['amazon_autonomous','amazon_cycle']:
                stack.enter_context(patch('rec_harness.adapters.'+module+'.development_readout',return_value=report))
            with patch.object(AmazonAutoresearch,'_rescore',side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):autoresearch(campaign,root/'config.json',agent=Agent())
            self.assertEqual(fits,['development']);self.assertEqual(calls,['evidence']+(['measurement'] if measurement else [])+['hypothesis','experiment'])
            self.assertEqual(resolver.call_args.args[1],'execution-test')
            with patch.object(AmazonAutoresearch,'_rescore',return_value=[]):
                if interrupt_control:
                    with self.assertRaises(KeyboardInterrupt):autoresearch(campaign,agent=Agent())
                    self.assertEqual(fits,['development'])
                result=autoresearch(campaign,agent=Agent())
                again=autoresearch(campaign,agent=Agent())
            expected_roles=7 if measurement else 6
            self.assertEqual(fits,['development','confirmation'] if confirmation else ['development']);self.assertEqual(len(calls),expected_roles)
            self.assertEqual(result,again);self.assertEqual(result['state']['round'],1)
            self.assertEqual(control_fits,[expected_reused] if confirmation else [])
            self.assertEqual(control_run.call_count,2 if interrupt_control else int(confirmation))
            if history:history_call.assert_called_once()
            self.assertEqual(result['role_calls']['charged_or_reserved'],expected_roles)
            from rec_harness.adapters.amazon_progress import progress_report
            progress=progress_report(campaign)['rows'][-1]
            self.assertEqual(progress['incumbent'],result['state']['incumbent'])
            self.assertEqual(progress['costs']['role_calls_charged_or_reserved'],expected_roles)
            self.assertEqual(progress['costs']['study_cost_status'],'recorded')
            self.assertEqual((campaign/'rounds/round-1/private-confirmation.json').exists(),confirmation)
            if extra_control or generated_control:
                private=read_json(campaign/'rounds/round-1/private-confirmation.json')
                self.assertTrue(all(v>.005 for v in private['effects'].values()))
                self.assertEqual(private['control_effects'],{'17':{control_id:.6-control_metric},'83':{control_id:.6-control_metric}})
                self.assertEqual(private['selected'],control_metric<.6)
            with patch('rec_harness.adapters.amazon_autonomous.JailedCodexAgent',side_effect=AssertionError('audit must not initialize service')):
                audit=AmazonAutoresearch(campaign,load_agent=False)
                try:
                    verified=audit.verify()
                    self.assertEqual(verified['verified_roles'],expected_roles);self.assertEqual(verified['verified_cycles'],1)
                    self.assertEqual(verified['measurement_cycle_bindings'],int(measurement and not legacy_measurement))
                    self.assertEqual(verified['measurement_roles_without_cycle_binding'],int(legacy_measurement))
                    from rec_harness.adapters.amazon_cycle import finalize_cycle
                    folder=campaign/'rounds/round-1'
                    original_readout=read_json(folder/'development-readout.json')
                    write_json(folder/'development-readout.json',{**original_readout,'hash':'different-readout'})
                    with self.assertRaisesRegex(ContractError,'readout differs from committed cycle'):audit.verify()
                    write_json(folder/'development-readout.json',original_readout)
                    cycle=read_json(folder/'cycle.json')
                    if transport_archive:
                        failed_role='evidence-transport-archive-fixture'
                        receipt=cycle['roles'][failed_role]
                        self.assertEqual(receipt['response_validation']['status'],'rejected')
                        self.assertEqual(set(receipt['artifacts']),{'context.json','request.json','resources.json','rejection.json','stdout.jsonl','process.json'})
                        path=folder/failed_role/'stdout.jsonl';original=path.read_bytes()
                        path.write_bytes(original+b'changed')
                        with self.assertRaisesRegex(ContractError,'role archive differs from committed cycle'):audit.verify()
                        with self.assertRaisesRegex(ContractError,failed_role+' archive changed'):
                            finalize_cycle(study,'auto-round-1',folder)
                        path.write_bytes(original)
                        self.assertEqual(audit.verify(),verified)
                    if generated_control:
                        self.assertIn('model/control-1.py',cycle['roles']['experiment']['artifacts'])
                        marker=folder/'experiment/accepted.json';accepted=read_json(marker)
                        for broken in [{k:v for k,v in accepted.items() if k!='model_source_hashes'},
                                       {**accepted,'model_source_hashes':{'model.py':accepted['source_hash']}}]:
                            write_json(marker,broken)
                            with self.assertRaisesRegex(ContractError,'incomplete generated model source bindings'):audit.verify()
                            with self.assertRaisesRegex(ContractError,'incomplete generated model source bindings'):
                                audit._role(folder,'experiment',{})
                        write_json(marker,accepted)
                        path=folder/'experiment/model/control-1.py';original=path.read_bytes()
                        path.write_bytes(original+b'\n# changed after completion\n')
                        with self.assertRaisesRegex(ContractError,'model source changed'):audit.verify()
                        with self.assertRaisesRegex(ContractError,'model source changed'):
                            audit._role(folder,'experiment',{})
                        with self.assertRaisesRegex(ContractError,'experiment archive changed'):
                            finalize_cycle(study,'auto-round-1',folder)
                        path.write_bytes(original)
                    for name in ['literature-search.json','stdout.jsonl']:
                        self.assertIn(name,cycle['roles']['hypothesis']['artifacts'])
                        path=folder/'hypothesis'/name;original_bytes=path.read_bytes()
                        path.write_bytes(original_bytes+b' ')
                        with self.assertRaisesRegex(ContractError,'hypothesis archive changed'):
                            finalize_cycle(study,'auto-round-1',folder)
                        path.write_bytes(original_bytes)
                    self.assertEqual(finalize_cycle(study,'auto-round-1',folder),cycle)
                    if measurement and not legacy_measurement:
                        from rec_harness.adapters.amazon_cycle import finalize_cycle
                        folder=campaign/'rounds/round-1';original=read_json(folder/'measurement/context.json')
                        write_json(folder/'measurement/context.json',{**original,'note':'changed'})
                        with self.assertRaisesRegex(ContractError,'measurement archive changed'):
                            finalize_cycle(study,'auto-round-1',folder)
                        write_json(folder/'measurement/context.json',original)
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
    def test_mixed_execution_baseline_rejected_before_role_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runner=object.__new__(AmazonAutoresearch)
            runner.root=root;runner.study=root/'study';runner.config={'execution_id':'execution-frozen'}
            cfg={'datasets':['tiny'],'development_seeds':[42]}
            with patch('rec_harness.adapters.amazon_autonomous.load_config',return_value=cfg),patch('rec_harness.adapters.amazon_autonomous.audit_job',return_value=({'metrics':{'HR@10':.5}},{})),patch.object(runner,'_role',side_effect=AssertionError('must reject before calling Codex')):
                with self.assertRaisesRegex(ContractError,'baseline execution differs'):
                    runner._round({'incumbent':'base'},root/'rounds/round-1')

    def test_confirmation_failures_do_not_enter_next_role_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);runner=object.__new__(AmazonAutoresearch)
            runner.root=root;runner.study=root/'study';runner.config={'task_id':'auto'}
            runner.store=Store(root/'memory');self.addCleanup(runner.store.close)
            ledger=Store(runner.study/'memory');ledger.configure_budget(100);ledger.close()
            prior=root/'rounds/round-1';write_json(prior/'failure.json',{'reason':'CONFIRMATION_SECRET'})
            write_json(prior/'private-confirmation-started.json',{'scope':'host'})
            write_json(root/'rounds/round-2/failure.json',{'reason':'invalid source syntax'})
            from rec_harness.core import digest
            report={'schema':'amazon-development-selection-v1','protocol_hash':'p','rows':[{'seed':42}]}
            report['hash']=digest(report)
            write_json(root/'rounds/round-2/development-readout.json',report)
            write_json(root/'rounds/round-2/handoff.json',{'cards':{'mechanism':'previous gated model',
                'gap':{'gap_id':'old-gap','model_version':'previous','protocol_hash':'p','status':'supported'}}})
            write_json(root/'rounds/round-2/knowledge/response.json',{'patch_json':'development-only explanation'})
            write_json(root/'rounds/round-2/private-confirmation.json',{'result':'PRIVATE_METRICS'})
            cfg={'datasets':['tiny'],'development_seeds':[42],'confirmation_seeds':[17,83],
                 'protocol':{},'protocol_hash':'p'}
            with patch('rec_harness.adapters.amazon_autonomous.load_config',return_value=cfg), patch('rec_harness.adapters.amazon_autonomous.audit_job',return_value=({'metrics':{'HR@10':0.5}},{'candidates_hash':'same','data_hashes':{}})):
                context=runner._context({'incumbent':'base'},root/'rounds/round-3')
            self.assertNotIn('CONFIRMATION_SECRET',str(context))
            self.assertIn('invalid source syntax',str(context))
            self.assertIn('previous gated model',str(context))
            self.assertIn('development-only explanation',str(context))
            self.assertNotIn('PRIVATE_METRICS',str(context))
            self.assertEqual(context['history'],[report])
            self.assertEqual(context['research_memory'][0]['development_hash'],report['hash'])
            self.assertNotIn('development',context['research_memory'][0])
            self.assertEqual(context['gap_applicability']['checks'][0]['applicability'],'model_changed')
            self.assertEqual(context['gap_applicability']['checks'][0]['current_gap_status'],'unknown')
            self.assertEqual(context['gap_applicability']['checks'][0]['source_round'],'round-2')
