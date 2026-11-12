import shutil
from contextlib import ExitStack
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.amazon_tool_cost import audit_utility,run_validation,real_transports
from rec_harness.amazon_tool_online import load_suite,create_study,GROUPS
from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
from rec_harness.core import Artifacts,ContractError,digest,read_json,write_json
from rec_harness.cost_audit import freeze_cost_prefix
from rec_harness.store import Store
import test_amazon_tool_online


class AmazonToolCostTests(unittest.TestCase):
    setUp=test_amazon_tool_online.AmazonToolOnlineTests.setUp
    freeze=test_amazon_tool_online.AmazonToolOnlineTests.freeze

    def prepare(self):
        stack=ExitStack();self.addCleanup(stack.close)
        self.suite_path=self.freeze();self.suite,objects=load_suite(self.suite_path)
        self.utility=self.root/'utility/result.json';self.online=self.utility.parent/'amazon-tool-online'
        for name in ['candidate','existing']:shutil.copytree(self.root/'tool',self.online/name)
        candidate=Artifacts(self.root/'objects').snapshot(self.root/'tool',self.suite['tool_files'])
        binding={'suite_hash':self.suite['hash'],'candidate_hash':digest(candidate),'stage':'utility','scripted_agent':False}
        write_json(self.online/'binding.json',binding)
        self.case=self.suite['cases'][0];row={'evaluation_id':self.case['id']};self.configs={}
        for group in GROUPS:
            root=self.online/self.case['id']/group
            execution=create_study(self.case,root/'study',objects)
            config={**self.case['controller'],'study':str(root/'study'),
                    'task_id':'online-'+digest([self.case['id'],str(root)])[:24],'execution_id':execution}
            tool=None if group=='no_tool' else {**self.suite['tool_runtime'],'files':self.suite['tool_files'],
                'source':str(self.online/('candidate' if group=='candidate_tool' else 'existing'))}
            AmazonAutoresearch.init(root/'campaign',config,_validation_tool=tool)
            write_json(root/'started.json',{'case_hash':digest(self.case),'tool_hash':digest(tool),'scripted_agent':False})
            self.configs[group]=read_json(root/'campaign/amazon-auto.json')
            row[group]={'score':0.5,'protocol_hash':'p','evaluation_conditions':[{'population':'fixed'}],
                'diagnostic_calls':int(tool is not None),'diagnostic_failures':0,
                'costs':{'role_calls':{'charged_or_reserved':6},'gpu_seconds':{'charged_or_reserved':10},
                         'diagnostic_units':{'charged_or_reserved':int(tool is not None)}}}
            write_json(root/'completed.json',row[group])
        self.report={'online':True,'evaluation_mode':'end_to_end_candidate_generation','comparison_groups':list(GROUPS),
            'binding':binding,'identity':self.suite['identity'],'evaluation_ids':[self.case['id']],'rows':[row],
            'passed':True,'gain':0,'gain_by_reference':{'no_tool':0,'existing_tool':0},'regression':0,
            'cost_axes':{'role_calls':18,'gpu_seconds':30,'diagnostic_units':2},'cost_weights':self.suite['cost_weights'],'total_cost':20.3}
        write_json(self.utility,self.report)
        stack.enter_context(patch('rec_harness.amazon_tool_cost.read_result',side_effect=lambda root:row[root.name]))
        stack.enter_context(patch('rec_harness.amazon_tool_cost.real_transports',return_value=6))
        stack.enter_context(patch('rec_harness.amazon_tool_cost.audit_exposure',side_effect=lambda root:{'exposures':[] if root.parent.name=='no_tool' else [{'measurement_status':'success','transport_completed':True}]}))

    def test_bound_utility_recomputed_and_inflated_score_rejected(self):
        self.prepare();report,suite,receipts,used=audit_utility(self.utility,self.suite_path)
        self.assertEqual(report,self.report);self.assertEqual(len(receipts),3);self.assertEqual(len(used),6)
        report['gain']=1;write_json(self.utility,report)
        with self.assertRaisesRegex(ContractError,'aggregate differs'):audit_utility(self.utility,self.suite_path)

    def test_changed_study_or_controller_not_accepted_with_same_scores(self):
        self.prepare();folder=self.online/self.case['id']/'candidate_tool'
        path=folder/'campaign/amazon-auto.json';config=read_json(path);config['research_policy']={'prompts':{'evidence':'new'}};write_json(path,config)
        with self.assertRaisesRegex(ContractError,'controller differs'):audit_utility(self.utility,self.suite_path)
        write_json(path,self.configs['candidate_tool'])
        path=folder/'study/study.json';study=read_json(path);study['development_seeds']=[999];write_json(path,study)
        with self.assertRaisesRegex(ContractError,'study differs'):audit_utility(self.utility,self.suite_path)

    def test_weighted_abandoned_gpu_cost_and_outer_bcu_are_retained_once(self):
        self.prepare();gpu=self.root/'abandoned-study/memory';cfg={'budget_gpu_seconds':100}
        write_json(gpu.parent/'study.json',cfg);write_json(gpu.parent/'study-lock.json',{'config_hash':digest(cfg)})
        offline=self.root/'offline/memory';outer=self.root/'outer/memory'
        for memory,limit,amount in [(gpu,100,60),(offline,10,3),(outer,10,2)]:
            store=Store(memory)
            try:
                store.configure_budget(limit);store.reserve('charged',amount);store.settle('charged',amount,'success')
                if memory==outer:write_json(self.root/'prefix.json',freeze_cost_prefix(store,limit))
            finally:store.close()
        result=run_validation(self.utility,self.root/'tool',self.root/'cost/result.json',[gpu,offline],self.root/'prefix.json',suite_path=self.suite_path)
        self.assertAlmostEqual(result['total_cost'],25.9)
        self.assertEqual([item['axis'] for item in result['development_costs']],['gpu_seconds','development_bcu'])
        self.assertEqual(result['evaluation_mode'],'end_to_end_candidate_generation')
        with self.assertRaisesRegex(ContractError,'counted twice'):
            run_validation(self.utility,self.root/'tool',self.root/'cost/result.json',[gpu,gpu],suite_path=self.suite_path)

    def test_scripted_flag_and_candidate_replacement_rejected(self):
        self.prepare();self.report['binding']['scripted_agent']=True;write_json(self.utility,self.report)
        with self.assertRaisesRegex(ContractError,'real Amazon'):audit_utility(self.utility,self.suite_path)
        self.report['binding']['scripted_agent']=False;write_json(self.utility,self.report)
        (self.root/'tool/diagnostic.py').write_text('# replacement')
        with self.assertRaisesRegex(ContractError,'candidate changed'):
            run_validation(self.utility,self.root/'tool',self.root/'cost/result.json',suite_path=self.suite_path)

    def test_transport_audits_every_role_and_rejects_missing_isolation(self):
        root=self.root/'transport';store=Store(root/'campaign/memory');objects=Artifacts(root/'campaign/artifacts/objects')
        self.addCleanup(store.close);isolation={'kind':'chroot-unprivileged','uid':200001}
        for role in ['evidence','hypothesis','experiment','evaluation','lead','knowledge']:
            folder=root/role
            write_json(folder/'request.json',{'argv':['codex','--model','fixture'],'isolation':isolation})
            write_json(folder/'resources.json',{'status':'success','isolation':isolation});write_json(folder/'response.json',{})
            store.event('role_completed',{'task':'round-1-'+role+'-1','archive':objects.snapshot(folder,['request.json','resources.json','response.json'])})
        # Call the real helper (prepare() and its patches are not used here).
        self.assertEqual(real_transports(root,{'max_rounds':1,'agent_backend':{'model':'fixture'}}),6)
        with self.assertRaisesRegex(ContractError,'transport differs'):
            real_transports(root,{'max_rounds':1,'agent_backend':{'model':'other'}})
        with self.assertRaisesRegex(ContractError,'completed real research roles'):
            real_transports(root,{'max_rounds':2,'agent_backend':{'model':'fixture'}})
        write_json(root/'knowledge/resources.json',{'status':'success','isolation':{}})
        store.event('role_completed',{'task':'round-1-knowledge-2','archive':objects.snapshot(root/'knowledge',['request.json','resources.json','response.json'])})
        with self.assertRaisesRegex(ContractError,'transport differs'):
            real_transports(root,{'max_rounds':1,'agent_backend':{'model':'fixture'}})

    def test_group_recovery_requires_committed_and_rechecked_completion(self):
        from rec_harness.online_recovery import completed_groups
        target=self.root/'validation';folder=target/'amazon-tool-online/case/candidate_tool'
        folder.mkdir(parents=True)
        with self.assertRaisesRegex(ContractError,'outcome unknown'):
            completed_groups(target,'rec_harness.tool_pipeline','utility')
        write_json(folder/'completed.json',{'score':0.5})
        with patch('rec_harness.amazon_tool_online.read_result',return_value={'score':0.5}):
            completed_groups(target,'rec_harness.tool_pipeline','utility')
            completed_groups(target,'rec_harness.amazon_tool_online','utility')
        with patch('rec_harness.amazon_tool_online.read_result',return_value={'score':0.6}):
            with self.assertRaisesRegex(ContractError,'comparison changed'):
                completed_groups(target,'rec_harness.tool_pipeline','utility')
