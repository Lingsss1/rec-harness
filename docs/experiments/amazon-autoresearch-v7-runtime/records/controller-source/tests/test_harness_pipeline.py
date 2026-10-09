import unittest
from pathlib import Path
from unittest.mock import patch
import test_online_validation
from rec_harness.core import Artifacts,ContractError,digest,read_json,write_json
from rec_harness.cost_audit import freeze_cost_prefix
from rec_harness.harness_pipeline import freeze_suite,load_suite,run_validation,costs
from rec_harness.online_validation import freeze_suite as freeze_online
from rec_harness.store import Store


class HarnessPipelineTests(unittest.TestCase):
    def setUp(self):
        test_online_validation.OnlineValidationTests.setUp(self)
        self.online=freeze_online(self.root/'spec.json',self.root/'online')
        context={'protocol':{'seeds':[7]}}
        prefix={'schema':'harness-prefix-suite-v1','identity':self.online['identity'],'baseline_policy':{},
            'agent_backend':{'model':'fixture-model'},'trace_source':{'task_id':'source-task'},
            'cases':[{'id':'historical','role':'evidence','context':context,'input_hash':digest(context),'source':{}}]}
        self.prefix=prefix;self.write_prefix()
        write_json(self.root/'pipeline-spec.json',{'prefix':'prefix/suite.json','online':'online/suite.json'})
        self.suite=self.root/'pipeline/suite.json'
        freeze_suite(self.root/'pipeline-spec.json',self.suite.parent)
        self.directory=self.root/'evolution/candidates/candidate/validation'
        parent=Store(self.root/'evolution/memory');parent.configure_budget(10);parent.reserve('stage',1)
        write_json(self.root/'cost-context.json',freeze_cost_prefix(parent,10));parent.close()

    def write_prefix(self):
        self.prefix['hash']=digest({k:v for k,v in self.prefix.items() if k!='hash'})
        write_json(self.root/'prefix/suite.json',self.prefix)

    def test_freeze_rejects_source_data_and_model_overlap_and_mutation(self):
        self.prefix['cases'][0]['context']['protocol']['seeds']=[101]
        self.prefix['cases'][0]['input_hash']=digest(self.prefix['cases'][0]['context']);self.write_prefix()
        with self.assertRaisesRegex(ContractError,'stage suite changed'):load_suite(self.suite)
        with self.assertRaisesRegex(ContractError,'data overlap'):freeze_suite(self.root/'pipeline-spec.json',self.root/'bad')
        self.prefix['cases'][0]['context']['protocol']['seeds']=[7]
        self.prefix['cases'][0]['input_hash']=digest(self.prefix['cases'][0]['context'])
        self.prefix['agent_backend']['model']='different';self.write_prefix()
        with self.assertRaisesRegex(ContractError,'base models'):freeze_suite(self.root/'pipeline-spec.json',self.root/'bad')

    def test_stage_routes_and_costs_include_controller_without_online_relabel(self):
        def run(suite,candidate,output):
            store=Store(output.parent/'prefix-replay/memory');store.configure_budget(4);store.reserve('pair',2);store.settle('pair',2,'accepted');store.close()
            report={'identity':self.online['identity'],'total_cost':2,'online':False,'passed':True}
            write_json(output,report);return report
        with patch('rec_harness.harness_prefix.run_validation',side_effect=run):
            result=run_validation(self.suite,'prefix_audit',self.root/'candidate',self.directory/'prefix_audit/result.json',self.root/'cost-context.json')
        self.assertEqual(result['stage_cost'],2);self.assertEqual(result['total_cost'],3)
        self.assertFalse(result['online']);self.assertEqual(len(result['cost_audit']['children']),1)
        self.assertEqual(read_json(self.directory/'prefix_audit/raw-result.json')['total_cost'],2)

    def test_predecessor_must_match_candidate_and_pipeline_before_online_call(self):
        implementation=Artifacts(self.root/'objects').snapshot(self.root/'candidate',['policy.json'])
        report={'identity':self.online['identity'],'pipeline_hash':'other'}
        write_json(self.directory/'prefix_audit/accepted.json',{'report':report,'report_hash':digest(report),'implementation_hash':digest(implementation)})
        with patch('rec_harness.online_validation.run_validation') as run:
            with self.assertRaisesRegex(ContractError,'another pipeline'):
                run_validation(self.suite,'development',self.root/'candidate',self.directory/'development/result.json',self.root/'cost-context.json')
            run.assert_not_called()

    def test_costs_retain_abandoned_prefix_and_development_charges(self):
        for path,charge in [(self.directory/'prefix_audit/prefix-replay/memory',2),
                            (self.directory/'development/online-episodes/case/baseline/memory',3),
                            (self.directory.parent/'validation-attempts/prefix_audit-1/prefix-replay/memory',1)]:
            store=Store(path);store.configure_budget(10);store.reserve('run',charge)
            if 'validation-attempts' not in str(path):store.settle('run',charge,'done')
            store.close()
        audit=costs(self.directory,'development',self.root/'cost-context.json')
        self.assertEqual(audit['total_cost'],7);self.assertEqual(len(audit['children']),3)
        self.assertEqual(costs(self.directory,'development',self.root/'cost-context.json'),audit)

    def test_precise_prompt_scope_is_valid_but_cannot_hide_other_edits(self):
        from rec_harness.evolution import validate_patch_scope
        validate_patch_scope(['prompts.lead'],{'prompts':{'lead':'Inspect recorded provenance.'}})
        validate_patch_scope(['prompts'],{'prompts':{'lead':'Inspect recorded provenance.'}})
        for scope,policy in [(['prompts.lead'],{'prompts':{'knowledge':'Unannounced'}}),
                             (['prompts.lead'],{'retrieval_limit':50}),
                             (['budget'],{'budget':999}),
                             (['prompts.unknown'],{'prompts':{}})]:
            with self.assertRaises(ContractError):validate_patch_scope(scope,policy)
