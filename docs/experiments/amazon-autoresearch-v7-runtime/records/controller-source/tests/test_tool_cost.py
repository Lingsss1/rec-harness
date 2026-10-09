import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import Artifacts,ContractError,digest,write_json
from rec_harness.store import Store
from rec_harness.tool_cost import audit_utility,run_validation


class ToolCostTests(unittest.TestCase):
    def test_mock_report_is_never_online_evidence(self):
        with self.assertRaisesRegex(ContractError,'real three-group'):audit_utility({'online':False})

    def test_development_cost_counted_once_and_candidate_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);candidate=root/'candidate';candidate.mkdir();(candidate/'diagnostic.py').write_text('fixture')
            objects=Artifacts(root/'objects');snapshot=objects.snapshot(candidate,['diagnostic.py'])
            write_json(root/'episode/campaign.json',{'diagnostic_tool':{'implementation':snapshot}})
            report={'online':True,'passed':True,'gain':0.1,'regression':0,'evaluation_ids':['task'],'identity':{},
                'comparison_groups':['no_tool','existing_tool','candidate_tool'],'total_cost':12,
                'binding':{'candidate_hash':digest(snapshot)},'rows':[{'candidate_tool':{'graph_directory':str(root/'episode/memory')}}]}
            write_json(root/'utility.json',report)
            store=Store(root/'development');store.configure_budget(10);store.reserve('tool-design',3);store.settle('tool-design',2,'success');store.close()
            # This unit test isolates accounting; actual utility validation is
            # deliberately mocked and cannot establish online evidence.
            with patch('rec_harness.tool_cost.audit_utility',side_effect=lambda *_: ([],set())):
                result=run_validation(root/'utility.json',candidate,root/'result.json',[root/'development'])
                self.assertEqual(result['total_cost'],14)
                from rec_harness.cost_audit import freeze_cost_prefix
                controller=Store(root/'controller');controller.configure_budget(10);controller.reserve('validator',1)
                write_json(root/'prefix.json',freeze_cost_prefix(controller,10));controller.settle('validator',1,'accepted');controller.close()
                with_prefix=run_validation(root/'utility.json',candidate,root/'result.json',[root/'development'],root/'prefix.json')
                self.assertEqual(with_prefix['total_cost'],15)
                with self.assertRaisesRegex(ContractError,'counted twice'):
                    run_validation(root/'utility.json',candidate,root/'result.json',[root/'controller'],root/'prefix.json')
                with self.assertRaisesRegex(ContractError,'counted twice'):
                    run_validation(root/'utility.json',candidate,root/'result.json',[root/'development',root/'development'])
                (candidate/'diagnostic.py').write_text('changed')
                with self.assertRaisesRegex(ContractError,'candidate changed'):
                    run_validation(root/'utility.json',candidate,root/'result.json')

    def test_frozen_suite_rejects_missing_cases_and_changed_comparison_configuration(self):
        import test_tool_online as fixture
        from rec_harness.core import read_json
        from rec_harness.tool_cost import bound_suite,check_episode_binding
        trial=fixture.ToolOnlineTests();trial.setUp();self.addCleanup(trial.doCleanups)
        with patch('rec_harness.toy_isolated.evaluate',side_effect=trial.evaluate),patch('rec_harness.diagnostics.run_cpu_job',side_effect=trial.execute):
            report=trial.run_groups()
        suite_path=trial.root/'suite/suite.json';suite,episodes,cases=bound_suite(report,suite_path)
        row=report['rows'][0];case=cases[row['evaluation_id']]
        for group in report['comparison_groups']:
            root=Path(row[group]['graph_directory']).parent;config=read_json(root/'campaign.json')
            check_episode_binding(root,config,case,episodes['baseline_policy'],suite,group,report)
            config['budget']+=1
            with self.assertRaisesRegex(ContractError,'configuration changed'):
                check_episode_binding(root,config,case,episodes['baseline_policy'],suite,group,report)
        report['evaluation_ids']=[]
        with self.assertRaisesRegex(ContractError,'omitted'):bound_suite(report,suite_path)
