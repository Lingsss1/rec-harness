import unittest
from unittest.mock import patch
from rec_harness.core import ContractError,digest,read_json,write_json
from rec_harness.validation_preflight import preflight
from rec_harness.project import ResearchProject
import test_tool_online
import test_project


class ValidationPreflightTests(unittest.TestCase):
    setUp=test_tool_online.ToolOnlineTests.setUp

    def plan(self):
        path=self.root/'suite/suite.json'
        return {'identity':read_json(path)['identity'],'stages':{'utility':{'command':[
            '{python}','-m','rec_harness.tool_online','run','--suite',str(path),'--candidate','{candidate}',
            '--output','{output}','--stage','utility']}}}

    def test_suite_checked_without_process_or_filesystem_mutation(self):
        before={str(p.relative_to(self.root)):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        with patch('rec_harness.agents.run_process',side_effect=AssertionError('preflight executed process')):
            result=preflight(self.plan())
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['checks'][0]['scope'],'frozen suite and bound inputs')
        self.assertEqual(before,{str(p.relative_to(self.root)):p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_old_worker_rejected_even_with_consistent_suite_hash(self):
        path=self.root/'suite/suite.json';suite=read_json(path);suite['worker_hash']='0'*64
        suite['hash']=digest({k:v for k,v in suite.items() if k!='hash'});write_json(path,suite)
        with self.assertRaisesRegex(ContractError,'worker changed'):preflight(self.plan())

    def test_missing_dependency_and_wrong_plan_identity_rejected(self):
        plan=self.plan();plan['identity']={**plan['identity'],'tools':'wrong'}
        with self.assertRaisesRegex(ContractError,'plan identity'):preflight(plan)
        path=self.root/'suite/episodes/objects';reference=next(path.iterdir());reference.unlink()
        with self.assertRaises(ContractError):preflight(self.plan())


class ProjectPreflightTests(unittest.TestCase):
    setUp=test_project.ProjectTests.setUp

    def test_invalid_builtin_suite_stops_before_project_or_agent_creation(self):
        config=read_json(test_project.ProjectTests.config(self,'tool'))
        config['tool']['plan']['stages']['utility']['command']=[
            '{python}','-m','rec_harness.tool_online','run','--suite',str(self.root/'missing.json')]
        with patch.object(test_project.ProjectAgent,'run',side_effect=AssertionError('called Codex')):
            with self.assertRaises((ContractError,OSError)):ResearchProject.init(self.campaign,config)
        self.assertFalse(self.campaign.exists())
        # Disabled slots do not cause a false startup dependency.
        config['max_tool_updates']=0
        ResearchProject.init(self.campaign,config)
        self.assertTrue((self.campaign/'project-lock.json').exists())
