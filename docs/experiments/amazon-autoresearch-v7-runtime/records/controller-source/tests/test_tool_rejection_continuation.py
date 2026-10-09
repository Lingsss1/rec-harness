"""Bounded continuation fixtures, not scientific tool utility evidence."""
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.autonomous import autoresearch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.project import ResearchProject
import test_project
import test_project_tool_plans


class ToolRejectionContinuationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_project_tool_plans.ProjectToolPlanTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.project_fixture=self.fixture.fixture
        self.root=self.project_fixture.root;self.campaign=self.project_fixture.campaign

    def config(self,*,all_fail=False,invalid=False,disabled=False):
        config=self.fixture.config()
        if disabled:config['continue_after_tool_rejection']=False
        plans=[config['tool']['plan']]+(config['tool_followup_plans'] if all_fail else [])
        for plan in plans:
            script=Path(plan['stages']['utility']['command'][1])
            text=script.read_text()
            text+='\nimport json\np=Path(sys.argv[1])\nr=json.loads(p.read_text())\nr["gain"]=0\n'
            if invalid:text+='r["online"]=False\n'
            script.write_text(text+'p.write_text(json.dumps(r))\n')
        path=self.root/'continued.json';write_json(path,config);return path

    def run_project(self,path,agent=None):
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=self.project_fixture.cpu):
            return autoresearch(self.campaign,path,agent=agent or test_project.ProjectAgent())

    def test_rejected_tool_history_reaches_next_attempt_before_model_resumes(self):
        path=self.config();seen=[]
        class Author(test_project.ProjectAgent):
            def run(self,role,context,directory):
                if role=='tool':seen.append(context)
                return super().run(role,context,directory)
        result=self.run_project(path,Author())
        self.assertEqual([e['kind'] for e in result['episodes']],['model','tool','tool','model'])
        self.assertEqual(len(seen),2)
        prior=seen[1]['context']['tool_history']['candidates'][0]
        self.assertEqual(prior['status'],'rejected')
        self.assertEqual(prior['validation'][-1]['failed_gates'],['minimum_gain'])
        self.assertEqual(result['state']['tool_updates'],2)
        self.assertEqual(result['state']['active_tool'],'episodes/tool-2/campaign')
        self.assertNotIn('pending_gap',result['state'])
        with patch.object(Author,'run',side_effect=AssertionError('repeated author')):
            self.assertEqual(self.run_project(None,Author()),result)

    def test_transition_interruption_does_not_repeat_rejected_author(self):
        path=self.config();calls=[]
        class Author(test_project.ProjectAgent):
            def run(self,role,context,directory):
                if role=='tool':calls.append(str(directory))
                return super().run(role,context,directory)
        original=ResearchProject.transition
        def interrupt(project,state,episode=None):
            original(project,state,episode)
            if episode and episode['kind']=='tool' and episode['index']==1:raise KeyboardInterrupt()
        with patch.object(ResearchProject,'transition',interrupt):
            with self.assertRaises(KeyboardInterrupt):self.run_project(path,Author())
        self.assertEqual(len(calls),1)
        result=self.run_project(None,Author())
        self.assertEqual(len(calls),2)
        self.assertEqual([e['kind'] for e in result['episodes']],['model','tool','tool','model'])
        self.assertEqual(len(set(calls)),2)

    def test_all_rejections_exhaust_plans_and_retain_gap_without_promotion(self):
        result=self.run_project(self.config(all_fail=True))
        self.assertEqual([e['kind'] for e in result['episodes']],['model','tool','tool'])
        self.assertEqual(result['state']['stop_reason'],'tool_rejected')
        self.assertEqual(result['state']['tool_updates'],2)
        self.assertIn('pending_gap',result['state']);self.assertNotIn('active_tool',result['state'])
        self.assertFalse((self.campaign/'episodes/tool-3').exists())

    def test_invalid_report_and_explicit_disable_never_retry(self):
        for invalid,disabled in [(True,False),(False,True)]:
            with self.subTest(invalid=invalid,disabled=disabled):
                self.campaign=self.root/('stop-'+str(invalid))
                result=self.run_project(self.config(invalid=invalid,disabled=disabled))
                self.assertEqual([e['kind'] for e in result['episodes']],['model','tool'])
                self.assertEqual(result['state']['stop_reason'],'tool_rejected')
                self.assertFalse((self.campaign/'episodes/tool-2').exists())

    def test_continuation_requires_predeclared_plans(self):
        config=read_json(self.project_fixture.config('tool'));config['continue_after_tool_rejection']=True
        with self.assertRaisesRegex(ContractError,'predeclared'):ResearchProject.init(self.campaign,config)
        self.assertFalse(self.campaign.exists())
