import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.project import ResearchProject
from rec_harness.project_tool_plans import select
import test_project


class ProjectToolPlanTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_project.ProjectTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)

    def config(self):
        config=read_json(self.fixture.config('tool'))
        config.update(max_model_episodes=3,max_tool_updates=2)
        second=copy.deepcopy(config['tool']['plan'])
        second['identity']={key:'second-'+value for key,value in second['identity'].items()}
        second['budget']=30
        for stage,rule in second['stages'].items():
            report={'passed':True,'gain':1,'regression':0,'total_cost':7,
                'evaluation_ids':['second-'+stage+'-a','second-'+stage+'-b'],
                'identity':second['identity'],'comparison_groups':['no_tool','existing_tool','candidate_tool'],'online':True}
            script=self.fixture.root/('second-'+stage+'.py')
            script.write_text('import sys\nfrom pathlib import Path\nPath(sys.argv[1]).write_text('+repr(json.dumps(report))+')\n')
            rule['command']=['{python}',str(script),'{output}']
        config['tool_followup_plans']=[second]
        return config

    def test_two_tools_use_scheduled_identity_and_resume_without_reexecution(self):
        config=self.config();path=self.fixture.root/'schedule.json';write_json(path,config)
        # Reuse the full two-tool control-flow fixture (including resume,
        # source inheritance, remeasurement and source-tampering rejection).
        with patch.object(self.fixture,'config',return_value=path):
            self.fixture.test_two_tool_updates_inherit_validated_source_and_remeasure_all_obligations()
        root=self.fixture.campaign
        first=read_json(root/'episodes/tool-1/campaign/evolution-auto.json')
        second=read_json(root/'episodes/tool-2/campaign/evolution-auto.json')
        self.assertNotEqual(first['plan']['identity'],second['plan']['identity'])
        self.assertEqual(second['plan']['budget'],30)
        model=read_json(root/'episodes/model-3/campaign/campaign.json')
        self.assertEqual(model['diagnostic_tool']['registration']['validated_conditions'],second['plan']['identity'])

    def test_invalid_later_plan_fails_before_project_creation(self):
        config=self.config()
        for followups in [[],[{},{}],[{}],[None]]:
            with self.subTest(followups=followups):
                with self.assertRaises(ContractError):
                    ResearchProject.init(self.fixture.campaign,{**config,'tool_followup_plans':followups})
                self.assertFalse(self.fixture.campaign.exists())
        config['tool_followup_plans'][0]['stages']['correctness']['command']=['{python}','/missing-validator.py']
        with self.assertRaises(ContractError):ResearchProject.init(self.fixture.campaign,config)
        self.assertFalse(self.fixture.campaign.exists())

    def test_later_validator_frozen_before_first_model_and_selection_does_not_mutate(self):
        config=self.config();config['model']['measurement_discovery']={'development_budget':20}
        ResearchProject.init(self.fixture.campaign,config)
        project=ResearchProject(self.fixture.campaign,agent=object());self.addCleanup(project.store.close)
        model=project.model_config({'model_index':1,'tool_updates':1},self.fixture.root/'next-model')
        self.assertEqual(model['measurement_discovery']['development_budget'],30)
        frozen=read_json(self.fixture.campaign/'project.json')
        selected=select(frozen,1,verify=True);selected['development_ids'].append('mutation')
        self.assertNotIn('mutation',select(frozen,1)['development_ids'])
        script=Path(config['tool_followup_plans'][0]['stages']['correctness']['command'][1])
        script.write_text(script.read_text()+'\n# changed after project admission\n')
        with self.assertRaisesRegex(ContractError,'scheduled tool validator changed'):select(frozen,1,verify=True)
        with self.assertRaisesRegex(ContractError,'exhausted'):select(frozen,2)

    def test_optimizer_handoff_uses_next_tool_plan_budget(self):
        config=self.config()
        config['harness']=read_json(self.fixture.config('harness'))['harness']
        ResearchProject.init(self.fixture.campaign,config)
        project=ResearchProject(self.fixture.campaign,agent=object());self.addCleanup(project.store.close)
        source=self.fixture.campaign/'episodes/model-prior/campaign'
        write_json(source/'campaign.json',{'task_id':'prior','research_policy':{}})
        state={'last_model':str(source.relative_to(self.fixture.campaign)),'tool_updates':1,'harness_updates':0}
        child=project.evolve_config(state,self.fixture.root/'next-harness','harness')
        self.assertEqual(child['optimizer_handoff_budget'],30)

    def test_builtin_conditions_are_excluded_from_subsequent_plans(self):
        import test_validation_preflight
        from rec_harness.project_tool_plans import preflight
        fixture=test_validation_preflight.ValidationPreflightTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        first=fixture.plan();first['development_ids']=[]
        second=copy.deepcopy(first)
        config={'tool':{'plan':first},'tool_followup_plans':[second]}
        with self.assertRaisesRegex(ContractError,'overlap development/history'):preflight(config)
        self.assertIn('development',second['development_ids'])

    def test_followup_paths_resolve_relative_to_project_configuration(self):
        from rec_harness.autonomous import configuration
        config=self.config();rule=config['tool_followup_plans'][0]['stages']['correctness']
        expected=rule['command'][1];rule['command'][1]=Path(expected).name
        path=self.fixture.root/'project-config.json';write_json(path,config)
        self.assertEqual(configuration(path)['tool_followup_plans'][0]['stages']['correctness']['command'][1],expected)
