import tempfile
import os
import unittest
from pathlib import Path
from rec_harness.autonomous import configuration
from rec_harness.core import ContractError,write_json


class ConfigurationPathsTests(unittest.TestCase):
    def test_nested_evolution_paths_do_not_depend_on_launch_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/'tool/config.json',{'adapter':'research-evolution-v1',
                'agent_backend':{'template':'jail','runtime_root':'runs'},
                'plan':{'stages':{'utility':{'validator_files':['contract.json'],
                    'command':['python3','-m','rec_harness.tool_pipeline','run','--suite','suite.json',
                               '--candidate','{candidate}','--output','{output}']}}}})
            write_json(root/'model/config.json',{'source':'.','execution':{'uid_registry':'uids'},
                'agent_backend':{'template':'jail','runtime_root':'runs'}})
            write_json(root/'project.json',{'adapter':'research-project-v1','model_config':'model/config.json',
                'tool_config':'tool/config.json','tool_runtime':{'execution':{'uid_registry':'project-uids'}}})
            before=configuration(root/'project.json');cwd=Path.cwd()
            try:
                os.chdir(root/'tool')
                after=configuration(root/'project.json')
            finally:os.chdir(cwd)
            self.assertEqual(before,after)
            rule=before['tool']['plan']['stages']['utility']
            self.assertEqual(rule['command'][5],str(root/'tool/suite.json'))
            self.assertEqual(rule['command'][-1],'{output}')
            self.assertEqual(rule['validator_files'],[str(root/'tool/contract.json')])
            self.assertEqual(before['tool']['agent_backend']['runtime_root'],str(root/'tool/runs'))
            self.assertEqual(before['model']['agent_backend']['runtime_root'],str(root/'model/runs'))
            self.assertEqual(before['model']['execution']['uid_registry'],str(root/'model/uids'))
            self.assertEqual(before['tool_runtime']['execution']['uid_registry'],str(root/'project-uids'))

    def test_custom_validator_arguments_not_reinterpreted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            command=['python3','validator.py','--suite','opaque-value','--output','{output}']
            write_json(root/'config.json',{'plan':{'stages':{'utility':{'command':command}}}})
            actual=configuration(root/'config.json')['plan']['stages']['utility']['command']
            self.assertEqual(actual[1],str(root/'validator.py'))
            self.assertEqual(actual[2:],command[2:])

    def test_nested_model_and_project_tool_recipes_use_their_own_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/'model/recipe.json',{'runtime_dirs':['model-runtime']})
            write_json(root/'recipe.json',{'runtime_dirs':['project-runtime']})
            write_json(root/'model/config.json',{'source':'.','execution':{'recipe_file':'recipe.json'},
                'diagnostic_registry':{'registry':'registry','execution':{'recipe_file':'recipe.json'}}})
            write_json(root/'project.json',{'adapter':'research-project-v1','model_config':'model/config.json',
                'tool_runtime':{'execution':{'recipe_file':'recipe.json'}}})
            config=configuration(root/'project.json')
            self.assertEqual(config['model']['execution']['recipe']['runtime_dirs'],['model-runtime'])
            self.assertEqual(config['model']['diagnostic_registry']['execution']['recipe'],config['model']['execution']['recipe'])
            self.assertEqual(config['tool_runtime']['execution']['recipe']['runtime_dirs'],['project-runtime'])
            self.assertEqual(config['model']['diagnostic_registry']['registry'],str(root/'model/registry'))
            self.assertNotIn('recipe_file',config['tool_runtime']['execution'])

    def test_conflicting_recipe_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'recipe.json',{})
            for owner in ['execution','diagnostic_registry','tool_runtime']:
                execution={'recipe':{},'recipe_file':'recipe.json'}
                config={owner:execution if owner=='execution' else {'execution':execution}}
                write_json(root/'config.json',config)
                with self.assertRaisesRegex(ContractError,'not both'):configuration(root/'config.json')
