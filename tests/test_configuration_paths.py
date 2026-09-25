import tempfile
import unittest
from pathlib import Path
from rec_harness.autonomous import configuration
from rec_harness.core import ContractError,write_json


class ConfigurationPathsTests(unittest.TestCase):
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
