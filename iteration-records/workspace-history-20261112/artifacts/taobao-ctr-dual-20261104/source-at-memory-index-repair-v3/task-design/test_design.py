"""Adversarial tests for startup injection and evidence/ownership boundaries."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('design_validator', ROOT / 'validate_design.py')
design = importlib.util.module_from_spec(spec)
spec.loader.exec_module(design)


class BoundaryTests(unittest.TestCase):
    def test_inline_context_or_prompt_override_is_rejected(self):
        launch = json.loads((ROOT / 'launch.example.json').read_text())
        for extra in ['context', 'system_prompt', 'metrics', 'dataset', 'hypothesis']:
            with self.subTest(extra=extra), self.assertRaises(ValidationError):
                design.validate_launch({**launch, extra: 'injected task instructions'})

    def test_wrong_reference_type_is_rejected(self):
        launch = json.loads((ROOT / 'launch.example.json').read_text())
        with self.assertRaises(ValidationError):
            design.validate_launch({**launch, 'task_ref': 'https://example.org/task'})

    def test_accidental_task_prompt_in_fixed_agent_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'agent').mkdir()
            (root / 'agent/instructions.txt').write_text('Research this task. Optimize ROC-AUC for Taobao.')
            with self.assertRaisesRegex(ValueError, 'Task-specific term'):
                design.check_fixed_context(root, ['Taobao', 'ROC-AUC'])

    def test_external_report_cannot_be_marked_a_local_result(self):
        task = ROOT / 'tasks/taobao_display_ctr'
        schema = json.loads((task / 'industry/card.schema.json').read_text())
        card = json.loads((task / 'industry/cards.jsonl').read_text().splitlines()[0])
        for changed in [{**card, 'local_reproduction': True}, {**card, 'scientific_status': 'controlled_support'}]:
            with self.subTest(changed=changed['scientific_status']), self.assertRaises(ValidationError):
                Draft202012Validator(schema).validate(changed)

    def test_confirmed_winners_are_not_replaced_by_unrelated_or_ninth_place_sources(self):
        result = design.validate(ROOT)
        self.assertEqual(result['task_checks']['industry_cards'], 5)
        self.assertNotIn('FA-RankMixer', result['task_checks']['placements'])
        self.assertFalse(result['task_checks']['personal_enabled'])
        self.assertFalse(result['runnable_campaign'])


if __name__ == '__main__':
    unittest.main()
