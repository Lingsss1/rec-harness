import ast
import json
from types import SimpleNamespace
import unittest

from .protocol_fixes import (bounded_history, compact_diagnosis,
                             diagnosis_presentation, restore_diagnosis_prompt)


class ProtocolFixTests(unittest.TestCase):
    def test_missing_presentation_is_restored_without_replacing_probe(self):
        source = 'class DiagnosisProbe:\n    marker = "evolved-probe"\n'
        repaired, changed = restore_diagnosis_prompt(source)
        namespace = {}
        exec(repaired, namespace)
        self.assertTrue(changed)
        self.assertEqual(namespace['DiagnosisProbe'].marker, 'evolved-probe')
        prompt = namespace['diagnosis_interpreter_prompt']({'metrics': {'margin': 1.25},
                                                          'metric_definitions': {'margin': 'observed margin'}})
        self.assertIn('1.25', prompt)
        self.assertIn('observed margin', prompt)
        self.assertEqual(restore_diagnosis_prompt(repaired), (repaired, False))

    def test_wrong_signature_is_repaired(self):
        repaired, changed = restore_diagnosis_prompt('def diagnosis_interpreter_prompt():\n    return "wrong"\n')
        namespace = {}
        exec(repaired, namespace)
        self.assertTrue(changed)
        self.assertIsInstance(namespace['diagnosis_interpreter_prompt']({'metrics': {}, 'metric_definitions': {}}), str)

    def test_nested_raw_examples_are_bounded_but_headlines_are_unchanged(self):
        raw = {'metrics': {'HR@10': .57, 'raw_examples': [
            {'user': i, 'observed': list(range(500))} for i in range(300)]},
            'metric_definitions': {'HR@10': 'Observed host HR@10', 'raw_examples': 'Per-user diagnostic observations'}}
        result, info = compact_diagnosis(raw, max_chars=10000)
        self.assertTrue(info['compacted'])
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 10000)
        self.assertEqual(result['metrics']['HR@10'], .57)
        self.assertEqual(result['metric_definitions'], raw['metric_definitions'])
        self.assertEqual(len(raw['metrics']['raw_examples']), 300)
        self.assertEqual(set(result['metrics']), set(raw['metrics']))

    def test_rendered_indentation_is_counted(self):
        module = SimpleNamespace(diagnosis_interpreter_prompt=lambda x: json.dumps(x, indent=4))
        raw = {'metrics': {'headline': 2, 'examples': [{'events': list(range(500))} for _ in range(1000)]},
               'metric_definitions': {'headline': 'measured', 'examples': 'observations'}}
        prompt, result, info = diagnosis_presentation(module, raw)
        self.assertLessEqual(len(prompt), 650000)
        self.assertEqual(result['metrics']['headline'], 2)
        self.assertTrue(info['compacted'])

    def test_history_trims_whole_old_messages_and_preserves_current_code(self):
        latest = {'role': 'user', 'content': 'Code: def scorer(x): return x\nexact SEARCH/REPLACE'}
        history = [{'role': 'user', 'content': 'old' * 1000},
                   {'role': 'assistant', 'content': 'old answer'}, latest]
        bounded, removed = bounded_history(history, max_chars=300)
        self.assertEqual(bounded, [latest])
        self.assertEqual(removed, 2)
        self.assertEqual(len(history), 3)
        with self.assertRaises(ValueError):
            bounded_history([{'role': 'user', 'content': 'current code' * 100}], max_chars=300)


if __name__ == '__main__':
    unittest.main()
