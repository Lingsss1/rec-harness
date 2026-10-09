import tempfile
import unittest
from pathlib import Path

from rec_harness.adapters.amazon_controls import CONTROL, OLD, NEW, control_source, load_config, prepare_control, register_control
from rec_harness.core import ContractError, digest, file_hash, write_json
from rec_harness.store import Store


class ControlRegistrationTests(unittest.TestCase):
    """Synthetic registration fixtures; they do not claim a model passed probes."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / 'incumbent.py'
        source.write_text('# synthetic text-only control fixture\n' + OLD + '\n')
        config = {'arms': {'legacy_incumbent': {'path': str(source), 'sha256': file_hash(source)}},
                  'protocol_hash': 'frozen-protocol', 'development_seeds': [42], 'confirmation_seeds': [17, 83]}
        write_json(self.root / 'study.json', config)
        write_json(self.root / 'study-lock.json', {'config_hash': digest(config)})
        self.original_config = (self.root / 'study.json').read_bytes()

    def probe(self):
        source = prepare_control(self.root)
        probe = {'source_hash': file_hash(source), 'checks': {k: True for k in
            ['causal_input_gradient', 'tail_readout', 'empty_history_finite', 'padding_gradient_zero']}}
        probe['checks']['parameter_gradients'] = {'fixture': {'finite': True}}
        probe['hash'] = digest(probe)
        path = self.root / 'probe.json'; write_json(path, probe)
        return path

    def test_exact_change_registered_without_mutating_frozen_protocol(self):
        probe = self.probe()
        result = register_control(self.root, probe)
        self.assertEqual(result['model_version'], CONTROL)
        self.assertEqual(result['seeds'], [42, 17, 83])
        config = load_config(self.root)
        self.assertIn(CONTROL, config['model_versions'])
        self.assertEqual((self.root / 'study.json').read_bytes(), self.original_config)
        self.assertTrue(register_control(self.root, probe)['already_registered'])
        store = Store(self.root / 'memory')
        try:
            self.assertEqual(sum(e['kind'] == 'amazon_control_registered' for e in store.events()), 1)
        finally: store.close()

    def test_unreviewed_change_or_missing_probe_cannot_register(self):
        probe = self.probe()
        source = self.root / 'frozen-controls' / (CONTROL + '.py')
        source.write_text(source.read_text() + 'extra_change = True\n')
        with self.assertRaisesRegex(ContractError, 'exceeds reviewed'):
            register_control(self.root, probe)
        with self.assertRaises(ContractError): control_source('arbitrary candidate code')
        with self.assertRaises(ContractError): control_source(OLD + '\n' + OLD)

    def test_source_mutation_after_registration_is_rejected(self):
        probe = self.probe(); register_control(self.root, probe)
        source = self.root / 'frozen-controls' / (CONTROL + '.py')
        source.write_text(source.read_text().replace(NEW, OLD))
        with self.assertRaisesRegex(ContractError, 'source changed'):
            load_config(self.root)

    def test_outcomes_before_preregistration_rejected(self):
        probe = self.probe()
        write_json(self.root / 'jobs' / (CONTROL + '-fake-42') / 'completed.json', {'status': 'success'})
        with self.assertRaisesRegex(ContractError, 'before preregistration'):
            register_control(self.root, probe)


if __name__ == '__main__':
    unittest.main()
