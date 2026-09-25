import math
from pathlib import Path
import tempfile
import unittest

from rec_harness.adapters.amazon_report import paired_report
from rec_harness.core import ContractError, digest, file_hash, read_json, write_json


class PairedReportTests(unittest.TestCase):
    def test_pair_cannot_mix_execution_versions(self):
        directory = self.root / 'jobs/legacy_incumbent-tiny-42'
        result = read_json(directory / 'completed.json')
        result['execution_id'] = 'other-reviewed-execution'
        write_json(directory / 'completed.json', result)
        write_json(directory / 'request.json', {'execution_id': 'other-reviewed-execution'})
        with self.assertRaisesRegex(ContractError, 'mixes execution'):
            paired_report(self.root)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {'arms': {a: {'sha256': a} for a in ['fixed_baseline', 'legacy_incumbent']},
                       'worker_sha256': 'worker', 'datasets': ['tiny'], 'development_seeds': [42],
                       'confirmation_seeds': [17, 83], 'protocol_hash': 'protocol',
                       'data_hashes': {'tiny_train.txt': 'train', 'tiny_valid.txt': 'valid'},
                       'protocol': {'epochs': 5, 'batches': 1, 'eval_users': 1, 'eval_seed': 9,
                                    'patience': 2, 'eval_every': 1, 'cudnn_deterministic': False,
                                    'metric': 'HR@10', 'minimum_effect': 0.005}}
        write_json(self.root / 'study.json', self.config)
        write_json(self.root / 'study-lock.json', {'config_hash': digest(self.config)})
        for arm in self.config['arms']:
            for seed in [42, 17, 83]:
                directory = self.root / 'jobs' / f'{arm}-tiny-{seed}'
                write_json(directory / 'validation_candidates.json', [{'user_id': 1, 'candidates': list(range(100))}])
                rank = 11 if arm == 'fixed_baseline' else 0
                write_json(directory / 'validation_ranks.json', {'user_ids': [1], 'ranks': [rank]})
                (directory / 'checkpoint.pt').write_bytes(b'fixture checkpoint, not a training result')
                write_json(directory / 'correctness.json', {'causal_prefix': True, 'tail_readout': True})
                hashes = {'source_hash': arm, 'worker_hash': 'worker',
                          'candidates_hash': file_hash(directory / 'validation_candidates.json')}
                write_json(directory / 'manifest.json', {**hashes, 'seed': seed, 'dataset': 'tiny',
                    'data_hashes': {'train': 'train', 'validation': 'valid'}, 'protocol': self.config['protocol'],
                    'initialization': 'fresh', 'cudnn_deterministic': False})
                write_json(directory / 'completed.json', {**hashes, 'seed': seed, 'dataset': 'tiny', 'arm': arm,
                    'status': 'success', 'process': {'status': 'success'}, 'checkpoint_sha256': file_hash(directory / 'checkpoint.pt'),
                    'metrics': {'HR@10': float(rank < 10), 'NDCG@10': 1 / math.log2(rank + 2) if rank < 10 else 0}})

    def test_separate_seed_partitions_and_missing_pairs(self):
        report = paired_report(self.root)
        self.assertEqual(report['partitions']['development']['seed_effects'], {'42': 1})
        self.assertEqual(report['partitions']['confirmation']['seed_effects'], {'17': 1, '83': 1})
        (self.root / 'jobs/legacy_incumbent-tiny-83/completed.json').unlink()
        partial = paired_report(self.root)['partitions']['confirmation']
        self.assertFalse(partial['complete'])
        self.assertNotIn('effect_gate_passed', partial)

    def test_score_cannot_disagree_with_archived_ranks(self):
        path = self.root / 'jobs/legacy_incumbent-tiny-42/completed.json'
        receipt = read_json(path); receipt['metrics']['HR@10'] = 0.9; write_json(path, receipt)
        with self.assertRaisesRegex(ContractError, 'rank-derived'):
            paired_report(self.root)

    def test_pair_must_use_identical_candidates_even_if_each_hash_is_valid(self):
        directory = self.root / 'jobs/legacy_incumbent-tiny-17'
        write_json(directory / 'validation_candidates.json', [{'user_id': 1, 'candidates': list(range(1, 101))}])
        for name in ['manifest.json', 'completed.json']:
            path = directory / name; value = read_json(path)
            value['candidates_hash'] = file_hash(directory / 'validation_candidates.json'); write_json(path, value)
        with self.assertRaisesRegex(ContractError, 'unpaired'):
            paired_report(self.root)


if __name__ == '__main__':
    unittest.main()
