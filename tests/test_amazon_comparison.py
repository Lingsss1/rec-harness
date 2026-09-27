import unittest
import tempfile

from rec_harness.adapters.amazon_comparison import contrast_facts
from rec_harness.core import ContractError
from rec_harness.runtime import node
from rec_harness.store import Store


class ComparisonFactsTests(unittest.TestCase):
    def fixture(self):
        config = {'protocol_hash': 'p', 'datasets': ['one', 'two'],
                  'development_seeds': [42], 'confirmation_seeds': [17, 83],
                  'protocol': {'metric': 'HR@10', 'minimum_effect': 0.005},
                  'model_versions': {'control': {'sha256': 'a'}, 'candidate': {'sha256': 'b'}}}
        partitions, results = {}, {}
        for partition in ['development', 'confirmation']:
            rows = []
            for seed in config[partition + '_seeds']:
                for ds, delta in [('one', -0.01), ('two', 0.05)]:
                    rows.append({'seed': seed, 'dataset': ds, 'baseline': {'HR@10': 0.5},
                                 'candidate': {'HR@10': 0.5 + delta}, 'delta': {'HR@10': delta}})
                    for version in ['control', 'candidate']:
                        results[(version, ds, seed)] = {'job_id': f'{version}-{ds}-{seed}'}
            partitions[partition] = {'complete': True, 'rows': rows}
        report = {'baseline': 'control', 'candidate': 'candidate', 'partitions': partitions}
        return config, report, results

    def test_independence_is_seed_aggregate_and_negative_slice_is_retained(self):
        config, report, results = self.fixture()
        facts = contrast_facts(config, report, results)
        self.assertEqual(len(facts), 3)
        self.assertEqual(len({f['independent_run_id'] for f in facts}), 3)
        self.assertAlmostEqual(facts[0]['effect'], 0.02)
        self.assertEqual(facts[0]['slices'][0]['delta']['HR@10'], -0.01)
        self.assertEqual(len(facts[0]['source_experiment_ids']), 4)
        results[('control', 'one', 42)]['task_id'] = 'retry-id'
        retried = contrast_facts(config, report, results)
        self.assertEqual(facts[0]['independent_run_id'], retried[0]['independent_run_id'])
        self.assertIn('retry-id', retried[0]['source_experiment_ids'])

    def test_missing_or_duplicate_dataset_cannot_qualify_as_complete_seed(self):
        config, report, results = self.fixture()
        report['partitions']['development']['rows'][1]['dataset'] = 'one'
        with self.assertRaisesRegex(ContractError, 'exactly the frozen'):
            contrast_facts(config, report, results)
        report['partitions']['development']['complete'] = False
        with self.assertRaisesRegex(ContractError, 'incomplete'):
            contrast_facts(config, report, results)

    def test_source_bound_seed_aggregates_support_claim_without_counting_reanalysis(self):
        config, report, results = self.fixture()
        facts = contrast_facts(config, report, results)
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            self.addCleanup(store.close)
            records = []
            for (version, dataset, seed), result in results.items():
                records.append(node(result['job_id'], 'Experiment', {
                    'valid': True, 'independent_run_id': result['job_id'],
                    'partition': 'development' if seed == 42 else 'confirmation',
                    'effect_gate_passed': False}, 'amazon-' + dataset, 'p'))
            for fact in facts:
                records.append(node(fact['comparison_id'], 'Experiment', {
                    **fact, 'kind': 'derived_paired_comparison', 'valid': True}, 'paired', 'p'))
            store.patch({'base_version': store.version, 'nodes': records, 'edges': []})
            confirmed = [fact for fact in facts if fact['partition'] == 'confirmation']
            claim = node('claim', 'Claim', {'status': 'supported', 'scope': 'paired',
                'evidence_refs': [fact['comparison_id'] for fact in confirmed]}, 'paired', 'p')
            store.patch({'base_version': store.version, 'nodes': [claim], 'edges': []})
            repeated = node('repeated', 'Experiment', {**confirmed[0],
                'kind': 'derived_paired_comparison', 'valid': True,
                'independent_run_id': 'renamed-seed'}, 'paired', 'p')
            store.patch({'base_version': store.version, 'nodes': [repeated], 'edges': []})
            claim['attrs']['evidence_refs'] = [confirmed[0]['comparison_id'], 'repeated']
            with self.assertRaisesRegex(ContractError, 'two independent'):
                store.patch({'base_version': store.version, 'nodes': [claim], 'edges': []})


if __name__ == '__main__':
    unittest.main()
