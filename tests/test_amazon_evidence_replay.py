import importlib.util
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('evidence_replay',ROOT/'scripts/replay_amazon_evidence_graph.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)


class EvidenceReplayTests(unittest.TestCase):
    def test_published_twenty_rounds_have_no_future_or_broken_references(self):
        result=replay.simulate()
        self.assertEqual(len(result['steps']),20)
        self.assertEqual(sum(a['readout_available'] for a in result['availability']),14)
        self.assertEqual(result['steps'][0]['repaired']['historical_nodes'],0)
        totals=result['totals']['repaired']
        self.assertEqual(totals['recent_readout_hits'],totals['recent_readout_opportunities'])
        for step in result['steps']:
            check=step['repaired']
            self.assertFalse(check['prefix_leaks']);self.assertFalse(check['broken_references'])
            self.assertFalse(check['fabricated_support_edges'])
            self.assertLessEqual(check['nodes'],64)
        # Round 20's readout only appears in the separate terminal read-back.
        last_hash=replay.read_json(ROOT/'docs/experiments/amazon-autoresearch-v5-v6-summary.json')['rounds'][-1]['readout_hash']
        self.assertNotIn(last_hash,result['steps'][-1]['available_readout_hashes'])

    def test_saturation_retains_recent_facts_and_discloses_omissions(self):
        result=replay.simulate(limit=16)
        totals=result['totals']['repaired']
        self.assertEqual(totals['max_nodes'],16)
        self.assertEqual(totals['recent_readout_hits'],totals['recent_readout_opportunities'])
        self.assertTrue(any(step['omissions'] for step in result['steps']))
        self.assertEqual(totals['prefix_leaks'],0)

    def test_replay_is_deterministic(self):
        self.assertEqual(replay.simulate()['hash'],replay.simulate()['hash'])
