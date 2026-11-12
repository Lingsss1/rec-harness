import unittest
import test_amazon_report as fixtures

from rec_harness.adapters.amazon_report import paired_report
from rec_harness.adapters.amazon_summary import summarize_contrast, render
from rec_harness.core import ContractError, read_json, write_json


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PairedReportTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_missing_pairs_cannot_be_presented_as_complete(self):
        root = self.fixture.root
        (root / 'jobs/legacy_incumbent-tiny-83/completed.json').unlink()
        with self.assertRaisesRegex(ContractError, 'missing experiment pairs'):
            summarize_contrast(paired_report(root), 'HR@10')

    def test_negative_slice_retained_despite_positive_aggregate(self):
        # Audited report fixture, then vary a slice to exercise aggregation only.
        report = paired_report(self.fixture.root)
        report['partitions']['confirmation']['rows'][0]['delta']['HR@10'] = -0.25
        summary = summarize_contrast(report, 'HR@10')
        self.assertEqual(summary['partitions']['confirmation']['negative_slices'],
                         [{'dataset': 'tiny', 'seed': 17, 'effect': -0.25}])
        rendered = render({'protocol_hash': 'frozen', 'metric': 'HR@10', 'contrasts': [summary],
            'budget': {'limit': 100, 'charged_or_reserved': 80, 'remaining': 20},
            'hash': 'fixture', 'event_state': {'head': 'event-fixture'}})
        self.assertIn('-25.000 个百分点', rendered)
        self.assertIn('已计费或保留 80.000', rendered)
        self.assertIn('不代表新 Codex 候选收益', rendered)

    def test_forged_metrics_fail_before_summary(self):
        root = self.fixture.root
        path = root / 'jobs/legacy_incumbent-tiny-42/completed.json'
        receipt = read_json(path); receipt['metrics']['HR@10'] = 0.75
        write_json(path, receipt)
        with self.assertRaisesRegex(ContractError, 'rank-derived'):
            summarize_contrast(paired_report(root), 'HR@10')
