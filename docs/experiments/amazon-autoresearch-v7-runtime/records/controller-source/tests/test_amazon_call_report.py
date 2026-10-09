from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rec_harness.core import ContractError,digest,write_json
from rec_harness.store import Store
from rec_harness.cost_audit import audit_cost
from rec_harness.adapters.amazon_call_report import call_report,submission_cadence


class CallReportTests(unittest.TestCase):
    def test_cadence_distinguishes_bursts_from_training_gaps(self):
        times=['00:00:00','00:00:30','00:01:00','00:10:00']
        report=submission_cadence([{'submitted_at':'2026-09-27T'+t+'+00:00'} for t in times])
        self.assertEqual(report['span_seconds'],600)
        self.assertEqual(report['minimum_interval_seconds'],30)
        self.assertEqual(report['median_interval_seconds'],30)
        self.assertEqual(report['peak_submissions_in_60_seconds'],3)

    def test_cadence_does_not_invent_missing_intervals_or_clock_order(self):
        self.assertIsNone(submission_cadence([])['minimum_interval_seconds'])
        single=[{'submitted_at':'2026-09-27T00:00:00+00:00'}]
        self.assertIsNone(submission_cadence(single)['median_interval_seconds'])
        self.assertEqual(submission_cadence(single+single)['peak_submissions_in_60_seconds'],2)
        for attempts in [[{'submitted_at':'invalid'}],[{'submitted_at':'2026-09-27T00:00:00'}],
                         single+[{'submitted_at':'2026-09-26T23:00:00+00:00'}]]:
            report=submission_cadence(attempts)
            self.assertEqual(report['status'],'unknown')
            self.assertIsNone(report['peak_submissions_in_60_seconds'])

    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        cfg={'role_call_limit':20};write_json(self.root/'amazon-auto.json',cfg)
        write_json(self.root/'auto-lock.json',{'config_hash':digest(cfg)})
        self.store=Store(self.root/'memory');self.addCleanup(self.store.close);self.store.configure_budget(20)

    def submit(self,task,role,outcome=None):
        self.store.reserve(task,1)
        self.store.event('agent_submitted',{'task_id':task,'role':role})
        if outcome:
            self.store.settle(task,1,'success' if outcome=='role_completed' else 'failed')
            self.store.event(outcome,{'task':task,'role':role})

    def test_counts_retries_pending_inheritance_and_unattributed_separately(self):
        self.store.reserve('prior',8);self.store.settle('prior',8,'retained')
        self.submit('round-1-evidence-1','evidence','role_rejected')
        self.submit('round-1-evidence-2','evidence','role_completed')
        self.submit('round-2-lead-3','lead')
        self.store.reserve('before-submit-crash',2);self.store.settle('before-submit-crash',None,'interrupted')
        before=self.store.verify();projection=(self.root/'memory/events.jsonl').read_bytes()
        with patch.object(Store,'export_events',side_effect=AssertionError('report must be read-only')):
            report=call_report(self.root)
            self.assertEqual(report,call_report(self.root))
        self.assertEqual(report['budget']['charged_or_reserved'],13)
        self.assertEqual(report['local_submissions'],3)
        self.assertEqual(report['retained_cost'],8);self.assertEqual(report['unattributed_cost'],2)
        self.assertEqual(report['rows'][0]['submitted'],2)
        self.assertEqual(report['rows'][0]['rejected'],1);self.assertEqual(report['rows'][0]['completed'],1)
        self.assertEqual(report['rows'][1]['pending'],1)
        self.assertIsNone(report['internal_api_requests'])
        self.assertIsNone(report['attempts'][0]['completed_wall_seconds'])
        self.assertEqual(self.store.verify(),before)
        self.assertEqual((self.root/'memory/events.jsonl').read_bytes(),projection)

    def test_concurrent_new_submission_does_not_mix_snapshot(self):
        self.submit('round-1-lead-1','lead','role_completed')
        before=self.store.verify()
        def concurrent(store,limit):
            result=audit_cost(store,limit)
            self.submit('round-2-lead-2','lead')
            return result
        with patch('rec_harness.adapters.amazon_call_report.audit_cost',side_effect=concurrent):
            report=call_report(self.root)
        self.assertEqual(report['local_submissions'],1)
        self.assertEqual(report['source_integrity'],before)
        self.assertEqual(call_report(self.root)['local_submissions'],2)

    def test_corrupt_projection_and_orphan_outcomes_are_rejected(self):
        self.submit('round-1-lead-1','lead','role_completed')
        self.store.db.execute("UPDATE budget SET actual=0 WHERE id='round-1-lead-1'")
        with self.assertRaisesRegex(ContractError,'projection differs'):call_report(self.root)
        self.store.db.execute("UPDATE budget SET actual=1 WHERE id='round-1-lead-1'")
        self.store.event('role_completed',{'task':'not-submitted','role':'lead'})
        with self.assertRaisesRegex(ContractError,'lacks submission'):call_report(self.root)

    def test_reused_markers_do_not_add_submissions_or_cost(self):
        folder=self.root/'rounds/round-1/evidence'
        write_json(folder/'reuse.json',{'source':'fixture'})
        write_json(folder/'accepted.json',{'response_hash':'fixture'})
        with patch('rec_harness.adapters.amazon_auto_audit.verify_reused_role') as verify:
            report=call_report(self.root);verify.assert_called_once()
        self.assertEqual(report['local_submissions'],0)
        self.assertEqual(report['rows'][0]['reused'],1)
        self.assertEqual(report['budget']['charged_or_reserved'],0)
