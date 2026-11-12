import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from rec_harness.core import ContractError,digest,write_json
from rec_harness.store import Store
from rec_harness.project_report import project_report
from rec_harness.cli import main


class ProjectReportTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        config={'task_id':'report-test'}
        write_json(self.root/'project.json',config)
        write_json(self.root/'project-lock.json',{'config_hash':digest(config)})
        store=Store(self.root/'memory');store.configure_budget(0)
        store.event('project_transition',{'state':{'phase':'model','status':'ready'}});store.close()
        # Stale projection is deliberately not authoritative.
        write_json(self.root/'state.json',{'status':'stopped'})
        self.child=self.root/'episodes/tool-1/campaign/validation/online/memory'
        store=Store(self.child);store.configure_budget(10)
        store.reserve('role-1',1);store.event('agent_submitted',{'task_id':'role-1','role':'evidence'})
        store.close()

    def test_pending_uncommitted_child_is_included_without_recovery(self):
        report=project_report(self.root)
        self.assertEqual(report['state']['status'],'ready')
        self.assertEqual(report['committed_episodes'],[])
        self.assertEqual(report['recorded_submissions'],1)
        self.assertEqual(report['pending_reservations'],1)
        self.assertEqual(report['ledgers'][1]['episode'],'tool-1')
        self.assertEqual(project_report(self.root),report)
        store=Store(self.child,read_only=True)
        try:self.assertEqual(store.budget()['charged_or_reserved'],1)
        finally:store.close()

    def test_external_ledger_rejected(self):
        with tempfile.TemporaryDirectory() as external:
            store=Store(Path(external)/'memory');store.configure_budget(1);store.close()
            alias=self.root/'episodes/model-2/memory';alias.mkdir(parents=True)
            (alias/'memory.sqlite').symlink_to(Path(external)/'memory/memory.sqlite')
            with self.assertRaisesRegex(ContractError,'outside project'):project_report(self.root)

    def test_prefix_completion_separate_and_markdown_cli(self):
        store=Store(self.child)
        store.event('prefix_replay_completed',{'key':'fixture'})
        store.close()
        report=project_report(self.root)
        self.assertEqual(report['prefix_completions'],1)
        self.assertEqual(report['recorded_submissions'],1)
        output=io.StringIO()
        with redirect_stdout(output):self.assertEqual(main(['project-report',str(self.root),'--format','markdown']),0)
        self.assertIn('| tool-1 |',output.getvalue())
        self.assertIn('External Amazon study/GPU ledgers are excluded',output.getvalue())

    def test_tampered_budget_projection_rejected(self):
        store=Store(self.child)
        store.db.execute("UPDATE budget SET reserved=9 WHERE id='role-1'");store.db.commit();store.close()
        with self.assertRaises(ContractError):project_report(self.root)
