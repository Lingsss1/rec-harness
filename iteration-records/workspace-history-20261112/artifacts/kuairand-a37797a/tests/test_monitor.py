import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rec_harness.core import write_json
from rec_harness.monitor import Monitor, snapshot, redact, export_html
from rec_harness.store import Store


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'campaign'
        (self.root / 'research').mkdir(parents=True)
        (self.root / 'study').mkdir()
        write_json(self.root / 'research/amazon-auto.json', {'max_rounds': 20, 'review_revision_limit': 2})
        write_json(self.root / 'study/study.json', {'gpu_ids': [4, 5, 6], 'datasets': ['Electronics'], 'development_seeds': [42]})
        write_json(self.root / 'research/state.json', {'round': 0, 'status': 'ready'})
        write_json(self.root / 'progress.json', {'stage': 'autoresearch_running'})

    def tearDown(self):
        self.temp.cleanup()

    def role(self, relative, input_tokens, output_tokens):
        p = self.root / 'research/rounds' / relative
        p.mkdir(parents=True, exist_ok=True)
        write_json(p / 'resources.json', {'token_usage': {'input_tokens': input_tokens, 'cached_input_tokens': input_tokens // 2,
                                                         'output_tokens': output_tokens, 'reasoning_output_tokens': 3}})
        write_json(p / 'process.json', {'status': 'success'})
        return p

    def test_readonly_receipts_retries_and_archived_revisions(self):
        self.role('round-1/researcher', 100, 10)
        self.role('round-1/researcher-rejected-1', 50, 5)
        self.role('round-1/review-repair/revision-1/reviewer', 70, 7)
        write_json(self.root / 'research/rounds/round-1/review-repair/state.json', {'count': 1})
        self.role('round-1/researcher/model/analyst', 9999, 9999)
        store = Store(self.root / 'research/memory')
        store.configure_budget(200)
        store.reserve('predecessor-consumed', 99)
        store.settle('predecessor-consumed', 99, 'retained')
        store.event('auto_round_completed', {'state': {'round': 1, 'incumbent': 'baseline'}, 'result': {'candidate': 'candidate'}})
        store.event('auto_round_completed', {'state': {'round': 2, 'incumbent': 'baseline'}, 'result': {'status': 'failed', 'reason': 'blocked'}})
        before = store.events()
        result = snapshot(self.root, False)
        self.assertEqual(result['usage']['calls'], 3)
        self.assertEqual(result['usage']['input_tokens'], 220)
        self.assertEqual(result['usage']['total_tokens'], 242)
        self.assertEqual(result['usage']['uncached_input_tokens'], 110)
        self.assertEqual(result['rounds'][0]['revisions'], 1)
        self.assertEqual(result['counts']['completed'], 1)
        self.assertEqual(result['counts']['failed'], 1)
        self.assertEqual(result['role_budget']['inherited'], 99)
        self.assertEqual(store.events(), before)
        store.close()

    def test_dead_controller_overrides_saved_running_phase(self):
        with patch('rec_harness.monitor.live', return_value=False):
            result = snapshot(self.root, False)
        self.assertEqual(result['status'], 'process_stopped')

    def test_missing_usage_and_partial_worker_json_are_not_fabricated(self):
        p = self.root / 'research/rounds/round-1/researcher'
        p.mkdir(parents=True)
        (p / 'stdout.jsonl').write_text('{"type":"turn.completed","usage":')
        result = snapshot(self.root, False)
        self.assertEqual(result['usage']['calls_with_usage'], 0)
        self.assertIsNone(result['rounds'][0]['calls'][0]['usage']['input_tokens'])

    def test_known_campaigns_only_and_redaction(self):
        monitor = Monitor([self.root])
        with self.assertRaises(KeyError):
            monitor.get('../other')
        self.assertEqual(redact('ghp_example123 Bearer token123'), '[REDACTED] Bearer [REDACTED]')

    def test_worker_stdout_shows_epoch_before_receipt(self):
        p = self.root / 'study/jobs/candidate-Electronics-42'
        p.mkdir(parents=True)
        write_json(p / 'request.json', {'gpu': 4})
        out = p / 'isolated/process/stdout.jsonl'
        out.parent.mkdir(parents=True)
        out.write_text(json.dumps({'epoch': 5, 'loss': .3, 'metrics': {'HR@10': .5}}) + '\n{"epoch":')
        with patch('rec_harness.monitor.live', return_value=True):
            result = snapshot(self.root, False)
        self.assertEqual(result['jobs'][0]['epoch'], 5)
        self.assertEqual(result['jobs'][0]['status'], 'running')
        self.assertIsNone(result['jobs'][0]['hr10'])

    def test_standalone_html_is_self_contained_and_escapes_research_text(self):
        monitor = Monitor([self.root])
        dangerous = '</script><script>bad()</script>'
        destination = self.root / 'panel.html'
        with patch.object(monitor, 'get', return_value={'mechanism': dangerous}):
            export_html(monitor, destination)
        text = destination.read_text()
        self.assertNotIn(dangerous, text)
        self.assertIn('\\u003c/script\\u003e', text)
        self.assertIn('window.__AUTORESEARCH_SNAPSHOTS__=', text)
        self.assertNotIn('src="/app.js"', text)
        self.assertNotIn('href="/style.css"', text)
        self.assertFalse(destination.with_name('panel.html.tmp').exists())


if __name__ == '__main__':
    unittest.main()
