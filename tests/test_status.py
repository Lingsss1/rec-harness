import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rec_harness.cli import main
from rec_harness.core import write_json
from rec_harness.status import inspect_status
from rec_harness.store import Store


class StatusTests(unittest.TestCase):
    def test_cli_reads_committed_round_without_repairing_stale_projections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'amazon-auto.json',{})
            old={'round':0,'incumbent':'base','status':'ready'}
            new={'round':1,'incumbent':'candidate','status':'ready'}
            write_json(root/'state.json',old)
            store=Store(root/'memory');store.configure_budget(10)
            store.event('auto_round_completed',{'state':new,'result':{}})
            before=store.verify();store.close()
            projection=root/'memory/events.jsonl';projection.write_text('interrupted export')
            with patch.object(Store,'export_events',side_effect=AssertionError('status mutated ledger')):
                output=io.StringIO()
                with contextlib.redirect_stdout(output):main(['status',str(root)])
                value=json.loads(output.getvalue())
                self.assertEqual(value['state'],new)
                self.assertEqual(value['state_projection'],'differs_from_event_snapshot')
                self.assertEqual(value['integrity'],before)
                self.assertEqual(value['process_status'],'not_inspected')
                (root/'state.json').unlink()
                self.assertEqual(inspect_status(root)['state'],new)
            self.assertEqual(projection.read_text(),'interrupted export')

    def test_terminal_overlay_is_explicit_and_reopen_event_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'amazon-auto.json',{})
            state={'round':1,'incumbent':'base','status':'ready'}
            store=Store(root/'memory');store.configure_budget(10)
            store.event('auto_round_completed',{'state':state})
            published={**state,'status':'stopped','stop_reason':'max_rounds'}
            write_json(root/'state.json',published)
            value=inspect_status(root)
            self.assertEqual(value['state'],published)
            self.assertEqual(value['state_source'],'committed_round_with_published_stop')
            reopened={**state,'round':0}
            store.event('auto_round_reopened',{'state':reopened});store.close()
            self.assertEqual(inspect_status(root)['state'],reopened)

    def test_project_transition_is_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_json(root/'project.json',{})
            state={'phase':'tool','status':'ready','model_index':1}
            store=Store(root/'memory');store.configure_budget(0)
            store.event('project_transition',{'state':state});store.close()
            write_json(root/'state.json',{'phase':'model','status':'ready','model_index':1})
            value=inspect_status(root)
            self.assertEqual(value['state'],state)
            self.assertEqual(value['state_source'],'committed_event')
            store=Store(root/'memory')
            failure={'campaign':'model-1','reason':'role_transport_failure'}
            store.event('project_child_transport_stopped',failure)
            self.assertEqual(inspect_status(root)['attention'],failure)
            store.event('project_transition',{'state':{**state,'phase':'model'}})
            self.assertIsNone(inspect_status(root)['attention'])
            store.close()
