import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import ContractError,digest
from rec_harness.store import Store


class ReadOnlyStoreTests(unittest.TestCase):
    def test_read_only_audit_preserves_projection_and_rejects_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);writer=Store(root);writer.event('original',{'value':1})
            expected=writer.verify();writer.close()
            projection=root/'events.jsonl';projection.write_text('interrupted projection retained')
            reader=Store(root,read_only=True)
            try:
                self.assertEqual(reader.verify(),expected)
                self.assertEqual(projection.read_text(),'interrupted projection retained')
                with self.assertRaisesRegex(ContractError,'read-only'):reader.event('forbidden',{})
                with self.assertRaisesRegex(ContractError,'read-only'):reader.export_events()
                with self.assertRaises(sqlite3.OperationalError):reader.db.execute("DELETE FROM events")
            finally:reader.close()
            writer=Store(root)
            self.assertIn('original',projection.read_text());writer.close()
            with self.assertRaises(sqlite3.OperationalError):Store(root/'missing',read_only=True)
            self.assertFalse((root/'missing').exists())

    def test_verification_uses_one_snapshot_during_concurrent_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer=Store(tmp);writer.event('first',{});before=writer.verify()
            reader=Store(tmp,read_only=True);appended=[]
            def concurrent_digest(value):
                if not appended:
                    appended.append(True);writer.event('second',{})
                return digest(value)
            try:
                with patch('rec_harness.store.digest',side_effect=concurrent_digest):
                    self.assertEqual(reader.verify(),before)
                self.assertEqual(reader.verify()['events'],2)
                writer.db.execute("UPDATE events SET payload='{}' WHERE seq=1")
                # Change a hashed field even though the original payload was empty.
                writer.db.execute("UPDATE events SET kind='tampered' WHERE seq=1")
                with self.assertRaisesRegex(ContractError,'event chain'):reader.verify()
            finally:reader.close();writer.close()
