import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rec_harness.core import Artifacts, ContractError, write_json, read_json
from rec_harness.store import Store
from rec_harness.adapters.amazon_review_repair import revise, recover, verify


class ReviewRepairRecoveryTests(unittest.TestCase):
    def test_interrupted_archive_resumes_without_spending_revision_twice(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'round-1';folder.mkdir()
            for role in ['researcher','experiment','reviewer']:
                write_json(folder/role/'response.json',{'role':role})
            store=Store(root/'memory')
            self.addCleanup(store.close)
            controller=SimpleNamespace(store=store,artifacts=Artifacts(root/'objects'),config={'review_revision_limit':2})
            review={'verdict':'request_changes','blocking_findings':['bad check']}
            original=Path.rename;calls=[]
            def interrupted(source,target):
                calls.append(source)
                if len(calls)==2:raise OSError('interrupted move')
                return original(source,target)
            with patch.object(Path,'rename',interrupted):
                with self.assertRaisesRegex(OSError,'interrupted'):
                    revise(controller,folder,review)
            state=recover(controller,folder)
            self.assertEqual(state['count'],1)
            self.assertEqual(state['phase'],'ready')
            self.assertEqual(len([e for e in store.events() if e['kind']=='review_revision_started']),1)
            self.assertEqual(recover(controller,folder),state)
            verify(controller,folder)
            old=folder/'review-repair/revision-1/reviewer/response.json'
            old.write_text('{}')
            with self.assertRaisesRegex(ContractError,'archive changed'):
                recover(controller,folder)

    def test_limit_persists_and_registration_cannot_be_revised(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'round-1';folder.mkdir()
            store=Store(root/'memory');self.addCleanup(store.close)
            controller=SimpleNamespace(store=store,artifacts=Artifacts(root/'objects'),config={'review_revision_limit':2})
            review={'verdict':'unresolved','blocking_findings':['bad check']}
            for count in [1,2]:
                for role in ['researcher','experiment','reviewer']:
                    write_json(folder/role/'response.json',{'revision':count})
                self.assertEqual(revise(controller,folder,review)['count'],count)
            with self.assertRaisesRegex(ContractError,'after 2 revisions'):
                revise(controller,folder,review)
            fresh=root/'round-2';fresh.mkdir();write_json(fresh/'preregistration.json',{})
            with self.assertRaisesRegex(ContractError,'registered'):
                revise(controller,fresh,review)
