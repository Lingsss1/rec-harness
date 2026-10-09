import tempfile
import unittest
from pathlib import Path
from rec_harness.core import ContractError,read_json
from rec_harness.generation_workspace import capture,archive_files


class GenerationWorkspaceTests(unittest.TestCase):
    def test_drafts_retained_without_following_links_or_admitting_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);workspace=root/'workspace';directory=root/'role'
            (workspace/'model/nested').mkdir(parents=True);directory.mkdir()
            (workspace/'model/nested/draft.py').write_bytes(b'broken python\x00')
            (workspace/'response.json').write_text('not JSON')
            outside=root/'outside';outside.mkdir();(outside/'private').write_text('not to be copied')
            (workspace/'model/link.py').symlink_to(outside/'private')
            (workspace/'model/linked-directory').symlink_to(outside,target_is_directory=True)
            capture(workspace,directory)
            manifest=read_json(directory/'returned-workspace/manifest.json')
            self.assertEqual(set(manifest['files']),{'model/nested/draft.py','response.json'})
            self.assertEqual({x['path'] for x in manifest['excluded']},{'model/link.py','model/linked-directory'})
            self.assertEqual(set(archive_files(directory)),{'returned-workspace/manifest.json',
                'returned-workspace/model/nested/draft.py','returned-workspace/response.json'})
            self.assertFalse((directory/'model').exists())
            with self.assertRaisesRegex(ContractError,'already captured'):capture(workspace,directory)
            (directory/'returned-workspace/model/nested/draft.py').write_text('changed')
            with self.assertRaisesRegex(ContractError,'file changed'):archive_files(directory)

    def test_no_returned_workspace_preserves_legacy_archive(self):
        with tempfile.TemporaryDirectory() as tmp:self.assertEqual(archive_files(tmp),[])
