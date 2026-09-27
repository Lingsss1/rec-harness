from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rec_harness.core import ContractError,read_json,write_json
from rec_harness.jailed_agent import JailedCodexAgent


class JailedAgentEnvironmentTests(unittest.TestCase):
    def test_empty_parent_environment_still_sets_staged_home_and_retains_failed_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);template=root/'template';write_json(template/'jail-manifest.json',{})
            agent=JailedCodexAgent(template,root/'jails',auth_root=root/'empty-auth')
            receipt={'status':'crash','returncode':1,'wall_seconds':0.1}
            def launch(argv,cwd,directory,**kwargs):
                self.assertEqual(kwargs['env']['HOME'],'/root')
                self.assertEqual(kwargs['env']['CODEX_HOME'],'/root/.codex')
                jail=Path(argv[4])
                self.assertTrue((jail/'root/.codex').is_dir())
                (directory/'stdout.jsonl').write_text('')
                return receipt
            with patch.dict('os.environ',{},clear=True),patch('os.geteuid',return_value=0), \
                 patch('os.chown'),patch('os.mknod',side_effect=lambda path,*args:Path(path).touch()), \
                 patch('rec_harness.jailed_agent.run_process',side_effect=launch):
                with self.assertRaisesRegex(ContractError,'isolated Codex failed'):
                    agent.run('evaluation',{'public':'fixture'},root/'role')
            retained=read_json(root/'role/resources.json')
            self.assertEqual(retained['status'],'crash')
            self.assertEqual(retained['returncode'],1)
            self.assertIsNone(retained['token_usage'])
            self.assertEqual(retained['isolation'],read_json(root/'role/request.json')['isolation'])
