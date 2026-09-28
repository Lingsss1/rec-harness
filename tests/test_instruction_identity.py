import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from rec_harness.agents import CodexAgent,schema_for
from rec_harness.core import Artifacts,ContractError,read_json,write_json
from rec_harness.instruction_identity import manifest,from_archive


class InstructionIdentityTests(unittest.TestCase):
    def test_transport_records_exact_prompt_and_schema(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def execute(argv,cwd,directory,**options):
                request=read_json(root/'request.json')
                self.assertEqual(request['prompt'],options['stdin'])
                write_json(root/'response.json',{'changed_files':[],'implementation_checks':[],'limitations':[]})
                (root/'stdout.jsonl').write_text('')
                return {'status':'success'}
            with patch('rec_harness.agents.run_process',side_effect=execute):
                CodexAgent(prompts={'tool':'A frozen override'}).run('tool',{},root)
            request=read_json(root/'request.json')
            self.assertIn('A frozen override',request['prompt'])
            self.assertEqual(request['instruction_identity'],manifest(request['prompt'],schema_for('tool')))
            artifacts=Artifacts(root/'objects');archive=artifacts.snapshot(root,['request.json','schema.json'])
            self.assertEqual(from_archive(archive,artifacts)['request_ref'],archive['request.json'])
            request['prompt']='modified';write_json(root/'request.json',request)
            archive=artifacts.snapshot(root,['request.json','schema.json'])
            with self.assertRaisesRegex(ContractError,'differs'):from_archive(archive,artifacts)

    def test_legacy_request_does_not_invent_instruction_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);write_json(root/'request.json',{'prompt':'legacy'})
            artifacts=Artifacts(root/'objects');archive=artifacts.snapshot(root,['request.json'])
            self.assertIsNone(from_archive(archive,artifacts))
