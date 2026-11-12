import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import Artifacts,ContractError,file_hash,write_json
from rec_harness.cpu_job import recover_completed_cpu_job


class CPURecoveryTests(unittest.TestCase):
    def test_missing_failed_live_or_differently_bound_process_cannot_recover(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'worker.py';source.write_text('source')
            files={'worker.py':{'path':str(source),'sha256':file_hash(source)}}
            job=root/'job';request={'inputs':files,'entry':'worker.py','arguments':[],'recipe':{}}
            write_json(job/'request.json',request);artifacts=Artifacts(root/'objects')
            def recover():return recover_completed_cpu_job(job,{},files,'worker.py',[],artifacts=artifacts)
            with self.assertRaisesRegex(ContractError,'successful CPU process'):recover()
            write_json(job/'process/process.json',{'status':'crash'})
            with self.assertRaisesRegex(ContractError,'successful CPU process'):recover()
            write_json(job/'process/process.json',{'status':'success'})
            with patch('rec_harness.cpu_job.assert_stopped',side_effect=ContractError('worker session still live')):
                with self.assertRaisesRegex(ContractError,'still live'):recover()
            write_json(job/'request.json',{**request,'arguments':['changed']})
            with self.assertRaisesRegex(ContractError,'binding changed'):recover()
            self.assertFalse((job/'receipt.json').exists())
