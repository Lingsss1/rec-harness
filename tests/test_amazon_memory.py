import tempfile
import unittest
from pathlib import Path
from rec_harness.adapters.amazon_memory import export_development_memory,validate_memory
from rec_harness.core import ContractError,digest,write_json

class DevelopmentMemoryTests(unittest.TestCase):
    def test_export_excludes_confirmation_results_and_preserves_failed_designs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study={'development_seeds':[42]};write_json(root/'study/study.json',study)
            write_json(root/'campaign/amazon-auto.json',{'study':str(root/'study'),'study_config_hash':digest(study),'protocol_hash':'p','task_id':'task'})
            first=root/'campaign/rounds/round-1';write_json(first/'handoff.json',{'cards':{'method':'model hypothesis'}})
            report={'schema':'amazon-development-selection-v1','protocol_hash':'p','rows':[{'seed':42}]};report['hash']=digest(report)
            write_json(first/'development-readout.json',report)
            write_json(first/'private-confirmation.json',{'metrics':'PRIVATE'})
            write_json(first/'private-confirmation-started.json',{})
            write_json(first/'failure.json',{'reason':'CONFIRMATION FAILURE'})
            second=root/'campaign/rounds/round-2';write_json(second/'handoff.json',{'cards':{'method':'failed implementation'}})
            write_json(second/'failure.json',{'reason':'invalid implementation'})
            pack=export_development_memory(root/'campaign',root/'memory.json')
            self.assertNotIn('PRIVATE',str(pack));self.assertNotIn('CONFIRMATION FAILURE',str(pack))
            self.assertIn('failed implementation',str(pack));self.assertIn('no mechanism refutation',str(pack))
            bad={**pack,'protocol_hash':'other'}
            with self.assertRaises(ContractError):validate_memory(bad,'p')
            report['rows']=[{'seed':17}];report['hash']=digest({k:v for k,v in report.items() if k!='hash'});write_json(first/'development-readout.json',report)
            with self.assertRaisesRegex(ContractError,'non-development seed'):export_development_memory(root/'campaign',root/'invalid.json')
