import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rec_harness.core import Artifacts,ContractError,file_hash,read_json,write_json
from rec_harness.store import Store
from rec_harness.adapters.amazon_history_context import development_history


class DevelopmentHistoryContextTests(unittest.TestCase):
    def test_event_bound_resume_and_aggregate_only_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/'study';study.mkdir();folder=root/'round-1';folder.mkdir()
            data=study/'tiny_train.txt';data.write_text('101 22\n101 23\n')
            cfg={'data':str(study),'data_hashes':{data.name:file_hash(data)},'datasets':['tiny'],
                 'development_seeds':[42],'confirmation_seeds':[17,83],'protocol_hash':'frozen'}
            binding={'base_model_id':'base','candidate_model_version':'candidate'}
            for model,ranks in [('base',[12]),('candidate',[1])]:
                write_json(study/'jobs'/f'{model}-tiny-42'/'validation_ranks.json',{'user_ids':[101],'ranks':ranks})
            def audit(study,cfg,model,dataset,seed):
                self.assertEqual(seed,42)
                return {'job_id':model,'execution_id':'same'},{'candidates_hash':'same','data_hashes':{}}
            store=Store(root/'memory');controller=SimpleNamespace(study=study,store=store,artifacts=Artifacts(root/'objects'))
            try:
                with patch('rec_harness.adapters.amazon_history_context.load_config',return_value=cfg),patch('rec_harness.adapters.history_diagnostic.load_config',return_value=cfg),patch('rec_harness.adapters.history_diagnostic.audit_job',side_effect=audit):
                    context=development_history(controller,folder,binding)
                    self.assertEqual(context['rows'][0]['slices'][0]['delta']['HR@10'],1)
                    self.assertEqual(set(context['rows'][0]),{'dataset','seed','slices'})
                    self.assertNotIn('101',str(context));self.assertNotIn('input_hashes',context)
                    before=store.verify()
                    with patch('rec_harness.adapters.history_diagnostic.diagnose',side_effect=AssertionError('duplicate diagnostic')):
                        self.assertEqual(development_history(controller,folder,binding),context)
                    self.assertEqual(store.verify(),before)
                    report_path=folder/'development-history/report.json';original=report_path.read_bytes()
                    report_path.write_bytes(original+b' ')
                    with self.assertRaisesRegex(ContractError,'projection changed'):development_history(controller,folder,binding)
                    report_path.write_bytes(original)
                    with self.assertRaisesRegex(ContractError,'binding changed'):
                        development_history(controller,folder,{**binding,'candidate_model_version':'other'})
                    data.write_text('101 99\n')
                    with self.assertRaisesRegex(ContractError,'input changed'):development_history(controller,folder,binding)
            finally:store.close()
