import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import ContractError,write_json,digest,file_hash
from rec_harness.store import Store
from rec_harness.adapters.amazon_campaign import campaign_guard
from rec_harness.adapters.amazon_final_selection import seal_selection,load_selection

class FinalSelectionTests(unittest.TestCase):
    def test_seal_uses_only_confirmed_receipts_and_prevents_research(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/'study';study.mkdir();source=root/'model.py';source.write_text('frozen source')
            cfg={'model_versions':{m:{'path':str(source),'sha256':file_hash(source)} for m in ['base','candidate']},
                 'datasets':['tiny'],'confirmation_seeds':[17,83],'protocol_hash':'p','data':'not-opened','data_hashes':{},'python':'/usr/bin/python3'}
            write_json(study/'study.json',cfg)
            store=Store(study/'memory');store.configure_budget(100);store.close();calls=[]
            def audit(root,config,model,dataset,seed):
                calls.append((model,seed));checkpoint=study/'jobs'/f'{model}-{dataset}-{seed}'/'checkpoint.pt'
                checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_bytes(b'frozen weights')
                return {'checkpoint_sha256':file_hash(checkpoint),'execution_id':'same'},{'itemnum':200}
            with patch('rec_harness.adapters.amazon_final_selection.load_config',return_value=cfg),patch('rec_harness.adapters.amazon_final_selection.audit_job',side_effect=audit):
                plan=seal_selection(study,root/'final',['base','candidate'],root/'nonexistent-test',eval_seed=99)
                self.assertEqual(len(calls),4)
                self.assertEqual(seal_selection(study,root/'final',['base','candidate'],root/'nonexistent-test',eval_seed=99),plan)
                self.assertEqual(len(calls),4)
                with self.assertRaisesRegex(ContractError,'differs'):seal_selection(study,root/'final',['candidate','base'],root/'nonexistent-test',eval_seed=99)
            self.assertEqual(load_selection(root/'final'),plan)
            with patch('rec_harness.adapters.independent_scoring.scoring_runtime',return_value={'version':'different'}):
                with self.assertRaisesRegex(ContractError,'scoring runtime changed'):load_selection(root/'final')
            self.assertIn('cpu_job.py',plan['package_code'])
            changed_code={**plan['package_code'],'cpu_job.py':'changed-recovery-code'}
            with patch('rec_harness.adapters.amazon_final_selection.package_code_hashes',return_value=changed_code):
                with self.assertRaisesRegex(ContractError,'package code changed'):load_selection(root/'final')
            store=Store(study/'memory')
            try:
                with self.assertRaisesRegex(ContractError,'sealed'):campaign_guard(cfg,store)
                store.event('unexpected_research_change',{})
            finally:store.close()
            with self.assertRaisesRegex(ContractError,'research changed'):load_selection(root/'final')
            with self.assertRaisesRegex(ContractError,'research changed'):
                seal_selection(study,root/'final',['base','candidate'],root/'nonexistent-test',eval_seed=99)

    def test_active_budget_cannot_be_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/'study';study.mkdir();store=Store(study/'memory');store.configure_budget(100);store.reserve('running',10);store.close()
            with self.assertRaisesRegex(ContractError,'unsettled'):
                seal_selection(study,root/'final',['base','candidate'],root/'test',eval_seed=1)
            self.assertFalse((root/'final').exists())
