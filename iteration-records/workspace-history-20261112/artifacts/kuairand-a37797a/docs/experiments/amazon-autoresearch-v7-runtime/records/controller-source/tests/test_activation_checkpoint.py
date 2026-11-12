import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.activation_checkpoint import resolve_checkpoint
from rec_harness.core import ContractError


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.config={'source':{'sha256':'model'},'checkpoint_source':{'study':'/tmp/study','model_version':'base','dataset':'A','seed':42}}
        self.study={'development_seeds':[42],'confirmation_seeds':[17,83],'datasets':['A'],'model_versions':{'base':{'sha256':'model'}}}

    def test_only_audited_development_checkpoint_is_resolved(self):
        with patch('rec_harness.activation_checkpoint.load_config',return_value=self.study),patch('rec_harness.activation_checkpoint.audit_job',return_value=({'checkpoint_sha256':'weights'},{'dataset':'A','seed':42})) as audit:
            self.assertEqual(resolve_checkpoint(self.config),{'path':'/tmp/study/jobs/base-A-42/checkpoint.pt','sha256':'weights'})
            self.assertEqual(audit.call_args.args[-1],42)

    def test_confirmation_or_source_mismatch_rejected_before_audit(self):
        with patch('rec_harness.activation_checkpoint.load_config',return_value=self.study),patch('rec_harness.activation_checkpoint.audit_job') as audit:
            for seed in [17,83,True]:
                self.config['checkpoint_source']['seed']=seed
                with self.assertRaises(ContractError):resolve_checkpoint(self.config)
            self.config['checkpoint_source']['seed']=42;self.config['source']['sha256']='other'
            with self.assertRaises(ContractError):resolve_checkpoint(self.config)
            audit.assert_not_called()
