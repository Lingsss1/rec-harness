import tempfile
import unittest
from pathlib import Path
from rec_harness.core import ContractError,file_hash
from rec_harness.evolution import freeze_validator_files


class ValidatorBindingTests(unittest.TestCase):
    def test_module_validator_binds_implementation_and_dependencies_not_only_suite(self):
        with tempfile.TemporaryDirectory() as root:
            suite=Path(root)/'suite.json';suite.write_text('{}')
            hashes=freeze_validator_files({'utility':{'command':['{python}','-m','rec_harness.tool_online','--suite',str(suite)]}})
            package=Path(__file__).resolve().parents[1]/'rec_harness'
            for name in ['tool_online.py','runtime.py','cpu_job.py','toy_isolated.py']:
                self.assertEqual(hashes[str(package/name)],file_hash(package/name))
            self.assertEqual(hashes[str(suite)],file_hash(suite))

    def test_external_module_requires_declared_files_and_rejects_missing_dependency(self):
        rule={'command':['{python}','-m','external_validator']}
        with self.assertRaisesRegex(ContractError,'explicit validator_files'):freeze_validator_files({'stage':rule})
        with tempfile.TemporaryDirectory() as root:
            source=Path(root)/'validator.py';source.write_text('version = 1\n')
            rule['validator_files']=[str(source)]
            frozen=freeze_validator_files({'stage':rule});source.write_text('version = 2\n')
            self.assertNotEqual(frozen[str(source)],file_hash(source))
            source.unlink()
            with self.assertRaisesRegex(ContractError,'dependency missing'):freeze_validator_files({'stage':rule})
        with self.assertRaisesRegex(ContractError,'unknown built-in'):
            freeze_validator_files({'stage':{'command':['{python}','-m','rec_harness.not_present']}})
