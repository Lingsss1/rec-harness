"""Case-level receipt recovery in both offline and shadow validators."""
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.store import Store
import test_tool_validation
import test_tool_shadow


class ValidationCaseRecoveryTests(unittest.TestCase):
    def fixture(self,kind):
        if kind=='reference':
            fixture=test_tool_validation.ToolReferenceTests();fixture.setUp()
            from rec_harness.tool_validation import run_validation
            call=lambda:run_validation(fixture.root/'suite.json',fixture.root/'candidate',fixture.root/'result.json','correctness')
            worker=fixture.worker;module='tool_validation';event='reference_case_completed';key='positive';count=2
        else:
            fixture=test_tool_shadow.ShadowTests();fixture.setUp()
            from rec_harness.tool_shadow import freeze_suite,run_validation
            freeze_suite(fixture.root/'spec.json',fixture.root/'suite')
            call=lambda:run_validation(fixture.root/'suite/suite.json',fixture.root/'candidate',fixture.root/'result.json')
            worker=fixture.execute;module='tool_shadow';event='shadow_measurement_completed';key='A-0';count=4
        self.addCleanup(fixture.doCleanups)
        def execute(directory,recipe,files,entry,arguments,**kwargs):
            receipt=worker(directory,recipe,files,entry,arguments,**kwargs)
            write_json(Path(directory)/'request.json',{'inputs':files,'recipe':recipe,'entry':entry,'arguments':arguments})
            write_json(Path(directory)/'process/process.json',receipt['process'])
            write_json(Path(directory)/'receipt.json',receipt)
            return receipt
        return fixture,call,execute,'rec_harness.'+module+'.run_cpu_job',event,key,count

    def test_completed_receipt_before_event_is_reused_and_mutable_copy_ignored(self):
        for kind in ['reference','shadow']:
            with self.subTest(kind=kind):
                fixture,call,execute,tool,event,key,count=self.fixture(kind)
                original=Store.event
                def interrupt(store,name,value):
                    if name==event:raise KeyboardInterrupt()
                    return original(store,name,value)
                with patch(tool,side_effect=execute) as worker:
                    with patch.object(Store,'event',interrupt):
                        with self.assertRaises(KeyboardInterrupt):call()
                    self.assertEqual(worker.call_count,1)
                    path=fixture.root/(kind+'-validation')/'cases'/key/'job-1/results/measurement.json'
                    write_json(path,{'measurements':{'changed':999},'limitations':['not authoritative']})
                    result=call();self.assertTrue(result['passed'])
                    self.assertEqual(result['total_cost'],count)
                    self.assertEqual(worker.call_count,count)
                    self.assertEqual(call(),result);self.assertEqual(worker.call_count,count)
                self.assertFalse(list((fixture.root/(kind+'-validation')).glob('cases/*/job-2')))

    def test_unknown_outcome_is_not_relaunched_or_recharged(self):
        for kind in ['reference','shadow']:
            with self.subTest(kind=kind):
                fixture,call,execute,tool,event,key,count=self.fixture(kind)
                def interrupt(directory,recipe,files,entry,arguments,**kwargs):
                    write_json(Path(directory)/'request.json',{'inputs':files,'recipe':recipe,'entry':entry,'arguments':arguments})
                    raise KeyboardInterrupt()
                with patch(tool,side_effect=interrupt) as worker:
                    with self.assertRaises(KeyboardInterrupt):call()
                    with self.assertRaisesRegex(ContractError,'completed successful'):call()
                    self.assertEqual(worker.call_count,1)
                store=Store(fixture.root/(kind+'-validation')/'memory',read_only=True)
                try:self.assertEqual(store.budget()['charged_or_reserved'],1)
                finally:store.close()

    def test_changed_process_binding_rejected_without_new_job(self):
        for kind in ['reference','shadow']:
            with self.subTest(kind=kind):
                fixture,call,execute,tool,event,key,count=self.fixture(kind);original=Store.event
                def interrupt(store,name,value):
                    if name==event:raise KeyboardInterrupt()
                    return original(store,name,value)
                with patch(tool,side_effect=execute),patch.object(Store,'event',interrupt):
                    with self.assertRaises(KeyboardInterrupt):call()
                path=fixture.root/(kind+'-validation')/'cases'/key/'job-1/request.json'
                value=read_json(path);value['arguments']=['changed'];write_json(path,value)
                with patch(tool,side_effect=AssertionError('must not relaunch')):
                    with self.assertRaisesRegex(ContractError,'binding changed'):call()

    def test_unbound_legacy_attempt_requires_review(self):
        for kind in ['reference','shadow']:
            with self.subTest(kind=kind):
                fixture,call,execute,tool,event,key,count=self.fixture(kind)
                (fixture.root/(kind+'-validation')/'cases'/key/'job-1').mkdir(parents=True)
                with patch(tool,side_effect=AssertionError('must not relaunch')):
                    with self.assertRaisesRegex(ContractError,'requires review'):call()
