"""Outer stage recovery composed with input-bound reference CPU receipts."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rec_harness.core import Artifacts,digest,write_json
from rec_harness.store import Store
from rec_harness.validation_stage import report,ValidationRecoveryRequired
import test_validation_case_recovery


class ValidationStageRecoveryTests(unittest.TestCase):
    def test_outer_resume_preserves_cases_and_accounts_for_wrapper_invocations(self):
        for unknown in [False,True]:
            with self.subTest(unknown=unknown):
                helper=test_validation_case_recovery.ValidationCaseRecoveryTests();self.addCleanup(helper.doCleanups)
                fixture,_,execute,tool,event,_,count=helper.fixture('reference')
                root=fixture.root/'outer';store=Store(root/'memory');self.addCleanup(store.close)
                store.configure_budget(10);artifacts=Artifacts(root/'objects')
                implementation=artifacts.snapshot(fixture.root/'candidate',['diagnostic.py'])
                candidate={'candidate_id':'candidate','plan_hash':'fixture',
                    'implementation':implementation,'implementation_hash':digest(implementation)}
                plan={'budget':10,'stages':{'correctness':{'cost':1,'timeout':5,
                    'command':['{python}','-m','rec_harness.tool_validation','--candidate','{candidate}',
                               '--output','{output}']}}}
                evolution=SimpleNamespace(root=root,store=store,artifacts=artifacts)
                from rec_harness.tool_validation import run_validation
                def runner(argv,cwd,directory,**kwargs):
                    run_validation(fixture.root/'suite.json',cwd/'candidate',cwd/'result.json','correctness')
                    write_json(directory/'process.json',{'status':'success'})
                    return {'status':'success'}
                original=Store.event
                def interrupt(store,kind,value):
                    if kind==event:raise KeyboardInterrupt()
                    return original(store,kind,value)
                def pending(directory,recipe,files,entry,arguments,**kwargs):
                    write_json(Path(directory)/'request.json',{'inputs':files,'recipe':recipe,'entry':entry,'arguments':arguments})
                    raise KeyboardInterrupt()
                with patch(tool,side_effect=pending if unknown else execute) as worker:
                    with patch.object(Store,'event',interrupt):
                        with self.assertRaises(KeyboardInterrupt):report(evolution,candidate,plan,'correctness',runner)
                    self.assertEqual(worker.call_count,1)
                    if unknown:
                        for _ in range(2):
                            with self.assertRaises(ValidationRecoveryRequired):
                                report(evolution,candidate,plan,'correctness',lambda *a,**kw:self.fail('relaunch'))
                        self.assertEqual(store.budget()['charged_or_reserved'],1)
                    else:
                        result=report(evolution,candidate,plan,'correctness',runner)
                        self.assertTrue(result['passed']);self.assertEqual(result['total_cost'],count)
                        self.assertEqual(worker.call_count,count)
                        self.assertEqual(store.budget()['charged_or_reserved'],2)
                        self.assertEqual(report(evolution,candidate,plan,'correctness',lambda *a,**kw:self.fail('relaunch')),result)
                stage=root/'candidates/candidate/validation/correctness'
                self.assertFalse(list(stage.glob('reference-validation/cases/*/job-2')))
                self.assertFalse((root/'candidates/candidate/validation-attempts').exists())
