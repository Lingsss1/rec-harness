from contextlib import nullcontext
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
from rec_harness.core import Artifacts,ContractError,RoleTransportFailure,digest,read_json,write_json
from rec_harness.measurement_gaps import MeasurementDeferred
from rec_harness.store import Store


class AmazonTerminalRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);root=Path(self.tmp.name)
        r=self.runner=object.__new__(AmazonAutoresearch);r.root=root;r.study=root/'study'
        r.store=Store(root/'memory');self.addCleanup(r.store.close);r.store.configure_budget(12)
        r.artifacts=Artifacts(root/'artifacts/objects');r.lock=lambda:nullcontext()
        r.config={'max_rounds':3,'role_call_limit':12}
        ledger=Store(r.study/'memory');ledger.configure_budget(10000);ledger.close()
        config={'model_versions':{}};write_json(r.study/'study.json',config)
        write_json(r.study/'study-lock.json',{'config_hash':digest(config)})
        write_json(root/'state.json',{'round':0,'incumbent':'base','status':'ready'})

    def assert_terminal_recovery(self,deferred):
        r=self.runner
        def terminal(state,folder):
            folder.mkdir(parents=True,exist_ok=True)
            r.store.reserve('spent',12);r.store.settle('spent',12,'accepted')
            if deferred:raise MeasurementDeferred('fixture gap')
            raise RoleTransportFailure('fixture transport')
        original=r.store.event
        def interrupt(kind,value):
            if kind=='auto_round_completed':raise KeyboardInterrupt()
            return original(kind,value)
        with patch.object(r,'_round',side_effect=terminal) as work,patch.object(r.store,'event',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):r.run()
            self.assertEqual(work.call_count,1)
        with patch.object(r,'_round',side_effect=AssertionError('repeated round')):
            report=r.run()
            self.assertEqual(report['state']['stop_reason'],'measurement_required' if deferred else 'role_transport_failure')
            self.assertEqual(report['state']['round'],1)
            self.assertEqual(report['role_calls']['charged_or_reserved'],12)
            before=r.store.verify();self.assertEqual(r.run(),report);self.assertEqual(r.store.verify(),before)
        event=next(e for e in r.store.events() if e['kind']=='auto_round_completed')
        self.assertIn('recovered_terminal_event',event['payload'])

    def test_committed_pause_recovers_before_budget_checks(self):
        self.assert_terminal_recovery(True)

    def test_committed_transport_failure_recovers_before_budget_checks(self):
        self.assert_terminal_recovery(False)

    def test_generic_failure_consumes_round_without_repeating_work(self):
        r=self.runner;r.config['max_rounds']=1
        result={'status':'failed','reason':'invalid candidate'}
        r.store.event('auto_round_failed',{'round':1,**result})
        with patch.object(r,'_round',side_effect=AssertionError('repeated round')):
            report=r.run()
        self.assertEqual(report['state']['round'],1)
        self.assertEqual(report['state']['stop_reason'],'max_rounds')
        self.assertEqual(report['failed_rounds'],1)
        self.assertEqual(read_json(r.root/'rounds/round-1/failure.json'),result)
        self.assertEqual(report['role_calls']['charged_or_reserved'],0)

    def test_conflicting_outcomes_do_not_commit_completion(self):
        r=self.runner
        for reason in ['first','second']:
            r.store.event('auto_round_deferred',{'round':1,'status':'deferred','reason':reason,'selected':False})
        with self.assertRaisesRegex(ContractError,'conflicting'):r.run()
        self.assertFalse(any(e['kind']=='auto_round_completed' for e in r.store.events()))

    def test_completed_round_event_survives_state_projection_interruption(self):
        r=self.runner
        r.store.event('auto_round_deferred',{'round':1,'status':'deferred','reason':'gap','selected':False})
        original=write_json
        def interrupt(path,value):
            if Path(path)==r.root/'state.json':raise KeyboardInterrupt()
            return original(path,value)
        with patch('rec_harness.adapters.amazon_autonomous.write_json',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):r.run()
        result=r.run();self.assertEqual(result['state']['stop_reason'],'measurement_required')
        self.assertEqual(sum(e['kind']=='auto_round_completed' for e in r.store.events()),1)
