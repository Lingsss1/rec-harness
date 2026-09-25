"""Controller contract tests; CPU process isolation has its own launcher suite."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rec_harness.core import Artifacts,ContractError,digest,write_json
from rec_harness.diagnostics import measure,validate_measurement
from rec_harness.runtime import node
from rec_harness.store import Store


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        root=Path(self.tmp.name);artifacts=Artifacts(root/'objects');store=Store(root/'memory')
        self.addCleanup(store.close);store.configure_budget(20)
        source=root/'diagnostic.py';source.write_text('def measure(x): return x\n')
        ref=artifacts.put(source);snapshot={'diagnostic.py':ref}
        tool={'candidate_id':'validated-tool','implementation':snapshot,'implementation_hash':digest(snapshot),
            'worker':ref,'execution':{'recipe':{},'python':'python3','uid_registry':'unused'},'cost':1,'timeout':10}
        self.research=SimpleNamespace(root=root,artifacts=artifacts,store=store,task='task',protocol_hash='protocol',config={'diagnostic_tool':tool})
        self.state={'incumbent':'root','baseline':{'runs':[{'seed':42,'valid':True,'metrics':{'quality':0.5},'private_path':'excluded'}]}}
        store.patch({'base_version':0,'nodes':[node('root','ModelVersion',{},'task','protocol')],'edges':[]})

    def execute(self,directory,*args,**kwargs):
        write_json(Path(directory)/'results/measurement.json',{'measurements':{'spread':0.5},'limitations':['aggregate only']})
        ref=self.research.artifacts.put(Path(directory)/'results/measurement.json')
        return {'process':{'status':'success'},'collection':{'artifacts':{'measurement.json':ref}}}

    def test_measurement_cached_with_versioned_graph_and_no_private_input(self):
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=self.execute) as runner:
            result=measure(self.research,self.state);before=self.research.store.verify()
            self.assertEqual(measure(self.research,self.state),result)
            self.assertEqual(runner.call_count,1);self.assertEqual(self.research.store.verify(),before)
        self.assertEqual(result['status'],'success');self.assertEqual(self.research.store.budget()['charged_or_reserved'],1)
        request=next((self.research.root/'research_archive/diagnostics').glob('*/request.json'))
        self.assertNotIn('private_path',request.read_text())
        self.assertIn('MEASURED_BY',{e['relation'] for e in self.research.store.snapshot()['edges']})

    def test_failure_remains_invalid_and_is_not_retried_into_positive_measurement(self):
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=ContractError('worker failed')) as runner:
            result=measure(self.research,self.state)
            self.assertEqual(result['status'],'invalid');self.assertNotIn('output',result)
            self.assertEqual(measure(self.research,self.state),result);self.assertEqual(runner.call_count,1)

    def test_host_rejects_nonfinite_or_authoritative_candidate_fields(self):
        for value in [{'measurements':{'x':float('nan')},'limitations':[]},
                      {'measurements':{'x':1},'limitations':[],'passed':True}]:
            with self.assertRaises(ContractError):validate_measurement(value)
