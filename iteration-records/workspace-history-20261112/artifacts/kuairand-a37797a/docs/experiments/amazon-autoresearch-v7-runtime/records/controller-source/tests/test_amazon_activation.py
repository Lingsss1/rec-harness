import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rec_harness.core import Artifacts,ContractError
from rec_harness.store import Store
from rec_harness.adapters.amazon_activation import activation_context,cost_summary,validate_options


class ActivationContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        store=Store(self.root/'memory');self.addCleanup(store.close);store.configure_budget(10)
        self.options={'settings':{},'execution':{},'budget':2,'cost':1,'timeout':20,'datasets':['A'],'seeds':[42]}
        self.study={'datasets':['A'],'development_seeds':[42],'confirmation_seeds':[17,83],
            'model_versions':{'base':{'path':'PRIVATE_MODEL_PATH','sha256':'source'}}}
        self.controller=SimpleNamespace(root=self.root,study=self.root/'study',store=store,artifacts=Artifacts(self.root/'objects'),
            config={'activation_diagnostic':self.options},task='task',protocol_hash='protocol')

    def probe(self,config,directory):
        self.assertEqual(config['checkpoint_source']['seed'],42)
        store=Store(directory/'memory');store.configure_budget(1)
        if not store.events():store.reserve('call',1);store.settle('call',1,'success')
        store.close()
        return {'status':'success','measurement':{'measurements':{'block_0.activation_rms':1.},'limitations':['synthetic only']},
            'artifacts':{'checkpoint.pt':{'path':'PRIVATE_WEIGHTS','raw':'DO_NOT_SEND'}},'binding':{'private':'DO_NOT_SEND'}}

    def test_only_aggregates_reach_context_and_graph_and_cost_is_not_repeated(self):
        with patch('rec_harness.adapters.amazon_activation.run_probe',side_effect=self.probe):
            result=activation_context(self.controller,'base',self.study);before=self.controller.store.verify()
            self.assertEqual(activation_context(self.controller,'base',self.study),result)
            self.assertEqual(self.controller.store.verify(),before)
        for forbidden in ['PRIVATE_MODEL_PATH','PRIVATE_WEIGHTS','DO_NOT_SEND','checkpoint.pt']:
            self.assertNotIn(forbidden,str(result));self.assertNotIn(forbidden,str(self.controller.store.snapshot()))
        self.assertEqual(cost_summary(self.root)['charged_or_reserved'],1)
        self.assertEqual(len(self.controller.store.snapshot()['nodes']),1)

    def test_confirmation_seed_and_failed_probe_never_become_observations(self):
        self.options['seeds']=[17]
        with self.assertRaises(ContractError):validate_options(self.options,self.study)
        self.options['seeds']=[42]
        with patch('rec_harness.adapters.amazon_activation.run_probe',return_value={'status':'invalid','reason':'DO_NOT_SEND'}):
            result=activation_context(self.controller,'base',self.study)
        self.assertIn('measurement_failure',result);self.assertNotIn('DO_NOT_SEND',str(result))
        self.assertEqual(self.controller.store.version,0)

    def test_new_model_cannot_bypass_total_probe_envelope(self):
        self.options['budget']=1
        with patch('rec_harness.adapters.amazon_activation.run_probe',side_effect=self.probe) as runner:
            activation_context(self.controller,'base',self.study)
            self.study['model_versions']['new']={'path':'other','sha256':'other'}
            result=activation_context(self.controller,'new',self.study)
            self.assertIn('measurement_failure',result);self.assertEqual(runner.call_count,1)
