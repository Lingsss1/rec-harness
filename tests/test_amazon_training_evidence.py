import tempfile
import unittest
from pathlib import Path
from rec_harness.core import ContractError,digest,write_json
from rec_harness.adapters.amazon_training_evidence import training_contract_evidence


class TrainingEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.cfg={'development_seeds':[42],'datasets':['tiny'],'model_versions':{'candidate':{'sha256':'source'}}}
        self.result={'status':'success','process':{'status':'success'},'model_version':'candidate','dataset':'tiny',
            'seed':42,'source_hash':'source','checkpoint_sha256':'checkpoint','contract_gates':{}}
        self.reports={}
        for stage in ['pre','post']:
            self.reports[stage]={'model_id':'candidate-tiny-42','source_hash':'source',
                'checkpoint_hash':'checkpoint' if stage=='post' else None,
                'checks':{key:True for key in ['causal_input_gradient','tail_readout','empty_history_finite','padding_gradient_zero']},
                'private_path':'DO_NOT_EXPORT'}
            self.save(stage)

    def save(self,stage):
        report=self.reports[stage];report['hash']=digest({k:v for k,v in report.items() if k!='hash'})
        write_json(self.root/'jobs/candidate-tiny-42'/(stage+'-contract')/'report.json',report)
        self.result['contract_gates'][stage]={'status':'success','report_hash':report['hash']}

    def evidence(self,seed=42):
        return training_contract_evidence(self.root,self.cfg,'candidate','tiny',seed,self.result)

    def test_fresh_and_trained_probes_are_bound_without_checkpoint_payload(self):
        value=self.evidence()
        self.assertEqual(value['stages']['pre']['weight_state'],'fresh_initialization')
        self.assertEqual(value['stages']['post']['weight_state'],'trained_development_checkpoint')
        self.assertEqual(value['stages']['post']['tensor_graph']['status'],'unavailable')
        self.assertNotIn('DO_NOT_EXPORT',str(value));self.assertNotIn('checkpoint_sha256',str(value))
        self.assertEqual(value,self.evidence())

    def test_rehashed_wrong_checkpoint_or_job_identity_rejected(self):
        for key,value in [('checkpoint_hash','other'),('model_id','other-job'),('source_hash','other')]:
            before=self.reports['post'][key]
            self.reports['post'][key]=value;self.save('post')
            with self.assertRaises(ContractError):self.evidence()
            self.reports['post'][key]=before;self.save('post')

    def test_confirmation_and_incomplete_gates_rejected(self):
        with self.assertRaisesRegex(ContractError,'development-only'):self.evidence(seed=17)
        self.result['contract_gates'].pop('post')
        with self.assertRaisesRegex(ContractError,'incomplete'):self.evidence()
        self.result.pop('contract_gates')
        self.assertEqual(self.evidence()['status'],'unavailable')
