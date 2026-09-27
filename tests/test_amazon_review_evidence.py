import tempfile
import unittest
from pathlib import Path

from rec_harness.core import Artifacts,ContractError,write_json
from rec_harness.store import Store
from rec_harness.adapters.amazon_review_evidence import admission_evidence,protocol_summary


class ReviewEvidenceTests(unittest.TestCase):
    def test_protocol_summary_preserves_audited_settings_and_excludes_private_payload(self):
        protocol=dict(epochs=200,batches=100,eval_users=1000,eval_seed=42,patience=4,eval_every=5)
        config={'protocol':protocol,'protocol_hash':'p','worker_sha256':'w','development_seeds':[42]}
        result={'worker_hash':'w','dataset':'tiny','seed':42,'user_ids':'PRIVATE'}
        manifest={'protocol':protocol,'worker_hash':'w','initialization':'fresh','checkpoint_path':'PRIVATE'}
        value=protocol_summary(config,result,manifest)
        self.assertEqual(value['protocol'],protocol);self.assertNotIn('PRIVATE',str(value))
        for bad in [{**result,'seed':17},{**result,'worker_hash':'other'}]:
            with self.assertRaises(ContractError):protocol_summary(config,bad,manifest)
        with self.assertRaises(ContractError):
            protocol_summary(config,result,{**manifest,'protocol':{**protocol,'eval_users':200}})
        self.assertEqual(protocol_summary(config,result,{})['status'],'unavailable')
        for key in ('cuda_matmul_allow_tf32','cudnn_allow_tf32'):
            protocol[key]=False
            for value in (None,True,0):
                manifest[key]=value
                with self.assertRaisesRegex(ContractError,'precision differs'):
                    protocol_summary(config,result,manifest)
            manifest[key]=False
            self.assertIs(protocol_summary(config,result,manifest)['protocol'][key],False)

    def fixture(self,root,**changes):
        source=root/'model.py';source.write_text('synthetic model')
        probe=root/'probe.json';write_json(probe,{'checks':'synthetic','private_path':'DO_NOT_SEND'})
        artifacts=Artifacts(root/'artifacts/objects')
        source_ref=artifacts.put(source);probe_ref=artifacts.put(probe)
        record={'proposal_id':'p','protocol_hash':'protocol','source_contract':{'contract':'model-v1'},
                'versions':{'candidate':{'source_ref':source_ref,'source':{'sha256':source_ref['sha256'],'path':str(source)},
                    'probe_ref':probe_ref,'gate':{'status':'success'},
                    'training_smoke':{'status':'success','report_ref':probe_ref}}},**changes}
        store=Store(root/'memory');store.event('amazon_model_candidate_admitted',record);store.close()
        return source,probe_ref

    def test_public_summary_is_stable_and_binds_candidate_and_archived_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,probe=self.fixture(root)
            first=admission_evidence(root,'p','protocol','candidate',source)
            self.assertEqual(first,admission_evidence(root,'p','protocol','candidate',source))
            self.assertEqual(first['status'],'verified')
            self.assertEqual(first['training_smoke']['report_sha256'],probe['sha256'])
            self.assertNotIn(tmp,str(first));self.assertNotIn('DO_NOT_SEND',str(first))
            source.write_text('modified candidate')
            with self.assertRaisesRegex(ContractError,'differs from admitted'):
                admission_evidence(root,'p','protocol','candidate',source)

    def test_control_uses_its_own_source_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,probe=self.fixture(root)
            store=Store(root/'memory')
            try:
                record=store.events()[0]['payload']
                control={**record['versions']['candidate'],'source_contract':{'contract':'control-specific','structural_changes':{'methods':['control-only-fixture']}}}
                store.event('amazon_model_candidate_admitted',{**record,'proposal_id':'control-plan','versions':{'control':control}})
            finally:store.close()
            value=admission_evidence(root,'control-plan','protocol','control',source)
            self.assertEqual(value['source_contract'],'control-specific')
            self.assertEqual(value['structural_changes'],{'methods':['control-only-fixture']})

    def test_rejects_protocol_identity_and_corrupted_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,probe=self.fixture(root)
            for protocol,candidate in [('other','candidate'),('protocol','other')]:
                with self.assertRaisesRegex(ContractError,'identity mismatch'):
                    admission_evidence(root,'p',protocol,candidate,source)
            (root/'artifacts/objects'/probe['sha256']).write_text('changed report')
            with self.assertRaises(ContractError):admission_evidence(root,'p','protocol','candidate',source)

    def test_legacy_absence_is_unavailable_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=Store(root/'memory');store.close()
            result=admission_evidence(root,'missing','protocol','candidate',root/'missing-source')
            self.assertEqual(result['status'],'unavailable')

    def test_failed_gate_cannot_be_reported_as_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source,_=self.fixture(root)
            store=Store(root/'memory');record=store.events()[0]['payload']
            record['versions']['candidate']['gate']['status']='failed'
            record['proposal_id']='failed';store.event('amazon_model_candidate_admitted',record);store.close()
            with self.assertRaisesRegex(ContractError,'numerical admission failed'):
                admission_evidence(root,'failed','protocol','candidate',source)
