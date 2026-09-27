import unittest
from rec_harness.core import Artifacts,ContractError,digest,read_json,write_json
from rec_harness.runtime import node
from rec_harness.store import Store
from rec_harness.adapters.amazon_contract_graph import attach_contracts
import test_amazon_training_evidence as fixtures


class ContractGraphTests(unittest.TestCase):
    setUp=fixtures.TrainingEvidenceTests.setUp
    save=fixtures.TrainingEvidenceTests.save
    def prepare_graph(self):
        self.cfg['protocol_hash']='p'
        job='candidate-tiny-42';write_json(self.root/'jobs'/job/'completed.json',self.result)
        store=Store(self.root/'graph/memory');self.addCleanup(store.close)
        store.patch({'base_version':0,'nodes':[node(job,'Experiment',{'valid':True},'source','p'),
            node('fact','Experiment',{'valid':True},'research','p')],'edges':[]})
        report={'rows':[{'model_version':'candidate','dataset':'tiny','seed':42}],
            'protocol_hash':'p','source_hashes':{'candidate':'source'},
            'source_receipts':{job:{'receipt_hash':digest(self.result)}}}
        report['hash']=digest(report)
        return store,report

    def test_contracts_are_archived_nonindependent_and_idempotent(self):
        store,report=self.prepare_graph()
        first=attach_contracts(self.root,store,self.cfg,report,'research','p','fact')
        self.assertEqual(first['added_nodes'],2);self.assertEqual(first['added_edges'],3)
        observation=next(n for n in store.snapshot()['nodes'] if n['id'] in first['observations'])
        attrs=observation['attrs'];self.assertFalse(attrs['independent_unit'])
        self.assertEqual(attrs['evidence_dependency_ids'],['candidate-tiny-42'])
        self.assertEqual(attrs['scientific_conclusion'],'unknown')
        self.assertNotIn('DO_NOT_EXPORT',str(attrs))
        artifacts=Artifacts(self.root/'graph/artifacts/objects')
        for ref in [attrs['summary_ref'],*attrs['report_refs'].values()]:artifacts.verify(ref)
        summary_path=self.root/'graph/research-readouts'/(first['observations'][0]+'.json')
        timestamp=summary_path.stat().st_mtime_ns
        before=store.verify()
        repeated=attach_contracts(self.root,store,self.cfg,report,'research','p','fact')
        self.assertEqual(repeated['added_nodes'],0);self.assertEqual(repeated['added_edges'],0)
        self.assertEqual(store.verify(),before)
        self.assertEqual(summary_path.stat().st_mtime_ns,timestamp)
        relation=store.snapshot()['edges'][0]
        relation['attrs']['evidence_refs']=[]
        store.patch({'base_version':store.version,'nodes':[],'edges':[relation]})
        with self.assertRaisesRegex(ContractError,'relation changed'):
            attach_contracts(self.root,store,self.cfg,report,'research','p','fact')

    def test_changed_receipt_rejected_without_graph_write(self):
        store,report=self.prepare_graph();before=store.verify()
        self.result['checkpoint_sha256']='changed'
        write_json(self.root/'jobs/candidate-tiny-42/completed.json',self.result)
        with self.assertRaisesRegex(ContractError,'receipt differs'):
            attach_contracts(self.root,store,self.cfg,report,'research','p','fact')
        self.assertEqual(store.verify(),before)
