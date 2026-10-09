import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch as mock_patch
from rec_harness.adapters.amazon_cycle import validate_knowledge_patch
from rec_harness.core import ContractError
from rec_harness.store import Store


class KnowledgeBoundaryTests(unittest.TestCase):
    def fixture(self):
        graph={'graph_version':7,'task_id':'task','nodes':[{'id':'development','type':'Experiment',
            'attrs':{'task_id':'task','protocol_hash':'p','partition':'development'}}]}
        patch={'base_version':7,'nodes':[{'id':'interpretation','type':'Claim','attrs':{
            'task_id':'task','protocol_hash':'p','origin':'agent_hypothesis','status':'provisional',
            'scope':'development only','evidence_refs':['development']}}],'edges':[]}
        return graph,patch

    def test_only_new_scoped_explanations_can_cross_boundary(self):
        graph,patch=self.fixture();validate_knowledge_patch(patch,graph)
        for key,value,message in [('type','Experiment','explanation'),('id','development','explanation')]:
            bad=copy.deepcopy(patch);bad['nodes'][0][key]=value
            with self.assertRaisesRegex(ContractError,message):validate_knowledge_patch(bad,graph)
        for key,value,message in [('status','supported','provisional'),('origin','local_evidence','provenance'),
                                  ('evidence_refs',['unknown-confirmation'],'outside')]:
            bad=copy.deepcopy(patch);bad['nodes'][0]['attrs'][key]=value
            with self.assertRaisesRegex(ContractError,message):validate_knowledge_patch(bad,graph)

    def test_stale_graph_is_not_silently_rebased(self):
        graph,patch=self.fixture();patch['base_version']=6
        with self.assertRaisesRegex(ContractError,'base differs'):validate_knowledge_patch(patch,graph)

    def test_store_contract_failures_are_rejected_before_role_acceptance(self):
        graph,patch=self.fixture()
        for field in ['scope','evidence_refs']:
            bad=copy.deepcopy(patch);bad['nodes'][0]['attrs'].pop(field)
            with self.assertRaisesRegex(ContractError,'missing '+field):validate_knowledge_patch(bad,graph)
        bad=copy.deepcopy(patch);bad['nodes'][0]['attrs']['hidden_score']=1
        with self.assertRaises(ContractError):validate_knowledge_patch(bad,graph)
        bad=copy.deepcopy(patch);bad['nodes'][0]['attrs']['evidence_refs']='development'
        with self.assertRaisesRegex(ContractError,'string array'):validate_knowledge_patch(bad,graph)
        from rec_harness.runtime import edge
        relation=edge('relation','interpretation','development','SUPPORTS',['development'],'task',1)
        patch['edges']=[relation]
        validate_knowledge_patch(patch,graph)
        foreign=copy.deepcopy(graph);foreign['protocol_hash']='p'
        foreign['nodes'][0]['attrs']['protocol_hash']='foreign'
        with self.assertRaisesRegex(ContractError,'incomparable'):validate_knowledge_patch(patch,foreign)
        patch['edges'][0]['attrs']['evidence_refs']='development'
        with self.assertRaisesRegex(ContractError,'string array'):validate_knowledge_patch(patch,graph)


class CycleCommitRecoveryTests(unittest.TestCase):
    def graph_patch(self):
        return {'base_version':0,'nodes':[{'id':'cycle-decision','type':'DecisionTrace',
            'attrs':{'task_id':'task','protocol_hash':'p','origin':'local_evidence',
                     'selection':'discard','scientific_conclusion':'unknown'}}],'edges':[]}

    def test_receipt_failure_rolls_back_graph_and_allows_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(tmp);original=store._event
            receipt={'identity':'cycle','graph_version':1}
            def fail_receipt(kind,payload):
                if kind=='amazon_research_cycle_completed':
                    raise RuntimeError('simulated interruption before receipt insertion')
                return original(kind,payload)
            with mock_patch.object(store,'_event',side_effect=fail_receipt):
                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                    store.patch(self.graph_patch(),accompanying_events=[('amazon_research_cycle_completed',receipt)])
            store.close()
            store=Store(tmp)
            try:
                self.assertEqual(store.version,0)
                self.assertEqual(store.events(),[])
                self.assertEqual(store.snapshot()['nodes'],[])
                store.patch(self.graph_patch(),accompanying_events=[('amazon_research_cycle_completed',receipt)])
                self.assertEqual([e['kind'] for e in store.events()],['graph_committed','amazon_research_cycle_completed'])
                self.assertEqual(store.events()[-1]['payload'],receipt)
                self.assertEqual(store.rebuild()['graph_version'],1)
            finally:store.close()

    def test_projection_failure_after_commit_recovers_graph_and_receipt_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(tmp);receipt={'identity':'cycle','graph_version':1}
            with mock_patch.object(store,'export_events',side_effect=OSError('projection interrupted')):
                with self.assertRaisesRegex(OSError,'projection interrupted'):
                    store.patch(self.graph_patch(),accompanying_events=[('amazon_research_cycle_completed',receipt)])
            store.close()
            store=Store(tmp)
            try:
                self.assertEqual(store.version,1)
                self.assertEqual(store.events()[-1]['payload'],receipt)
                self.assertEqual(len((Path(tmp)/'events.jsonl').read_text().splitlines()),2)
                self.assertEqual(store.verify()['events'],2)
                self.assertEqual(store.rebuild()['graph_version'],1)
            finally:store.close()
