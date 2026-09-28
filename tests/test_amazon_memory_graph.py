import copy
import tempfile
import unittest
from rec_harness.adapters.amazon_memory_graph import prior_explanations
from rec_harness.core import ContractError,digest
from rec_harness.knowledge import validate_knowledge_patch
from rec_harness.runtime import node,edge
from rec_harness.store import Store


class PriorExplanationTests(unittest.TestCase):
    def test_local_revision_preserves_old_explanation_and_requires_current_reference(self):
        proposal='old-proposal';readout='abcdef0123456789remaining'
        task='amazon-research-'+digest(proposal)[:16]
        fact_id=task+'-'+readout[:16]+'-development'
        entries=[{'round':'old','development_hash':readout,'design':{'proposal':{'proposal_id':proposal}}}]
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(tmp);self.addCleanup(store.close)
            old=node('old-gap','GapHypothesis',{'status':'proposed'},task,'p','agent_hypothesis')
            fact=node(fact_id,'Experiment',{'partition':'development','valid':True,
                'kind':'derived_development_comparison'},task,'p')
            current=node('current-fact','Experiment',{'partition':'development','valid':True},'new','p')
            store.patch({'base_version':0,'nodes':[old,fact,current],'edges':[]})
            graph={'task_id':'new','protocol_hash':'p','graph_version':store.version,'nodes':[current],'edges':[]}
            view=prior_explanations(graph,store.snapshot(),entries)
            self.assertEqual(graph['nodes'],[current])
            self.assertEqual(view['historical_explanation_ids'],['old-gap'])
            self.assertNotIn(fact_id,[n['id'] for n in view['nodes']])
            claim=node('new-interpretation','Claim',{'status':'provisional','scope':'new',
                'evidence_refs':['current-fact']},'new','p','agent_hypothesis')
            revision=edge('revision','new-interpretation','old-gap','SUPERSEDES',['current-fact'],'new',2)
            revision['attrs'].update(origin='agent_hypothesis',status='provisional')
            patch={'base_version':store.version,'nodes':[claim],'edges':[revision]}
            validate_knowledge_patch(patch,view)
            for field,value in [('relation','CONTRADICTS'),('source','current-fact')]:
                invalid=copy.deepcopy(patch);invalid['edges'][0][field]=value
                with self.assertRaisesRegex(ContractError,'scoped provisional'):validate_knowledge_patch(invalid,view)
            invalid=copy.deepcopy(patch);invalid['edges'][0]['attrs']['evidence_refs']=['old-gap']
            with self.assertRaisesRegex(ContractError,'current development'):validate_knowledge_patch(invalid,view)
            for field,value in [('task_id','foreign'),('protocol_hash','other'),('valid',False),('partition','confirmation')]:
                invalid_view=copy.deepcopy(view)
                next(n for n in invalid_view['nodes'] if n['id']=='current-fact')['attrs'][field]=value
                with self.assertRaisesRegex(ContractError,'current development'):validate_knowledge_patch(patch,invalid_view)
            store.patch(patch)
            self.assertEqual(next(n for n in store.snapshot()['nodes'] if n['id']=='old-gap'),old)
            self.assertEqual(store.snapshot()['edges'][0]['relation'],'SUPERSEDES')
            missing=prior_explanations(graph,{'nodes':[old]},entries)
            self.assertEqual(missing['historical_explanation_ids'],[])
            self.assertEqual(len(missing['historical_explanations_unavailable']),1)
            other=prior_explanations({**graph,'protocol_hash':'other'},store.snapshot(),entries)
            self.assertEqual(other['historical_explanation_ids'],[])
            for field,value in [('task_id','foreign'),('kind','unrelated'),('valid',False),('partition','confirmation')]:
                invalid_snapshot=copy.deepcopy(store.snapshot())
                next(n for n in invalid_snapshot['nodes'] if n['id']==fact_id)['attrs'][field]=value
                self.assertEqual(prior_explanations(graph,invalid_snapshot,entries)['historical_explanation_ids'],[])
            private_snapshot=copy.deepcopy(store.snapshot())
            next(n for n in private_snapshot['nodes'] if n['id']=='old-gap')['attrs']['partition']='confirmation'
            self.assertEqual(prior_explanations(graph,private_snapshot,entries)['historical_explanation_ids'],[])
