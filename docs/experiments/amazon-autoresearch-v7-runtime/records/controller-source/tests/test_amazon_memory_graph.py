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
            for field in ['created_at','revision_id']:
                invalid=copy.deepcopy(patch);invalid['edges'][0]['attrs'].pop(field)
                with self.assertRaisesRegex(ContractError,'missing '+field):validate_knowledge_patch(invalid,view)
            invalid=copy.deepcopy(patch);invalid['edges'].append(copy.deepcopy(revision))
            with self.assertRaisesRegex(ContractError,'duplicate'):validate_knowledge_patch(invalid,view)
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

    def test_later_review_receives_committed_revision_and_its_development_basis(self):
        entries=[];nodes=[]
        for index in [1,2]:
            proposal='proposal-'+str(index);readout=digest(proposal)
            task='amazon-research-'+digest(proposal)[:16]
            fact=task+'-'+readout[:16]+'-development'
            entries.append({'round':'round-'+str(index),'development_hash':readout,'design':{'proposal':{'proposal_id':proposal}}})
            nodes.extend([node(fact,'Experiment',{'partition':'development','valid':True,
                'kind':'derived_development_comparison'},task,'p'),
                node('explanation-'+str(index),'Claim',{'status':'provisional','scope':task,
                    'evidence_refs':[fact]},task,'p','agent_hypothesis')])
        relation=edge('revision','explanation-2','explanation-1','SUPERSEDES',[nodes[2]['id']],nodes[2]['attrs']['task_id'],1)
        relation['attrs'].update(origin='agent_hypothesis',status='provisional')
        observation=node('slice','Observation',{'partition':'development','evidence_dependency_ids':[nodes[2]['id']]},nodes[2]['attrs']['task_id'],'p')
        nodes.append(observation);relation['attrs']['evidence_refs'].append('slice')
        snapshot={'nodes':nodes,'edges':[relation]}
        original=copy.deepcopy(snapshot)
        graph={'task_id':'third','protocol_hash':'p','nodes':[],'edges':[]}
        view=prior_explanations(graph,snapshot,entries)
        self.assertEqual(view['historical_revision_edge_ids'],['revision'])
        self.assertEqual(view['historical_evidence_ids'],sorted([nodes[2]['id'],'slice']))
        self.assertEqual(view['edges'],[relation])
        self.assertEqual(snapshot,original)
        self.assertEqual(view,prior_explanations(graph,snapshot,entries))
        ids={item['id'] for item in view['nodes']}
        self.assertTrue(set(relation['attrs']['evidence_refs'])<=ids)
        self.assertEqual(next(item for item in view['nodes'] if item['id']=='explanation-1')['attrs']['status'],'provisional')
        # The revision does not promote its historical basis to a current fact.
        self.assertNotEqual(next(item for item in view['nodes'] if item['id']==nodes[2]['id'])['attrs']['task_id'],'third')
        for field,value in [('partition','confirmation'),('protocol_hash','other'),('valid',False)]:
            changed=copy.deepcopy(snapshot);changed['nodes'][2]['attrs'][field]=value
            self.assertEqual(prior_explanations(graph,changed,entries)['historical_revision_edge_ids'],[])
        changed=copy.deepcopy(snapshot);changed['edges'][0]['attrs']['evidence_refs']=['private-evidence']
        self.assertEqual(prior_explanations(graph,changed,entries)['edges'],[])
        # Memory selection remains bounded; do not follow links into omitted rounds.
        self.assertEqual(prior_explanations(graph,snapshot,entries[:1])['edges'],[])
