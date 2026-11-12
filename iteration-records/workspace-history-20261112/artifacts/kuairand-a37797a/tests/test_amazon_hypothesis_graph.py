import copy
import unittest
from rec_harness.adapters.amazon_hypothesis_graph import project
from rec_harness.core import digest


class HypothesisGraphTests(unittest.TestCase):
    def fixture(self):
        report={'proposal_id':'old','hash':'a'*64}
        task='amazon-research-'+digest('old')[:16]
        fact=task+'-'+report['hash'][:16]+'-development'
        def node(key,node_type,**attrs):
            return {'id':key,'type':node_type,'attrs':{'task_id':task,'protocol_hash':'p','visibility':'public',
                    'origin':'local_evidence','partition':'development',**attrs}}
        nodes=[node(fact,'Experiment',kind='derived_development_comparison',valid=True,counterevidence=True),
               node('slice','Observation',evidence_dependency_ids=[fact],effect=-.1),
               node('claim','Claim',origin='agent_hypothesis',status='provisional',evidence_refs=['slice']),
               node('revision','Claim',origin='agent_hypothesis',status='provisional',evidence_refs=[fact]),
               node('private','Experiment',partition='confirmation',secret='PRIVATE'),
               node('bad-claim','Claim',origin='agent_hypothesis',evidence_refs=['private']),
               node('future','Claim',origin='agent_hypothesis',task_id='unselected-task',evidence_refs=[],secret='FUTURE')]
        edges=[{'id':'produces','source':fact,'target':'slice','relation':'PRODUCES','attrs':{'evidence_refs':[fact]}},
               {'id':'support','source':'slice','target':'claim','relation':'SUPPORTS','attrs':{'evidence_refs':['slice']}},
               {'id':'rev','source':'revision','target':'claim','relation':'SUPERSEDES','attrs':{'evidence_refs':[fact]}},
               {'id':'private-ref','source':'revision','target':'claim','relation':'BASED_ON','attrs':{'evidence_refs':['private']}}]
        return {'nodes':nodes,'edges':edges,'graph_version':7},[report],fact

    def test_native_relations_and_counterevidence_reach_pre_design_view(self):
        snapshot,history,fact=self.fixture();before=copy.deepcopy(snapshot)
        graph=project(snapshot,history,'base','p',[{'dataset':'a','seed':42,'metrics':{'HR@10':.5}}])
        self.assertEqual({n['id'] for n in graph['nodes']} - {graph['nodes'][0]['id']},{fact,'slice','claim','revision'})
        self.assertEqual({e['id'] for e in graph['edges']},{'produces','support','rev'})
        self.assertEqual(graph['retrieval']['selection_reasons'][fact],'counterevidence')
        self.assertNotIn('PRIVATE',str(graph));self.assertNotIn('FUTURE',str(graph))
        self.assertEqual(snapshot,before)
        self.assertEqual(graph['hash'],digest({k:v for k,v in graph.items() if k!='hash'}))

    def test_limit_preserves_reference_boundary_and_reports_omissions(self):
        snapshot,history,fact=self.fixture()
        graph=project(snapshot,history,'base','p',[],limit=3)
        ids={n['id'] for n in graph['nodes']}
        self.assertLessEqual(len(ids),3)
        self.assertGreater(graph['retrieval']['omitted_node_count'],0)
        for edge in graph['edges']:
            self.assertTrue({edge['source'],edge['target'],*edge['attrs']['evidence_refs']}<=ids)
        for n in graph['nodes']:
            if n['type']=='Claim':self.assertTrue(set(n['attrs']['evidence_refs'])<=ids)

    def test_imported_history_does_not_fabricate_native_edges(self):
        snapshot,history,_=self.fixture()
        graph=project(snapshot,[{'proposal_id':'elsewhere','hash':'b'*64}],'base','p',[])
        self.assertEqual(len(graph['nodes']),1);self.assertEqual(graph['edges'],[])
        self.assertEqual(graph['retrieval']['unavailable_readout_hashes'],['b'*64])

    def test_other_protocol_not_exposed(self):
        snapshot,history,_=self.fixture()
        graph=project(snapshot,history,'base','different',[])
        self.assertEqual(len(graph['nodes']),1)
