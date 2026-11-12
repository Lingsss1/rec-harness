import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rec_harness.core import ContractError,digest
from rec_harness.store import Store
from rec_harness.adapters.amazon_evidence_view import evidence_view,fact_id,ordered_entries,attach_analysis_history,prior_round_paths
from rec_harness.adapters.amazon_hypothesis_graph import context_graph


def report(index):
    body={'schema':'amazon-development-selection-v1','proposal_id':'proposal-'+str(index),
          'protocol_hash':'p','rows':[{'seed':42}],
          'summary':{'eligible_for_confirmation':False,'metric':'HR@10','slices':[]}}
    return {**body,'hash':digest(body)}


def memory(r,index=1):
    task='amazon-research-'+digest(r['proposal_id'])[:16]
    attrs={'task_id':task,'protocol_hash':'p','origin':'agent_hypothesis','status':'provisional',
           'evidence_refs':['original-unavailable-receipt'],'description':'Measure attribution before repeating.'}
    return {'round':f'campaign/round-{index}','development':r,
            'provisional_knowledge':{'patch_json':json.dumps({'nodes':[
                {'id':'gap-'+str(index),'type':'MeasurementGap','attrs':attrs}], 'edges':[],'base_version':7})}}


class EvidenceViewTests(unittest.TestCase):
    def view(self,history=(),entries=(),failures=(),**kw):
        snapshot=kw.pop('snapshot',{'version':7,'nodes':[],'edges':[]})
        return evidence_view(snapshot,list(history),list(entries),list(failures),'base','p',[],**kw)

    def assert_closed(self,graph):
        ids={n['id'] for n in graph['nodes']}
        for n in graph['nodes']:
            self.assertTrue(set(n['attrs'].get('evidence_refs',[]))<=ids)
            if n['attrs'].get('origin')=='imported_memory':
                self.assertTrue(set(n['attrs'].get('evidence_dependency_ids',[]))<=ids)
        for e in graph['edges']:
            self.assertTrue({e['source'],e['target'],*e['attrs'].get('evidence_refs',[])}<=ids)

    def test_exported_gap_keeps_documentary_lineage_without_inventing_support(self):
        r=report(1);e=memory(r);before=copy.deepcopy(e)
        g=self.view([r],[e]);self.assert_closed(g)
        gap=next(n for n in g['nodes'] if n['type']=='MeasurementGap')
        self.assertEqual(gap['attrs']['reported_evidence_refs'],['original-unavailable-receipt'])
        self.assertNotIn('original-unavailable-receipt',{n['id'] for n in g['nodes']})
        self.assertTrue(all(e['relation']=='DERIVED_FROM' for e in g['edges']))
        self.assertEqual(e,before);self.assertEqual(g['graph_version'],7)
        self.assertEqual(g['hash'],digest({k:v for k,v in g.items() if k!='hash'}))

    def test_recent_experiment_survives_stale_semantic_shortlist(self):
        reports=[report(i) for i in range(12)]
        g=self.view(reports,[memory(r,i) for i,r in enumerate(reports)],selected_history=reports[:2],limit=12)
        ids={n['id'] for n in g['nodes']}
        self.assertTrue({fact_id(r) for r in reports[-2:]}<=ids)
        self.assertTrue(any(n['type']=='MeasurementGap' for n in g['nodes']))
        self.assertLessEqual(len(g['nodes']),12);self.assert_closed(g)

    def test_tiny_budget_reports_omitted_pins_and_keeps_closed_view(self):
        reports=[report(1),report(2)]
        g=self.view(reports,[memory(reports[-1])],limit=2)
        self.assertEqual(len(g['nodes']),2)
        self.assertEqual(g['nodes'][1]['id'],fact_id(reports[-1]))
        self.assertTrue(g['retrieval']['omitted_requests']);self.assert_closed(g)

    def test_native_measurement_gap_and_revision_survive(self):
        r=report(1);task='amazon-research-'+digest(r['proposal_id'])[:16];f=fact_id(r)
        common={'task_id':task,'protocol_hash':'p','visibility':'public','partition':'development'}
        nodes=[{'id':f,'type':'Experiment','attrs':{**common,'origin':'local_evidence',
                'kind':'derived_development_comparison','valid':True}},
               {'id':'gap','type':'MeasurementGap','attrs':{**common,'origin':'agent_hypothesis','evidence_refs':[f]}},
               {'id':'old','type':'Claim','attrs':{**common,'origin':'agent_hypothesis','evidence_refs':[f]}},
               {'id':'new','type':'Claim','attrs':{**common,'origin':'agent_hypothesis','evidence_refs':[f]}}]
        edges=[{'id':'revision','source':'new','target':'old','relation':'SUPERSEDES','attrs':{'evidence_refs':[f]}}]
        g=self.view([r],snapshot={'version':9,'nodes':nodes,'edges':edges})
        self.assertIn('gap',{n['id'] for n in g['nodes']});self.assertEqual(g['edges'],edges)
        self.assertEqual(g['graph_version'],9);self.assert_closed(g)

    def test_native_optional_visibility_matches_public_knowledge_contract(self):
        r=report(1);task='amazon-research-'+digest(r['proposal_id'])[:16];f=fact_id(r)
        attrs={'task_id':task,'protocol_hash':'p','origin':'agent_hypothesis','evidence_refs':[f]}
        nodes=[{'id':f,'type':'Experiment','attrs':{'task_id':task,'protocol_hash':'p',
                'origin':'local_evidence','kind':'derived_development_comparison','valid':True,
                'visibility':'public','partition':'development'}},
               {'id':'implicit-public','type':'MeasurementGap','attrs':attrs},
               {'id':'private','type':'MeasurementGap','attrs':{**attrs,'visibility':'private'}},
               {'id':'confirmation','type':'MeasurementGap','attrs':{**attrs,'partition':'confirmation'}},
               {'id':'missing-ref','type':'Claim','attrs':{**attrs,'evidence_refs':['absent']}}]
        g=self.view([r],snapshot={'version':9,'nodes':nodes,'edges':[]})
        ids={n['id'] for n in g['nodes']}
        self.assertIn('implicit-public',ids)
        self.assertFalse(ids & {'private','confirmation','missing-ref'})
        self.assert_closed(g)

    def test_other_protocol_tamper_and_private_patch_are_rejected(self):
        r=report(1);r['protocol_hash']='wrong'
        with self.assertRaises(ContractError):self.view([r])
        r=report(1);r['summary']['metric']='tampered'
        with self.assertRaises(ContractError):self.view([r])
        r=report(1);e=memory(r);patch=json.loads(e['provisional_knowledge']['patch_json'])
        patch['nodes'][0]['attrs']['visibility']='private';e['provisional_knowledge']['patch_json']=json.dumps(patch)
        with self.assertRaises(ContractError):self.view([r],[e])

    def test_failure_is_not_mechanism_counterevidence(self):
        g=self.view(failures=[{'round':'round-1','reason':'review failed'}])
        n=g['nodes'][1]
        self.assertEqual(n['attrs']['kind'],'execution_failure')
        self.assertNotIn('counterevidence',n['attrs'])

    def test_numeric_round_order_preserves_campaign_order(self):
        entries=[{'round':r} for r in ['v5/round-8','v6/round-1','v6/round-10','v6/round-2','v6/round-12']]
        self.assertEqual([e['round'] for e in ordered_entries(entries)],
                         ['v5/round-8','v6/round-1','v6/round-2','v6/round-10','v6/round-12'])

    def test_resumed_round_cannot_read_future_directories(self):
        paths=[Path(f'/rounds/round-{i}/development-readout.json') for i in [1,10,11,2,3]]
        self.assertEqual([p.parent.name for p in prior_round_paths(paths,Path('/rounds/round-10'))],
                         ['round-1','round-2','round-3'])

    def test_prefix_never_contains_future_records(self):
        reports=[report(i) for i in range(5)]
        for i in range(5):
            g=self.view(reports[:i],[memory(r,j) for j,r in enumerate(reports[:i])])
            ids={n['id'] for n in g['nodes']}
            self.assertFalse(ids & {fact_id(r) for r in reports[i:]})

    def test_analyst_receives_frozen_history_without_expanding_patch_authority(self):
        r=report(1);view=self.view([r],[memory(r)])
        request={'graph':{'nodes':[]},'patch_contract':{'evidence_refs':['current-only']},'instructions':[]}
        combined=attach_analysis_history(request,view)
        self.assertEqual(combined['prior_research_evidence'],view)
        self.assertEqual(combined['patch_contract'],request['patch_contract'])
        self.assertNotIn('prior_research_evidence',request)
        bad=copy.deepcopy(view);bad['nodes']=[]
        with self.assertRaises(ContractError):attach_analysis_history(request,bad)

    def test_context_dispatch_preserves_legacy_and_does_not_mutate_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            s=Store(Path(tmp)/'memory');before=s.verify();s.close()
            controller=SimpleNamespace(study=Path(tmp),config={})
            r=report(1)
            old=context_graph(controller,[r],'base','p',[])
            self.assertEqual(len(old['nodes']),1)
            controller.config={'evidence_graph_policy':'prefix-evidence-v2'}
            new=context_graph(controller,[],'base','p',[],available_history=[r],entries=[memory(r)])
            self.assertGreater(len(new['nodes']),1)
            self.assertEqual(new['source_event_head'],old['source_event_head'])
            s=Store(Path(tmp)/'memory');self.assertEqual(s.verify(),before);s.close()

    def test_unselected_future_native_task_and_confirmation_are_invisible(self):
        r=report(1);f=report(2)
        nodes=[]
        for x,partition in [(r,'confirmation'),(f,'development')]:
            nodes.append({'id':fact_id(x),'type':'Experiment','attrs':{
                'task_id':'amazon-research-'+digest(x['proposal_id'])[:16],
                'protocol_hash':'p','visibility':'public','partition':partition,
                'kind':'derived_development_comparison','origin':'local_evidence','valid':True}})
        g=self.view([r],snapshot={'version':1,'nodes':nodes,'edges':[]})
        self.assertNotIn(fact_id(f),{n['id'] for n in g['nodes']})
        self.assertFalse(any(n['attrs'].get('partition')=='confirmation' for n in g['nodes']))
