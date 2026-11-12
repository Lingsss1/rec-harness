from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rec_harness.core import read_json
from rec_harness.runtime import Research
from rec_harness.store import Store
from test_runtime import ROOT,ScriptedAgent


CITATION='https://example.org/fictional-paper.pdf section 2; version unknown'


class MethodSourceGraphTests(unittest.TestCase):
    def assert_sources(self,graph):
        sources=[n for n in graph['nodes'] if n['type']=='Source']
        self.assertEqual(len(sources),1)
        source=sources[0];self.assertEqual(source['attrs']['citation'],CITATION)
        self.assertEqual(source['attrs']['verification_status'],'agent_reported_unverified')
        self.assertEqual(source['attrs']['source_version'],'unknown')
        self.assertIsNone(source['attrs']['retrieved_content_ref'])
        method=next(n for n in graph['nodes'] if n['type']=='Method')
        self.assertEqual(method['attrs']['origin'],'paper_claim')
        relation=next(e for e in graph['edges'] if e['source']==method['id'] and e['target']==source['id'])
        self.assertEqual(relation['relation'],'DERIVED_FROM')
        self.assertEqual(relation['attrs']['origin'],'paper_claim')
        self.assertFalse(any(e['relation']=='SUPPORTS' and e['source']==source['id'] for e in graph['edges']))

    def test_model_loop_indexes_citation_without_promoting_it_to_evidence(self):
        test=self
        class Cited(ScriptedAgent):
            def run(self,role,context,directory):
                if role=='experiment':test.assertTrue(any(n['type']=='Source' for n in context['evidence']['nodes']))
                result=super().run(role,context,directory)
                if role=='hypothesis':result['method'].update(origin='paper_claim',sources=[CITATION,CITATION])
                return result
        with tempfile.TemporaryDirectory() as temp:
            config=read_json(ROOT/'examples/toy/campaign.json');config.update(source=str(ROOT/'examples/toy'),max_rounds=1)
            root=Path(temp)/'campaign';Research.init(root,config);research=Research(root,agent=Cited())
            try:
                research.run();self.assert_sources(research.store.snapshot())
                before=research.store.verify()
                with patch.object(research.agent,'run',side_effect=AssertionError('no repeated role')):research.run()
                self.assertEqual(research.store.verify(),before)
            finally:research.store.close()

    def test_amazon_development_projection_preserves_paper_origin_and_resume(self):
        from rec_harness.adapters.amazon_research_graph import ingest_development
        with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
            root=Path(temp);store=Store(root/'memory')
            cards={'gap':{},'proposal':{},'method':{'origin':'paper_claim','sources':[CITATION,CITATION]}}
            store.event('amazon_research_plan_registered',{'proposal_id':'fixture','baseline':'base','cards':cards,'versions':{'candidate':{'model_version':'recent'}}});store.close()
            report={'protocol_hash':'p','hash':'a'*64,'source_receipts':{},'summary':{'metric':'HR@10',
                'selection_policy':'baseline-positive-v1','model_minus_baseline':{'recent':-.01,'early':.01},
                'candidate_minus_comparator':{'base':-.01},'eligible_for_confirmation':True,'minimum_effect':0.01,'slices':[],
                'mechanism_comparisons':[{'candidate':'recent','control':'early','candidate_minus_control':-.01,'mechanism_status':'unknown'}]}}
            stack.enter_context(patch('rec_harness.adapters.amazon_research_graph.development_readout',return_value=report))
            stack.enter_context(patch('rec_harness.adapters.amazon_research_graph.ingest'))
            stack.enter_context(patch('rec_harness.adapters.amazon_contract_graph.attach_contracts'))
            stack.enter_context(patch('rec_harness.adapters.amazon_controls.load_config',return_value={}))
            graph=ingest_development(root,'fixture',include_method_sources=True)
            self.assert_sources(graph)
            comparisons=[n for n in graph['nodes'] if n['attrs'].get('kind')=='mechanism_comparison']
            fact=next(n for n in graph['nodes'] if n['attrs'].get('kind')=='derived_development_comparison')
            self.assertFalse(fact['attrs']['effect_gate_passed'])
            self.assertTrue(fact['attrs']['counterevidence'])
            self.assertEqual(len(comparisons),1)
            self.assertEqual(comparisons[0]['attrs']['mechanism_status'],'unknown')
            self.assertFalse(comparisons[0]['attrs']['independent_unit'])
            self.assertEqual(ingest_development(root,'fixture',include_method_sources=True),graph)
