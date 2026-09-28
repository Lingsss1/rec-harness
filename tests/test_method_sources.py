import tempfile
import unittest
from pathlib import Path
from rec_harness.core import CARD_FIELDS,ContractError,read_json,validate_card
from rec_harness.runtime import Research
from test_runtime import ROOT,ScriptedAgent


class MethodSourceTests(unittest.TestCase):
    def test_locatable_citations_are_required_only_under_new_contract(self):
        method={key:[] for key in CARD_FIELDS['method'].split()}
        method.update(method_id='m',mechanism='fixture',cost='fixture',origin='paper_claim')
        validate_card('method',method)  # Legacy interpretation stays unchanged.
        for sources in [[],['unknown'],['paper title only'],['https://example.org/'],[' '],'not a list']:
            with self.subTest(sources=sources):
                method['sources']=sources
                with self.assertRaises(ContractError):validate_card('method',method,require_method_sources=True)
        for source in ['https://example.org/paper.pdf section 2','doi:10.1234/fictional-paper',
                       'arXiv:1234.56789v2 section 3','arXiv:cs/0123456v1']:
            method['sources']=[source]
            validate_card('method',method,require_method_sources=True)
        # These are deliberately fictional references: passing syntax is not truth.
        method.update(origin='agent_hypothesis',sources=[])
        validate_card('method',method,require_method_sources=True)

    def test_missing_paper_sources_stop_before_candidate_implementation(self):
        calls=[]
        class Unsourced(ScriptedAgent):
            def run(self,role,context,directory):
                calls.append(role)
                result=super().run(role,context,directory)
                if role=='hypothesis':
                    self_contract=context['method_source_contract']
                    assert self_contract['schema']=='locatable-method-sources-v1'
                    result['method']['origin']='paper_claim'
                return result
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'campaign';config=read_json(ROOT/'examples/toy/campaign.json')
            config.update(source=str(ROOT/'examples/toy'),max_rounds=1)
            Research.init(root,config);research=Research(root,agent=Unsourced())
            try:
                self.assertTrue(research.config['require_method_sources'])
                research.run()
                self.assertIn('hypothesis',calls);self.assertNotIn('experiment',calls)
                self.assertFalse(any(n['type']=='Method' for n in research.store.snapshot()['nodes']))
            finally:research.store.close()
