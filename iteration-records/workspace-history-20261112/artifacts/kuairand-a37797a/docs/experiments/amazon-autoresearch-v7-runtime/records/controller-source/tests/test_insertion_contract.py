import json
import unittest
from rec_harness.core import ContractError
from rec_harness.adapters.insertion_contract import parse_contract,verify_scope,method_catalog,SCHEMA
from rec_harness.adapters.model_source_contract import review_source
from test_model_source_contract import SOURCE

class InsertionContractTests(unittest.TestCase):
    def contract(self,methods=None):
        return json.dumps({'schema':SCHEMA,'allowed_methods':methods or ['NextItNetModel.forward'],'rationale':'Restrict readout change.'})

    def test_predeclared_scope_matches_actual_changes(self):
        contract=parse_contract(self.contract(),required=True,available_methods=method_catalog(SOURCE))
        review=review_source(SOURCE,SOURCE.replace('return self.block(x)','return self.block(x)*0.9'))
        self.assertEqual(verify_scope(contract,review)['changed_methods'],['NextItNetModel.forward'])
        with self.assertRaisesRegex(ContractError,'actual model AST change'):
            verify_scope(contract,review_source(SOURCE,SOURCE+'\n# no change\n'))
        with self.assertRaisesRegex(ContractError,'outside frozen'):
            verify_scope(contract,review_source(SOURCE,SOURCE.replace('return self.linear(x)','return self.linear(x)*0.9')))

    def test_reject_invalid_or_nonexistent_methods(self):
        for methods in [['NextItNetModel.missing'],['NextItNetModel.forward']*2,['NextItNetModel._causal_conv']]:
            with self.assertRaises(ContractError):parse_contract(self.contract(methods),required=True,available_methods=method_catalog(SOURCE))
        with self.assertRaises(ContractError):parse_contract('free prose',required=True)
        with self.assertRaises(ContractError):parse_contract('{broken JSON')
        self.assertIsNone(parse_contract('legacy free prose'))
