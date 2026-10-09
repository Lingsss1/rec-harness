import unittest
from rec_harness.bound_output import bound_schema
from rec_harness.agents import schema_for,validate_schema
from rec_harness.core import ContractError


class BoundOutputTests(unittest.TestCase):
    def test_legacy_schema_unchanged(self):
        for role in ['hypothesis','experiment','evaluation','lead']:
            self.assertEqual(bound_schema(role,{}),schema_for(role))

    def test_frozen_ids_and_budget_are_schema_constraints(self):
        binding={'proposal_id':'p','base_model_id':'base','controls':['base','control'],
                 'train_protocol_hash':'protocol','seeds':[42,17,83]}
        context={'host_bound_output':True,'bindings':binding,'budget':{'remaining':100},'gap':{'gap_id':'g'}}
        props=bound_schema('hypothesis',context)['properties']['proposal']['properties']
        for field,value in binding.items():
            validate_schema(value,props[field])
            with self.assertRaises(ContractError):validate_schema('wrong',props[field])
        for bad in [0,101]:
            with self.assertRaises(ContractError):validate_schema(bad,props['budget'])
        validate_schema(10,props['budget'])

    def test_files_and_evidence_bound_without_rewriting_output(self):
        context={'host_bound_output':True,'generated_controls':[{'file':'control-1.py'}],
                 'development':{'hash':'readout'}}
        files=bound_schema('experiment',context)['properties']['changed_files']
        validate_schema(['model.py','control-1.py'],files)
        with self.assertRaises(ContractError):validate_schema(['model.py'],files)
        refs=bound_schema('evaluation',context)['properties']['evidence_refs']
        with self.assertRaises(ContractError):validate_schema(['wrong'],refs)
        validate_schema(['readout'],refs)
