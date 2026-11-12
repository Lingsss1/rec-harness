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

    def test_graph_binding_does_not_constrain_unrelated_text(self):
        context={'host_bound_output':True,'evidence_graph':{'nodes':[{'id':'node'}]}}
        schema=bound_schema('researcher',context)
        gap=schema['properties']['gap']['properties']
        self.assertEqual(gap['observation_refs']['items']['enum'],['node'])
        for field in ['affected_components','alternative_explanations','predicted_observations']:
            self.assertNotIn('enum',gap[field]['items'])
        self.assertNotIn('enum',schema_for('evidence')['properties']['observation_refs']['items'])

    def test_service_projection_preserves_strict_host_validation(self):
        from rec_harness.bound_output import service_schema
        context={'host_bound_output':True,'bindings':{'proposal_id':'p','base_model_id':'b',
            'controls':['b','c'],'train_protocol_hash':'h','seeds':[42,17]},'budget':{'remaining':50}}
        host=bound_schema('hypothesis',context);wire=service_schema(host)
        props=wire['properties']['proposal']['properties']
        self.assertNotIn('enum',props['controls'])
        self.assertEqual(props['controls']['items']['enum'],['b','c'])
        self.assertEqual(props['controls']['minItems'],2)
        self.assertEqual(props['controls']['maxItems'],2)
        self.assertEqual(props['budget']['minimum'],0)
        strict=host['properties']['proposal']['properties']
        for value in [['c','b'],['b','b'],['b']]:
            with self.assertRaises(ContractError):validate_schema(value,strict['controls'])
        with self.assertRaises(ContractError):validate_schema(0,strict['budget'])
        self.assertEqual(strict['controls']['enum'],[['b','c']])
