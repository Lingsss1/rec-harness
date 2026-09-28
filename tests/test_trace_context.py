import copy
import unittest
from rec_harness.core import ContractError,canonical,digest
from rec_harness.trace_context import compact_traces,expand_traces


class TraceContextTests(unittest.TestCase):
    def test_lifecycle_trace_identity_uses_fully_restored_input(self):
        from rec_harness.evolution_graph import lifecycle_graph
        original=self.fixture();compact=compact_traces(original)
        events=[{'kind':'candidate_proposed','hash':'a'*64,'created_at':'fixture',
                 'payload':{'candidate_id':'c','kind':'harness','plan_hash':'p','context_hash':digest(compact)}},
                {'kind':'candidate_implemented','hash':'b'*64,'created_at':'fixture',
                 'payload':{'candidate_id':'c','implementation_hash':'i','proposal_ref':{'sha256':'r'}}}]
        proposal={'failure_pattern':'fixture','causal_hypothesis':'fixture','expected_decision_change':'fixture',
                  'regression_risks':[],'trace_refs':['0']}
        graph=lifecycle_graph(events,{'c':compact},{'c':proposal})
        trace=next(n for n in graph['nodes'] if n['attrs'].get('external_trace_id')=='0')
        self.assertEqual(trace['attrs']['trace_hash'],digest(original['decision_traces'][0]))

    def fixture(self):
        return {'decision_traces':[{'trace_id':str(i),'visible_state':{'baseline':{'metrics':[0.5]*300},'unique':i},'role_output':{'choice':i}}
            for i in range(4)],'base_policy':{'retrieval_limit':25}}

    def test_lossless_sharing_preserves_each_visible_input_and_output(self):
        original=self.fixture();before=copy.deepcopy(original);compact=compact_traces(original)
        self.assertEqual(original,before)
        self.assertLess(len(canonical(compact)),len(canonical(original)))
        self.assertEqual(expand_traces(compact),original)
        self.assertEqual(len(compact['shared_visible_values']),1)
        self.assertEqual(compact_traces(compact),compact)
        for i,trace in enumerate(compact['decision_traces']):
            self.assertEqual(trace['visible_state'],{'unique':i})
            self.assertEqual(trace['role_output'],{'choice':i})

    def test_tampering_and_missing_references_are_rejected(self):
        compact=compact_traces(self.fixture());key=next(iter(compact['shared_visible_values']))
        corrupt=copy.deepcopy(compact);corrupt['shared_visible_values'][key]['metrics'][0]=9
        with self.assertRaises(ContractError):expand_traces(corrupt)
        corrupt=copy.deepcopy(compact);del corrupt['shared_visible_values'][key]
        with self.assertRaises(ContractError):expand_traces(corrupt)
        corrupt=copy.deepcopy(compact);corrupt['decision_traces'][0]['visible_state']['unique']=8
        with self.assertRaises(ContractError):expand_traces(corrupt)

    def test_no_repeats_and_reserved_fields_preserve_original(self):
        original={'decision_traces':[{'trace_id':'t','visible_state':{'x':'small'}}]}
        self.assertEqual(compact_traces(original),original)
        original=self.fixture();original['shared_visible_values']={'user':'data'}
        self.assertEqual(compact_traces(original),original)
