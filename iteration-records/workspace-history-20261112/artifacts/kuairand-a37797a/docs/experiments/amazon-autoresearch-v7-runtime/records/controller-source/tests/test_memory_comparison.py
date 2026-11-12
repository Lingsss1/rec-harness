import copy
import unittest
from rec_harness.core import ContractError
from rec_harness.memory_comparison import compare_traces
from rec_harness.memory_view import present
import test_diagnostics


class MemoryComparisonTests(unittest.TestCase):
    def traces(self):
        bundle={'nodes':[{'id':'fact'}],'edges':[]}
        common={'trace_id':'evidence-0','role':'evidence','visible_source_snapshot':{},
            'transport_status':'success','resource_cost':{'declared_bcu':1}}
        return tuple([{**copy.deepcopy(common),'visible_state':{'evidence':present(bundle,mode),'budget':10}}] for mode in ['graph','flat'])

    def test_equal_facts_and_source_are_distinct_from_cost_or_quality(self):
        left,right=self.traces();right[0]['resource_cost']['token_usage']={'input_tokens':5}
        result=compare_traces(left,right,10000)
        self.assertTrue(result['all_inputs_equal'])
        self.assertNotIn('gain',result)

    def test_operational_context_or_source_changes_are_not_silently_ignored(self):
        for field in ['visible_state','visible_source_snapshot']:
            left,right=self.traces();right[0][field]['different']='receipt-or-source'
            self.assertFalse(compare_traces(left,right,10000)['all_inputs_equal'])
        left,right=self.traces();extra=copy.deepcopy(right[0]);extra['role']='lead';right.append(extra)
        result=compare_traces(left,right,10000)
        self.assertFalse(result['all_inputs_equal']);self.assertEqual(len(result['pairs']),2)

    def test_missing_provenance_wrong_mode_and_oversize_fail(self):
        left,right=self.traces()
        with self.assertRaisesRegex(ContractError,'byte limit'):compare_traces(left,right,1)
        with self.assertRaisesRegex(ContractError,'presentation'):compare_traces(right,left,10000)
        right[0]['visible_source_snapshot']=None
        with self.assertRaisesRegex(ContractError,'source snapshot'):compare_traces(left,right,10000)

    def test_runtime_rejects_context_overflow_before_call_or_reservation(self):
        test_diagnostics.DiagnosticTests.setUp(self)
        research=self.research;research.config['context_byte_limit']=1
        before=research.store.budget()
        from rec_harness.runtime import Research
        with self.assertRaisesRegex(ContractError,'byte limit'):
            Research._agent(research,'evidence',{'evidence':{'nodes':[],'edges':[]}},'attempt')
        self.assertEqual(research.store.budget(),before)
