import copy
import unittest
from rec_harness.optimizer_handoff import request,validate
from rec_harness.core import ContractError


class AmazonHandoffEvidenceTests(unittest.TestCase):
    def test_existing_baseline_metric_cannot_be_referred_as_missing_tool_output(self):
        packet=request(self.fixture(),10)
        review={'assessment':'measurement_gap','rationale':'fixture','evidence_refs':['trace'],
            'prerequisites':{k:{'status':'satisfied','rationale':'fixture'} for k in ['invocation','implementation','data','applicability']},
            'requirements':[{'measurement':'HR@10','blocked_decision':'fixture','competing_explanations':['a','b'],
                'cheapest_test':'fixture','development_budget':5,'acceptance_contract':'fixture'}]}
        with self.assertRaisesRegex(ContractError,'already exists'):
            validate({'measurement_handoff_request':packet},review,'candidate')
        review['requirements'][0]['measurement']='HR@10.category_spread'
        self.assertEqual(validate({'measurement_handoff_request':packet},review,'candidate')['required_information'],'HR@10.category_spread')

    def fixture(self):
        rows=[{'dataset':name,'seed':42,'metrics':{'HR@10':0.2}} for name in ['A','B']]
        receipts=[{'dataset':name,'seed':42,'status':'verified','protocol_hash':'p','receipt_hash':'a'*64,'manifest_hash':'b'*64} for name in ['B','A']]
        return {'decision_traces':[{'trace_id':'trace','transport_status':'success','model_version':'model',
            'visible_state':{'protocol_hash':'p','development_baseline':rows,'baseline_execution_evidence':receipts}}]}

    def test_verified_summaries_match_rows_independent_of_order(self):
        context=self.fixture();packet=request(context,10)
        self.assertEqual(packet['allowed_evidence_refs'],['trace'])
        self.assertEqual(packet['trace_evidence'][0]['rows'],context['decision_traces'][0]['visible_state']['development_baseline'])

    def test_mismatched_or_unverified_receipts_are_not_measurement_evidence(self):
        for change in [{'dataset':'other'},{'seed':83},{'protocol_hash':'other'},{'status':'success'},
                       {'status':'unavailable'},{'receipt_hash':''},{'manifest_hash':None}]:
            with self.subTest(change=change):
                context=self.fixture();context['decision_traces'][0]['visible_state']['baseline_execution_evidence'][0].update(change)
                self.assertEqual(request(context,10)['allowed_evidence_refs'],[])

    def test_duplicate_rows_or_receipts_cannot_hide_missing_conditions(self):
        for field in ['development_baseline','baseline_execution_evidence']:
            with self.subTest(field=field):
                context=self.fixture();values=context['decision_traces'][0]['visible_state'][field]
                values[1]=copy.deepcopy(values[0])
                self.assertEqual(request(context,10)['allowed_evidence_refs'],[])

    def test_legacy_list_baseline_does_not_crash_or_bypass_verified_evidence(self):
        context=self.fixture();visible=context['decision_traces'][0]['visible_state']
        visible['baseline']=[{'dataset':'A','seed':42,'metrics':{'HR@10':0.2}}]
        self.assertEqual(request(context,10)['allowed_evidence_refs'],['trace'])
        visible.pop('baseline_execution_evidence')
        self.assertEqual(request(context,10)['allowed_evidence_refs'],[])
