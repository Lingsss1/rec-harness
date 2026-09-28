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

    def with_observations(self):
        context=self.fixture();visible=context['decision_traces'][0]['visible_state']
        for field,name in [('activation_diagnostics','rms'),('block_score_diagnostics','block.delta.HR@10')]:
            visible[field]=[{'observation_id':field+'-one','model_version':'model','dataset':'A','seed':42,
                'measurements':{name:0.1},'limitations':['fixed development weights only'],
                'scope':'public development aggregate; no retrained intervention'}]
        return context

    def test_observations_reach_packet_and_cannot_be_proposed_as_new_measurements(self):
        context=self.with_observations();packet=request(context,10)
        for field in ['activation_diagnostics','block_score_diagnostics']:
            self.assertEqual(packet['trace_evidence'][0][field],context['decision_traces'][0]['visible_state'][field])
        review={'assessment':'measurement_gap','rationale':'fixture','evidence_refs':['trace'],
            'prerequisites':{k:{'status':'satisfied','rationale':'fixture'} for k in ['invocation','implementation','data','applicability']},
            'requirements':[{'measurement':'new','blocked_decision':'fixture','competing_explanations':['a','b'],
                'cheapest_test':'fixture','development_budget':5,'acceptance_contract':'fixture'}]}
        validate({'measurement_handoff_request':packet},review,'candidate')
        for name in ['rms','block.delta.HR@10']:
            review['requirements'][0]['measurement']=name
            with self.assertRaisesRegex(ContractError,'already exists'):
                validate({'measurement_handoff_request':packet},review,'candidate')
        self.assertIn('does not expand',packet['measurement_contract']['authoring_evidence_scope'])

    def test_failed_or_mismatched_observation_trace_is_not_usable_evidence(self):
        for change in [{'model_version':'other'},{'seed':83},{'dataset':'other'},
                       {'measurements':{'rms':float('nan')}},{'private_path':'excluded'},
                       {'limitations':'missing'},{'scope':''}]:
            with self.subTest(change=change):
                context=self.with_observations()
                context['decision_traces'][0]['visible_state']['activation_diagnostics'][0].update(change)
                self.assertEqual(request(context,10)['allowed_evidence_refs'],[])
        context=self.with_observations()
        context['decision_traces'][0]['visible_state']['measurement_failure']='activation budget unavailable'
        self.assertEqual(request(context,10)['allowed_evidence_refs'],[])

    def test_committed_handoff_export_preserves_observations_without_source_requery(self):
        import tempfile
        from pathlib import Path
        from rec_harness.core import Artifacts,digest,write_json
        from rec_harness.optimizer_handoff import load_context
        from rec_harness.store import Store
        packet=request(self.with_observations(),10)
        context={'measurement_handoff_request':packet,'source':{'task_id':'source-task'}}
        review={'assessment':'measurement_gap','rationale':'fixture','evidence_refs':['trace'],
            'prerequisites':{k:{'status':'satisfied','rationale':'fixture'} for k in ['invocation','implementation','data','applicability']},
            'requirements':[{'measurement':'new','blocked_decision':'fixture','competing_explanations':['a','b'],
                'cheapest_test':'fixture','development_budget':5,'acceptance_contract':'fixture'}]}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);config={'context':context}
            write_json(root/'evolution-auto.json',config)
            write_json(root/'auto-lock.json',{'config_hash':digest(config)})
            write_json(root/'review.json',review);objects=Artifacts(root/'objects')
            store=Store(root/'memory')
            try:
                store.event('measurement_handoff_recorded',{'candidate_id':'candidate','proposal_ref':objects.put(root/'review.json'),
                    'context_hash':digest(context),'measurement_gap':validate(context,review,'candidate')})
                before=store.verify()
                first=load_context(root)
                self.assertEqual(first['observations'],packet['trace_evidence'])
                self.assertEqual(load_context(root),first)
                self.assertEqual(store.verify(),before)
            finally:store.close()
