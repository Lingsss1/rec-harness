import copy
from pathlib import Path
from unittest.mock import patch
import unittest
import test_diagnostics
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.runtime import node
from rec_harness.measurement_discovery import discover,review_context,validate_review
from rec_harness.measurement_gaps import MeasurementDeferred


class DiscoveryTests(unittest.TestCase):
    def test_review_preserves_tool_scope_without_runtime_paths(self):
        from rec_harness.diagnostics import measure
        registration={'schema':'validated-diagnostic-registration-v1',
            'validated_conditions':{'protocol':'frozen-protocol'},
            'usage':{'interpretation':'aggregate measurement only'}}
        self.research.config['diagnostic_tool']['registration']=registration
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=lambda *a,**kw:test_diagnostics.DiagnosticTests.execute(self,*a,**kw)):
            diagnostic=measure(self.research,self.state)
        context={'diagnostic':{**diagnostic,'private_path':'excluded'}}
        request=review_context(self.research,self.state,context,self.gap)
        self.assertEqual(request['diagnostic']['tool_registration'],registration)
        self.assertEqual(request['diagnostic']['output'],diagnostic['output'])
        self.assertNotIn('private_path',str(request))
        self.assertNotIn('artifacts',request['diagnostic'])

    def test_existing_baseline_metric_is_not_a_missing_measurement(self):
        request=review_context(self.research,self.state,self.context,self.gap)
        request['baseline']=[{'metrics':{'HR@10':0.2}},{'metrics':{'HR@10':0.3}}]
        response=copy.deepcopy(self.response);response['requirements'][0]['measurement']='HR@10'
        with self.assertRaisesRegex(ContractError,'already exists'):validate_review(request,response)
        # A differently defined quantity, or missing condition, is not silently
        # declared covered just because another row has a similarly named metric.
        response['requirements'][0]['measurement']='HR@10.seed_spread'
        validate_review(request,response)
        response['requirements'][0]['measurement']='HR@10';request['baseline'][1]['metrics']={}
        validate_review(request,response)

    def setUp(self):
        test_diagnostics.DiagnosticTests.setUp(self)
        r=self.research
        r.config['measurement_discovery']={'development_budget':10}
        self.state['baseline']['runs'][0]['run_id']='run-one'
        r.store.patch({'base_version':r.store.version,'nodes':[node('run-one','Experiment',{'valid':True},r.task,r.protocol_hash)],'edges':[]})
        self.gap={'gap_id':'model-gap','alternative_explanations':['capacity','noise']}
        self.context={'diagnostic':None}
        self.response={'assessment':'measurement_gap','rationale':'Aggregate mean does not show whether seed variation exceeds the proposed model effect.',
            'evidence_refs':['run-one'], 'prerequisites':{k:{'status':'satisfied','rationale':'Fixture prerequisite evidence'} for k in ['invocation','implementation','data','applicability']},
            'requirements':[{'measurement':'spread','blocked_decision':'choose architecture or retain incumbent',
                'competing_explanations':['capacity','noise'],'cheapest_test':'compute range over supplied runs',
                'development_budget':5,'acceptance_contract':'positive/null host reference and independent decision utility'}]}

    def test_unregistered_gap_enters_graph_and_recovery_does_not_repeat_call(self):
        calls=[]
        def invoke(request):
            calls.append(request)
            self.assertNotIn('private_path',str(request));return copy.deepcopy(self.response)
        gaps=discover(self.research,self.state,self.context,self.gap,'attempt',invoke)
        before=self.research.store.verify()
        self.assertEqual(discover(self.research,self.state,self.context,self.gap,'attempt',invoke),gaps)
        self.assertEqual(self.research.store.verify(),before);self.assertEqual(len(calls),1)
        self.assertEqual(gaps[0]['required_information'],'spread')
        self.assertEqual(gaps[0]['related_gap_ids'],['attempt-gap'])
        n=next(n for n in self.research.store.snapshot()['nodes'] if n['type']=='MeasurementGap')
        self.assertEqual(n['attrs']['origin'],'agent_hypothesis')
        self.assertEqual(len([e for e in self.research.store.events() if e['kind']=='measurement_reviewed']),1)

    def test_failed_measurements_never_trigger_tool_discovery(self):
        for state,context in [(self.state,{'diagnostic':{'status':'invalid'}}),
            ({'incumbent':'root','baseline':{'runs':[]}},self.context)]:
            with self.assertRaises(MeasurementDeferred):
                discover(self.research,state,context,self.gap,'a',lambda _:self.fail('must not call Codex'))

    def test_unresolved_prerequisites_and_sufficient_evidence_do_not_create_gap(self):
        for assessment in ['sufficient','prerequisite_unresolved']:
            response={**self.response,'assessment':assessment,'requirements':[]}
            gap={**self.gap,'gap_id':assessment}
            if assessment=='prerequisite_unresolved':
                with self.assertRaises(MeasurementDeferred):discover(self.research,self.state,self.context,gap,'a',lambda _:response)
            else:self.assertEqual(discover(self.research,self.state,self.context,gap,'a',lambda _:response),[])
        self.assertFalse(any(n['type']=='MeasurementGap' for n in self.research.store.snapshot()['nodes']))

    def test_host_rejects_unknown_evidence_budget_and_already_available_measurement(self):
        request=review_context(self.research,self.state,self.context,self.gap)
        for field,value in [('evidence_refs',['future']),('rationale','')]:
            response=copy.deepcopy(self.response);response[field]=value
            with self.assertRaises(ContractError):validate_review(request,response)
        response=copy.deepcopy(self.response);response['requirements'][0]['development_budget']=11
        with self.assertRaises(ContractError):validate_review(request,response)
        request['diagnostic']={'output':{'measurements':{'spread':1}}}
        with self.assertRaises(ContractError):validate_review(request,self.response)

    def test_gap_cannot_ignore_unresolved_or_inapplicable_data(self):
        request=review_context(self.research,self.state,self.context,self.gap)
        for status in ['unresolved','not_applicable']:
            response=copy.deepcopy(self.response)
            response['prerequisites']['data']['status']=status
            with self.assertRaisesRegex(ContractError,'prerequisite'):
                validate_review(request,response)

    def activation_context(self):
        row={'observation_id':'activation-one','model_version':self.state['incumbent'],
            'dataset':'A','seed':42,'measurements':{'block_0.activation_rms':1.},
            'limitations':['synthetic histories only'],'scope':'diagnostic observation only'}
        self.research.store.patch({'base_version':self.research.store.version,'nodes':[
            node(row['observation_id'],'Observation',{**row,'valid':True},self.research.task,self.research.protocol_hash)],'edges':[]})
        return {'diagnostic':None,'activation_diagnostics':[row]}

    def test_existing_activation_measurement_cannot_trigger_duplicate_tool(self):
        context=self.activation_context()
        request=review_context(self.research,self.state,context,self.gap)
        self.assertIn('activation-one',request['allowed_evidence_refs'])
        response=copy.deepcopy(self.response)
        response['evidence_refs']=['activation-one']
        response['requirements'][0]['measurement']='block_0.activation_rms'
        with self.assertRaisesRegex(ContractError,'already exists'):
            discover(self.research,self.state,context,self.gap,'attempt',lambda _:response)
        self.assertFalse(any(n['type']=='MeasurementGap' for n in self.research.store.snapshot()['nodes']))
        response['requirements'][0]['measurement']='real_history.activation_rms'
        validate_review(request,response)

    def test_activation_context_must_match_committed_current_model_observation(self):
        context=self.activation_context()
        for key,value in [('observation_id','unknown'),('model_version','other'),
                ('measurements',{'block_0.activation_rms':99.}),('limitations',[]),('extra','untracked')]:
            altered=copy.deepcopy(context);altered['activation_diagnostics'][0][key]=value
            with self.subTest(field=key),self.assertRaises(ContractError):
                review_context(self.research,self.state,altered,self.gap)

    def test_block_scores_require_committed_observation_and_prevent_duplicate_measurement(self):
        row={'observation_id':'block-score-one','model_version':self.state['incumbent'],
            'dataset':'A','seed':42,'measurements':{'A.block-0.delta.HR@10':-.01},
            'limitations':['Fixed weights; not retrained deletion effects.'],'scope':'development block interventions'}
        context={'diagnostic':None,'block_score_diagnostics':[row]}
        with self.assertRaisesRegex(ContractError,'current-model observation'):
            review_context(self.research,self.state,context,self.gap)
        self.research.store.patch({'base_version':self.research.store.version,'nodes':[
            node(row['observation_id'],'Observation',{**row,'valid':True},self.research.task,self.research.protocol_hash)],'edges':[]})
        request=review_context(self.research,self.state,context,self.gap)
        self.assertIn(row['observation_id'],request['allowed_evidence_refs'])
        response=copy.deepcopy(self.response);response['evidence_refs']=[row['observation_id']]
        response['requirements'][0]['measurement']='A.block-0.delta.HR@10'
        with self.assertRaisesRegex(ContractError,'already exists'):validate_review(request,response)
        changed=copy.deepcopy(context);changed['block_score_diagnostics'][0]['measurements']['A.block-0.delta.HR@10']=1.
        with self.assertRaisesRegex(ContractError,'committed observation'):
            review_context(self.research,self.state,changed,self.gap)
