import json
import unittest
from rec_harness.agents import schema_for
from rec_harness.amazon_prefix import check_response
from rec_harness.core import ContractError
from test_amazon_plan import sample


class CompactPrefixTests(unittest.TestCase):
    def test_researcher_checks_internal_handoff_and_visible_graph_citations(self):
        value=sample(schema_for('researcher'))
        bindings={'proposal_id':'p','base_model_id':'base','controls':['base'],'train_protocol_hash':'protocol','seeds':[42,17]}
        context={'bindings':bindings,'budget':{'remaining':10},'evidence_graph':{'nodes':[{'id':'fact'}]}}
        value['gap'].update(gap_id='g',model_version='base',protocol_hash='protocol',observation_refs=['fact'])
        value['method'].update(method_id='m',origin='agent_hypothesis')
        value['proposal'].update(**bindings,gap_ids=['g'],method_ids=['m'],budget=1)
        value['evidence_refs']=['fact'];case={'role':'researcher','require_method_sources':False}
        check_response(case,context,value)
        value['evidence_refs']=['future']
        with self.assertRaisesRegex(ContractError,'outside visible graph'):check_response(case,context,value)

    def test_reviewer_does_not_need_hypothesis_bindings_and_rejection_is_valid(self):
        value={'verdict':'request_changes','findings':['wrong intervention'],'evidence_refs':['codehash'],'limitations':[]}
        check_response({'role':'reviewer'},{'implementation_hash':'codehash'},value)
        value['verdict']='approve'
        with self.assertRaisesRegex(ContractError,'blocking'):check_response({'role':'reviewer'},{'implementation_hash':'codehash'},value)

    def test_analyst_checks_both_readout_and_graph_patch(self):
        response=sample(schema_for('analyst'))
        context={'bindings':{},'development':{'hash':'readout','summary':{'eligible_for_confirmation':True}},
                 'graph':{'graph_version':0,'task_id':'task','protocol_hash':'protocol','nodes':[],'edges':[]}}
        response['decision'].update(selection='keep',conclusion_status='unknown',evidence_refs=['readout'])
        response['knowledge']['patch_json']=json.dumps({'base_version':0,'nodes':[],'edges':[]})
        check_response({'role':'analyst'},context,response)
        response['knowledge']['patch_json']=json.dumps({'base_version':1,'nodes':[],'edges':[]})
        with self.assertRaises(ContractError):check_response({'role':'analyst'},context,response)
