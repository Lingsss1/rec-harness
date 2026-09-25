import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import Artifacts,ContractError,digest,read_json,write_json
from rec_harness.harness_prefix import freeze_suite,run_validation,check_response


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        campaign=self.root/'campaign';write_json(campaign/'campaign.json',{'research_policy':{}})
        (campaign/'model.py').write_text('old_source = True\n');objects=Artifacts(campaign/'artifacts/objects')
        self.context={'model_version':'root','protocol_hash':'protocol','evidence':{'nodes':[{'id':'observed'}]},'baseline':{'runs':[]}}
        trace={'trace_id':'trace-one','role':'evidence','visible_state':self.context,'input_hash':digest(self.context),
            'visible_source_snapshot':objects.snapshot(campaign,['model.py']),'submitted_event':'event',
            'role_output':{'later_answer':'DO_NOT_SEND'}}
        write_json(self.root/'spec.json',{'campaign':'campaign','roles':['evidence'],'budget':4,'call_cost':1,
            'agent_backend':{'model':'fixture'},'identity':{'fixture':'prefix'}})
        with patch('rec_harness.harness_prefix.collect_traces',return_value={'decision_traces':[trace],'source':{'event_head':'prefix'}}):
            freeze_suite(self.root/'spec.json',self.root/'suite')
        write_json(self.root/'candidate/policy.json',{'prompts':{'evidence':'Consider alternatives.'}})
        self.response={'gap_id':'gap','model_version':'root','task_context':'task','protocol_hash':'protocol',
            'observation_refs':['observed'],'suspected_capability':'unknown','affected_components':[],
            'alternative_explanations':['noise','model'],'predicted_observations':['difference'],
            'cheapest_discriminating_test':'independent measurement','status':'unknown'}

    def test_prefix_replay_never_sends_original_response_and_resumes_exactly(self):
        outer=self;calls=[]
        class Agent:
            def run(self,role,context,directory):
                calls.append(role);outer.assertNotIn('DO_NOT_SEND',str(context))
                outer.assertEqual((directory/'model/model.py').read_text(),'old_source = True\n')
                return copy.deepcopy(outer.response)
        kwargs={'agent_factory':lambda *_:Agent()}
        result=run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json',**kwargs)
        self.assertTrue(result['passed']);self.assertFalse(result['online']);self.assertEqual(result['total_cost'],2)
        self.assertEqual(run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json',**kwargs),result)
        self.assertEqual(len(calls),2)
        write_json(self.root/'candidate/policy.json',{'retrieval_limit':25})
        with self.assertRaisesRegex(ContractError,'requires exact historical graph'):
            run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json',**kwargs)

    def test_stale_identity_and_future_evidence_are_rejected(self):
        for change in [{'protocol_hash':'other'},{'observation_refs':['future-result']}]:
            with self.assertRaises(ContractError):check_response('evidence',self.context,{**self.response,**change})
        decision={'selection':'keep','conclusion_status':'provisional','rationale':'fixture','evidence_refs':['observed'],'unresolved_explanations':[]}
        with self.assertRaisesRegex(ContractError,'effect gate'):check_response('lead',self.context,decision)

    def test_uncovered_patch_role_fails_before_any_call(self):
        write_json(self.root/'candidate/policy.json',{'prompts':{'experiment':'change code'}})
        with self.assertRaisesRegex(ContractError,'lack frozen prefix coverage'):
            run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json')

    def test_retrieval_rebuild_uses_only_frozen_graph_and_preserves_counterevidence(self):
        from rec_harness.harness_prefix import replay_context
        from rec_harness.runtime import node
        from rec_harness.store import Store
        store=Store(self.root/'history');self.addCleanup(store.close)
        store.patch({'base_version':0,'nodes':[node('observed','Observation',{},'task','protocol'),
            node('counter','Failure',{},'task','protocol'),node('other-task','Observation',{},'other','protocol')],'edges':[]})
        original=store.bundle('task','protocol','evidence',limit=1);graph=store.snapshot()
        case={'context':{**self.context,'evidence':original},'graph':graph}
        store.patch({'base_version':1,'nodes':[node('future','Observation',{},'task','protocol')],'edges':[]})
        context=replay_context(case,{'retrieval_limit':2},{'retrieval_limit':1})
        ids={n['id'] for n in context['evidence']['nodes']}
        self.assertEqual(ids,{'observed','counter'})
        self.assertEqual(context['evidence']['graph_snapshot_id'],1)
        self.assertNotIn('future',str(context));self.assertNotIn('other-task',str(context))
        self.assertEqual(len(original['nodes']),1)
        broken=copy.deepcopy(case);broken['graph']=store.snapshot()
        with self.assertRaisesRegex(ContractError,'binding changed'):
            replay_context(broken,{'retrieval_limit':2},{'retrieval_limit':1})

    def test_empty_source_snapshot_does_not_create_model_directory(self):
        suite=read_json(self.root/'suite/suite.json');suite['cases'][0]['source']={}
        suite['hash']=digest({k:v for k,v in suite.items() if k!='hash'});write_json(self.root/'suite/suite.json',suite)
        outer=self
        class Agent:
            def run(self,role,context,directory):
                if (directory/'model').exists():raise ContractError('unexpected empty model directory')
                return copy.deepcopy(outer.response)
        result=run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json',agent_factory=lambda *_:Agent())
        self.assertTrue(result['passed'])

    def test_real_transport_path_forces_search_disabled(self):
        suite=read_json(self.root/'suite/suite.json')
        suite['agent_backend'].update({'template':'unused','runtime_root':'unused','web_search':'live'})
        suite['hash']=digest({k:v for k,v in suite.items() if k!='hash'})
        write_json(self.root/'suite/suite.json',suite)
        with patch('rec_harness.harness_prefix.JailedCodexAgent') as transport:
            transport.return_value.run.return_value=copy.deepcopy(self.response)
            result=run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json')
        self.assertTrue(result['passed'])
        self.assertEqual(transport.call_count,2)
        for call in transport.call_args_list:
            self.assertEqual(call.kwargs['web_search'],'disabled')
        self.assertEqual(result['binding']['web_search'],'disabled')
        self.assertIn('agents.py',result['binding']['transport_hashes'])
