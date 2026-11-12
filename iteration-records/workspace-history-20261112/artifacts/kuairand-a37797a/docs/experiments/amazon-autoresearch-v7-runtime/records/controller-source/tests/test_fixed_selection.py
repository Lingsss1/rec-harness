import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.fixed_selection import freeze_suite,run_validation
from rec_harness.store import Store


class FixedSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.spec={'direction':'maximize','call_cost':1,'budget':6,
            'agent_backend':{'model':'fixture','template':'unused','runtime_root':'unused-runtime'},'cases':[]}
        for stage in ['development','confirmation']:
            self.spec['cases'].append({'id':stage,'stage':stage,'context':{'task':'select one'},
                'candidates':[{'candidate_id':'a','public':{'width':1}},{'candidate_id':'b','public':{'width':2}}],
                'scores':{'a':0.25,'b':0.75},'observations':{'no_tool':{},
                    'existing_tool':{'a':{'signal':0},'b':{'signal':0}},
                    'candidate_tool':{'a':{'signal':0},'b':{'signal':1}}}})
        self.calls=[]

    def freeze(self):
        write_json(self.root/'spec.json',self.spec)
        freeze_suite(self.root/'spec.json',self.root/'suite')
        return self.root/'suite/suite.json'

    def agents(self,case,group):
        outer=self
        class Selector:
            def run(self,role,context,directory):
                outer.calls.append((case,group,context))
                outer.assertEqual(role,'selection')
                outer.assertNotIn('scores',context);outer.assertNotIn('scores',context['context'])
                outer.assertEqual({c['candidate_id'] for c in context['candidates']},{'a','b'})
                return {'candidate_id':'b' if group=='candidate_tool' else 'a','rationale':'fixture'}
        return Selector()

    def test_paired_choices_scored_only_by_host_and_recovery_reuses_decisions(self):
        suite=self.freeze();output=self.root/'dev/report.json'
        result=run_validation(suite,output,'development',agent_factory=self.agents)
        self.assertEqual(result['evaluation_mode'],'fixed_candidate_selection')
        self.assertFalse(result['online']);self.assertFalse(result['live_decisions'])
        self.assertEqual(result['gain'],0.5);self.assertEqual(result['total_cost'],3)
        self.assertEqual(result['rows'][0]['candidate_tool']['regret'],0)
        self.assertEqual(len(self.calls),3)
        self.assertEqual(run_validation(suite,output,'development',agent_factory=self.agents),result)
        self.assertEqual(len(self.calls),3)
        confirmation=run_validation(suite,self.root/'confirmation/report.json','confirmation',agent_factory=self.agents)
        self.assertEqual(confirmation['evaluation_ids'],['confirmation']);self.assertEqual(len(self.calls),6)
        self.assertTrue(all(case=='development' for case,_,_ in self.calls[:3]))

    def test_unknown_submitted_call_is_not_repeated_or_refunded(self):
        suite=self.freeze();output=self.root/'dev/report.json'
        class Interrupted:
            def run(self,*args):raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):run_validation(suite,output,'development',agent_factory=lambda *_:Interrupted())
        with self.assertRaisesRegex(ContractError,'unknown'):
            run_validation(suite,output,'development',agent_factory=self.agents)
        self.assertEqual(self.calls,[])
        store=Store(self.root/'dev/fixed-selection/memory',read_only=True)
        try:self.assertEqual(store.budget()['charged_or_reserved'],1)
        finally:store.close()

    def test_committed_response_survives_interruption_before_settlement(self):
        suite=self.freeze();output=self.root/'dev/report.json'
        with patch('rec_harness.fixed_selection.settle_once',side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):run_validation(suite,output,'development',agent_factory=self.agents)
        self.assertEqual(len(self.calls),1)
        result=run_validation(suite,output,'development',agent_factory=self.agents)
        self.assertEqual(len(self.calls),3);self.assertEqual(result['total_cost'],3)

    def test_out_of_pool_response_rejected_and_never_scores_as_zero(self):
        suite=self.freeze()
        class Invalid:
            def run(self,*args):return {'candidate_id':'invented','rationale':'fixture'}
        report=run_validation(suite,self.root/'dev/report.json','development',agent_factory=lambda *_:Invalid())
        self.assertFalse(report['passed']);self.assertEqual(report['gain_by_reference'],{})
        self.assertIsNone(report['rows'][0]['candidate_tool']['score'])

    def test_minimization_and_suite_tampering(self):
        self.spec['direction']='minimize';suite=self.freeze()
        report=run_validation(suite,self.root/'dev/report.json','development',agent_factory=self.agents)
        self.assertEqual(report['gain'],-0.5);self.assertEqual(report['regression'],0.5)
        altered=read_json(suite);altered['cases'][0]['scores']['a']=100;write_json(suite,altered)
        with self.assertRaisesRegex(ContractError,'suite changed'):
            run_validation(suite,self.root/'dev/report.json','development',agent_factory=self.agents)

    def test_wrong_pool_observation_coverage_rejected_before_freeze(self):
        self.spec['cases'][0]['observations']['candidate_tool'].pop('b')
        with self.assertRaisesRegex(ContractError,'same fixed candidate pool'):self.freeze()
        self.assertFalse((self.root/'suite/suite.json').exists())

    def test_default_backend_uses_isolated_codex_with_supplied_context_only(self):
        suite=self.freeze()
        with patch('rec_harness.fixed_selection.JailedCodexAgent') as backend:
            backend.return_value.run.return_value={'candidate_id':'a','rationale':'transport fixture'}
            report=run_validation(suite,self.root/'dev/report.json','development')
            self.assertTrue(report['live_decisions']);self.assertFalse(report['online'])
            self.assertEqual(backend.return_value.run.call_count,3)
            for call in backend.call_args_list:
                self.assertEqual(call.kwargs['web_search'],'disabled')
            for call in backend.return_value.run.call_args_list:
                self.assertEqual(call.args[0],'selection')
                self.assertNotIn('scores',call.args[1])
                self.assertNotIn('agent_backend',call.args[1])

    def test_changed_implementation_refuses_before_any_role(self):
        suite=self.freeze()
        with patch('rec_harness.fixed_selection.implementation_hashes',return_value={}):
            with self.assertRaisesRegex(ContractError,'implementation changed'):
                run_validation(suite,self.root/'dev/report.json','development',agent_factory=self.agents)
        self.assertEqual(self.calls,[])
