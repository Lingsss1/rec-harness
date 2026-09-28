import copy
import unittest
from unittest.mock import patch

import test_amazon_tool_online as fixture
from rec_harness import amazon_harness_online as online
from rec_harness.amazon_tool_online import episode
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.online_recovery import completed_groups
from rec_harness.validation_preflight import check_suite,evaluation_ids


class AmazonHarnessOnlineTests(unittest.TestCase):
    def setUp(self):
        fixture.AmazonToolOnlineTests.setUp(self)
        self.spec={k:v for k,v in self.spec.items() if k in {'cost_weights','cases'}}
        self.spec['cost_weights'].pop('diagnostic_units')
        self.spec['baseline_policy']={'retrieval_limit':5}
        self.candidate=self.root/'candidate';self.candidate.mkdir()
        self.policy={'retrieval_limit':10,'prompts':{'hypothesis':'Consider counterevidence.'}}
        write_json(self.candidate/'policy.json',self.policy)

    def freeze(self):
        write_json(self.root/'spec.json',self.spec)
        online.freeze_suite(self.root/'spec.json',self.root/'suite')
        return self.root/'suite/suite.json'

    def result(self,group):
        return {'score':0.6 if group=='candidate' else 0.5,'protocol_hash':'p',
            'evaluation_conditions':[{'population':'fixed'}],'diagnostic_calls':0,'diagnostic_failures':0,
            'costs':{'role_calls':{'charged_or_reserved':6},'gpu_seconds':{'charged_or_reserved':10}}}

    def test_freeze_without_tool_budget_and_preflight_partitions(self):
        for path in self.configs:
            config=read_json(path);config.pop('diagnostic_budget');write_json(path,config)
        path=self.freeze();suite,_=online.load_suite(path)
        self.assertEqual(check_suite('rec_harness.amazon_harness_online',path),suite)
        self.assertEqual(evaluation_ids('rec_harness.amazon_harness_online',path,'development'),{'development'})
        self.assertEqual(evaluation_ids('rec_harness.amazon_harness_online',path,'confirmation'),{'confirmation'})
        with self.assertRaises(ContractError):evaluation_ids('rec_harness.amazon_harness_online',path,'utility')
        with patch.object(online,'code_identity',return_value={}):
            with self.assertRaisesRegex(ContractError,'implementation changed'):online.load_suite(path)
        (self.root/'data/A_valid.txt').write_text('changed')
        with self.assertRaisesRegex(ContractError,'private data identity'):online.load_suite(path)

    def test_reused_seed_and_policy_kernel_changes_rejected_before_execution(self):
        self.spec['cases'][1].update(development_seeds=[100],confirmation_seeds=[111])
        with self.assertRaisesRegex(ContractError,'reuse a training condition'):self.freeze()
        self.spec['baseline_policy']={'score_function':'changed'}
        write_json(self.root/'spec.json',self.spec)
        with self.assertRaisesRegex(ContractError,'stable control kernel'):
            online.freeze_suite(self.root/'spec.json',self.root/'other-suite')

    def test_pair_receives_exact_policies_and_separate_roots(self):
        path=self.freeze();calls=[]
        def run(case,root,objects,tool,**kwargs):
            calls.append((case,root,tool));return self.result(root.name)
        with patch.object(online,'episode',side_effect=run),patch.object(online,'verify_group',side_effect=lambda root,*a,**kw:self.result(root.name)):
            report=online.run_validation(path,self.candidate,self.root/'dev/report.json','development',agent_factory=lambda *_:object())
            self.assertFalse(report['online']);self.assertAlmostEqual(report['gain'],0.1)
            self.assertAlmostEqual(report['total_cost'],12.2)
            self.assertEqual(report['cost_axes'],{'role_calls':12,'gpu_seconds':20})
            self.assertEqual(calls[0][0]['controller']['research_policy'],self.spec['baseline_policy'])
            self.assertEqual(calls[1][0]['controller']['research_policy'],self.policy)
            self.assertNotEqual(calls[0][1],calls[1][1]);self.assertTrue(all(row[2] is None for row in calls))
            online.run_validation(path,self.candidate,self.root/'confirm/report.json','confirmation',agent_factory=lambda *_:object())
            self.assertEqual([row[0]['id'] for row in calls],['development','development','confirmation','confirmation'])
            with self.assertRaisesRegex(ContractError,'binding changed'):
                online.run_validation(path,self.candidate,self.root/'dev/report.json','confirmation',agent_factory=lambda *_:object())

    def test_actual_controller_policy_binding_and_completed_reuse(self):
        suite,objects=online.load_suite(self.freeze());case=suite['cases'][0]
        adapted=online.policy_case(case,self.policy);root=self.root/'development/candidate'
        saved=self.result('candidate')
        with patch('rec_harness.amazon_tool_online.run_study'),patch('rec_harness.amazon_tool_online.AmazonAutoresearch._rescore'), \
             patch('rec_harness.amazon_tool_online.AmazonAutoresearch.run'),patch('rec_harness.amazon_tool_online.read_result',return_value=saved):
            episode(adapted,root,objects,None,agent=object())
            with patch('rec_harness.amazon_tool_online.create_study',side_effect=AssertionError('duplicate study')):
                self.assertEqual(episode(adapted,root,objects,None,agent=object()),saved)
        with patch.object(online,'read_result',return_value=saved),patch.object(online,'real_transports',side_effect=AssertionError('scripted')):
            self.assertEqual(online.verify_group(root,case,self.policy,scripted=True),saved)
            with self.assertRaisesRegex(ContractError,'launch binding differs'):
                online.verify_group(root,case,{},scripted=True)
            config=read_json(root/'campaign/amazon-auto.json');config['research_policy']={}
            write_json(root/'campaign/amazon-auto.json',config)
            with self.assertRaisesRegex(ContractError,'controller differs'):
                online.verify_group(root,case,self.policy,scripted=True)

    def test_real_path_requires_transport_verification(self):
        with patch.object(online,'check_binding',return_value={'agent_backend':{'model':'fixed'}}), \
             patch.object(online,'read_result',return_value=self.result('baseline')), \
             patch.object(online,'read_json',return_value=self.result('baseline')), \
             patch.object(online,'real_transports',side_effect=ContractError('missing actual transport')) as transport:
            with self.assertRaisesRegex(ContractError,'missing actual transport'):
                online.verify_group(self.root/'baseline',{'controller':{}},{})
            self.assertEqual(transport.call_count,1)

    def test_report_recomputed_and_tampered_score_rejected(self):
        path=self.freeze();output=self.root/'dev/report.json'
        with patch.object(online,'episode',side_effect=lambda c,r,*a,**kw:self.result(r.name)), \
             patch.object(online,'verify_group',side_effect=lambda r,*a,**kw:self.result(r.name)), \
             patch.object(online,'real_transports',side_effect=AssertionError('no new calls')):
            original=online.run_validation(path,self.candidate,output,'development',agent_factory=lambda *_:object())
            self.assertEqual(online.audit_report(path,output),original)
            changed=copy.deepcopy(original);changed['gain']=0.5;write_json(output,changed)
            with self.assertRaisesRegex(ContractError,'committed evidence'):online.audit_report(path,output)

    def test_population_mismatch_and_unpriced_costs_rejected(self):
        suite,_=online.load_suite(self.freeze());binding={'scripted_agent':True}
        row={'evaluation_id':'development',**{g:self.result(g) for g in online.GROUPS}}
        row['candidate']['evaluation_conditions']=[{'population':'other'}]
        with self.assertRaisesRegex(ContractError,'populations differ'):online.summarize(suite,binding,[row])
        row['candidate']['evaluation_conditions']=row['baseline']['evaluation_conditions']
        row['candidate']['costs']['unknown_axis']={'charged_or_reserved':1}
        with self.assertRaisesRegex(ContractError,'unpriced cost axis'):online.summarize(suite,binding,[row])

    def test_recovery_checks_initialized_amazon_harness_groups(self):
        target=self.root/'dev';folder=target/'amazon-harness-online/development/baseline';folder.mkdir(parents=True)
        with self.assertRaisesRegex(ContractError,'outcome unknown'):
            completed_groups(target,'rec_harness.amazon_harness_online','development')
        result=self.result('baseline');write_json(folder/'completed.json',result)
        with patch('rec_harness.amazon_tool_online.read_result',return_value=result):
            completed_groups(target,'rec_harness.amazon_harness_online','development')
        with patch('rec_harness.amazon_tool_online.read_result',return_value={}):
            with self.assertRaisesRegex(ContractError,'comparison changed'):
                completed_groups(target,'rec_harness.amazon_harness_online','development')


if __name__=='__main__':unittest.main()
