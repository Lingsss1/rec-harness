"""Control-flow fixtures: a zero-gain report never establishes online utility."""
import unittest
import copy
from unittest.mock import patch
from rec_harness.autonomous import autoresearch
from rec_harness.core import ContractError,digest,read_json,write_json
from rec_harness.project import ResearchProject
from rec_harness.store import Store
import test_project


class RejectionContinuationTests(unittest.TestCase):
    setUp=test_project.ProjectTests.setUp

    def test_pending_validator_does_not_commit_project_terminal_or_repeat_optimizer(self):
        from rec_harness.validation_stage import ValidationRecoveryRequired
        with patch('rec_harness.evolution.run_process',side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):autoresearch(self.campaign,self.config(),agent=self.agent)
        project_store=Store(self.campaign/'memory',read_only=True)
        try:before=project_store.verify()
        finally:project_store.close()
        with patch.object(test_project.ProjectAgent,'run',side_effect=AssertionError('replayed role')):
            for _ in range(2):
                with self.assertRaises(ValidationRecoveryRequired):autoresearch(self.campaign,agent=self.agent)
        project=ResearchProject(self.campaign,agent=self.agent)
        try:
            self.assertEqual(project.store.verify(),before)
            state=[e['payload']['state'] for e in project.store.events() if e['kind']=='project_transition'][-1]
            self.assertEqual(state['phase'],'harness');self.assertNotEqual(state['status'],'stopped')
            self.assertEqual(len(project.verify()['verified_episodes']),1)
        finally:project.store.close()

    def test_report_commit_interruption_resumes_from_project_entry(self):
        original=Store.event
        def interrupt(store,kind,value):
            if kind=='validation_passed':raise KeyboardInterrupt()
            return original(store,kind,value)
        with patch.object(Store,'event',interrupt):
            with self.assertRaises(KeyboardInterrupt):autoresearch(self.campaign,self.config(),agent=self.agent)
        class NoOptimizer(test_project.ProjectAgent):
            def run(agent,role,context,directory):
                if role.startswith('optimizer'):raise AssertionError('replayed optimizer')
                return super().run(role,context,directory)
        from rec_harness.agents import run_process
        with patch('rec_harness.evolution.run_process',wraps=run_process) as runner:
            report=autoresearch(self.campaign,agent=NoOptimizer())
        self.assertEqual(runner.call_count,1)  # Only the zero-gain development stage remains.
        self.assertEqual([e['kind'] for e in report['episodes']],['model','harness','model'])
        child=read_json(self.campaign/'episodes/harness-1/campaign/run-report.json')
        self.assertEqual(child['rejection_kind'],'validation_gate')
        self.assertEqual(child['budget']['charged_or_reserved'],3)

    def config(self,*,second_only=False,invalid=False):
        path=test_project.ProjectTests.config(self,'harness');config=read_json(path)
        config['model']['research_policy']={'prompts':{'evidence':'Retain original guidance'}}
        if second_only:config.update(max_model_episodes=3,max_harness_updates=2)
        script=self.root/'development.py'
        identity=config['harness']['plan']['identity']
        report={'passed':True,'gain':0,'regression':0,'total_cost':7,
                'evaluation_ids':['project-model-1' if invalid else 'independent-development'],
                'identity':identity,'comparison_groups':['baseline','candidate'],'online':True}
        script.write_text('import sys,json\nfrom pathlib import Path\nreport='+repr(report)+'\n'
            + ('if "harness-1" in str(Path.cwd()): report["gain"]=1\n' if second_only else '')
            +'Path(sys.argv[1]).write_text(json.dumps(report))\n')
        write_json(path,config);return path

    def test_gate_rejection_continues_original_policy_and_terminal_resume_is_idle(self):
        report=autoresearch(self.campaign,self.config(),agent=self.agent)
        self.assertEqual([e['kind'] for e in report['episodes']],['model','harness','model'])
        self.assertEqual(report['state']['harness_updates'],1)
        self.assertNotIn('active_harness',report['state'])
        child=self.campaign/'episodes/harness-1/campaign'
        rejected=read_json(child/'run-report.json')
        self.assertEqual(rejected['rejection_kind'],'validation_gate')
        self.assertEqual(rejected['rejected_stage'],'development')
        self.assertFalse((child/'active-harness.json').exists())
        final=read_json(self.campaign/'episodes/model-2/campaign/campaign.json')
        self.assertEqual(final['research_policy'],{'prompts':{'evidence':'Retain original guidance'}})
        store=Store(child/'memory',read_only=True)
        try:
            event=next(e['payload'] for e in store.events() if e['kind']=='validation_rejected')
            self.assertEqual(event['report_hash'],digest(event['report']))
            self.assertEqual(event['report']['gain'],0)
        finally:store.close()
        with patch.object(test_project.ProjectAgent,'run',side_effect=AssertionError('replayed role')):
            self.assertEqual(autoresearch(self.campaign,agent=self.agent),report)

    def test_second_rejection_preserves_promoted_policy_across_transition_interruption(self):
        transition=ResearchProject.transition
        def interrupt(project,state,episode=None):
            transition(project,state,episode)
            if episode and episode['kind']=='harness' and episode['index']==2:raise KeyboardInterrupt()
        with patch.object(ResearchProject,'transition',interrupt):
            with self.assertRaises(KeyboardInterrupt):autoresearch(self.campaign,self.config(second_only=True),agent=self.agent)
        class NoMoreOptimizers(test_project.ProjectAgent):
            def run(agent,role,context,directory):
                if role.startswith('optimizer'):raise AssertionError('replayed optimizer')
                return super().run(role,context,directory)
        report=autoresearch(self.campaign,agent=NoMoreOptimizers())
        self.assertEqual([e['kind'] for e in report['episodes']],['model','harness','model','harness','model'])
        self.assertEqual(report['state']['active_harness'],'episodes/harness-1/campaign')
        final=read_json(self.campaign/'episodes/model-3/campaign/campaign.json')
        self.assertEqual(final['research_policy'],{'prompts':{'evidence':'Retain original guidance'},'retrieval_limit':25})
        project=ResearchProject(self.campaign,agent=self.agent)
        try:self.assertEqual(len(project.verify()['verified_episodes']),5)
        finally:project.store.close()

    def test_invalid_report_is_not_a_gate_rejection(self):
        report=autoresearch(self.campaign,self.config(invalid=True),agent=self.agent)
        self.assertEqual(report['state']['stop_reason'],'harness_rejected')
        self.assertEqual(len(report['episodes']),2)
        rejected=read_json(self.campaign/'episodes/harness-1/campaign/run-report.json')
        self.assertNotIn('rejection_kind',rejected)
        self.assertIn('overlaps',rejected['reason'])

    def test_explicit_stop_and_legacy_frozen_project_still_stop(self):
        for legacy in [False,True]:
            with self.subTest(legacy=legacy):
                config=read_json(self.config());config['continue_after_harness_rejection']=False
                target=self.root/('legacy' if legacy else 'explicit-stop')
                ResearchProject.init(target,config)
                if legacy:
                    frozen=read_json(target/'project.json');frozen.pop('continue_after_harness_rejection')
                    write_json(target/'project.json',frozen)
                    write_json(target/'project-lock.json',{'config_hash':digest(frozen)})
                report=autoresearch(target,agent=self.agent)
                self.assertEqual(report['state']['stop_reason'],'harness_rejected')
                self.assertEqual(len(report['episodes']),2)

    def test_policy_requires_boolean_before_initialization(self):
        config=read_json(self.config());config['continue_after_harness_rejection']='yes'
        with self.assertRaises(ContractError):ResearchProject.init(self.campaign,config)
        self.assertFalse(self.campaign.exists())

    def test_invalid_conditions_never_classified_as_numerical_rejection(self):
        from rec_harness.evolution import Evolution,ValidationGateRejected
        plan=read_json(self.config())['harness']['plan']
        report={'passed':True,'gain':0,'regression':0,'total_cost':7,
                'evaluation_ids':['independent'],'identity':plan['identity'],
                'comparison_groups':['baseline','candidate'],'online':True}
        evolution=Evolution(self.root/'report-check')
        try:
            with self.assertRaises(ValidationGateRejected):
                evolution._check_report({'kind':'harness'},plan,'development',report)
            for key,value in [('evaluation_ids',[]),('evaluation_ids',['source']),
                              ('comparison_groups',['candidate']),('online',False),
                              ('passed',False),('identity',{}),('gain',float('nan')),
                              ('total_cost',-1)]:
                with self.subTest(field=key,value=value):
                    changed=copy.deepcopy(report);changed[key]=value
                    with self.assertRaises((ContractError,ValueError)) as caught:
                        evolution._check_report({'kind':'harness'},plan,'development',changed)
                    self.assertNotIsInstance(caught.exception,ValidationGateRejected)
        finally:evolution.store.close()
