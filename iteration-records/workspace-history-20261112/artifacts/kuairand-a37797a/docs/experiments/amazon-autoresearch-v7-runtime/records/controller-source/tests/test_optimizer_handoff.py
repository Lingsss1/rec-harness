import json
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.autonomous import autoresearch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.evolution import Evolution
from rec_harness.project import ResearchProject
from rec_harness.store import Store
import test_project
from test_project import ProjectAgent


class ReferringAgent(ProjectAgent):
    def __init__(self,unresolved=False):self.calls=[];self.unresolved=unresolved
    def run(self,role,context,directory):
        self.calls.append(role)
        if role!='optimizer_router':return super().run(role,context,directory)
        packet=context['context']['measurement_handoff_request']
        review={'assessment':'measurement_gap','rationale':'Fixture: missing spread blocks a concrete comparison',
            'evidence_refs':packet['allowed_evidence_refs'][:1],
            'prerequisites':{k:{'status':'unresolved' if self.unresolved and k=='data' else 'satisfied','rationale':'fixture'} for k in ['invocation','implementation','data','applicability']},
            'requirements':[{'measurement':'spread','blocked_decision':'compare alternatives','competing_explanations':['capacity','noise'],
                'cheapest_test':'aggregate spread','development_budget':10,'acceptance_contract':'independent reference and decision utility'}]}
        return {'kind':'measurement_gap','payload_json':json.dumps(review)}


class OptimizerHandoffTests(unittest.TestCase):
    setUp=test_project.ProjectTests.setUp
    cpu=test_project.ProjectTests.cpu
    def configuration(self,tool=True):
        config=read_json(test_project.ProjectTests.config(self,'harness'))
        if tool:
            other=read_json(test_project.ProjectTests.config(self,'tool'))
            config.update(tool=other['tool'],tool_runtime=other['tool_runtime'])
        config['model']['measurement_discovery']=None
        path=self.root/'handoff.json';write_json(path,config);return path

    def test_handoff_skips_harness_validation_recovers_and_remeasures(self):
        path=self.configuration();agent=ReferringAgent();original=ResearchProject.transition
        def interrupt(project,state,episode=None):
            original(project,state,episode)
            if episode and episode['kind']=='harness':raise KeyboardInterrupt()
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=self.cpu):
            with patch.object(ResearchProject,'transition',interrupt):
                with self.assertRaises(KeyboardInterrupt):autoresearch(self.campaign,path,agent=agent)
            report=autoresearch(self.campaign,agent=agent)
            self.assertEqual([e['kind'] for e in report['episodes']],['model','harness','tool','model'])
            self.assertEqual(agent.calls.count('optimizer_router'),1)
            count=len(agent.calls);self.assertEqual(autoresearch(self.campaign,agent=agent),report);self.assertEqual(len(agent.calls),count)
        source=self.campaign/'episodes/harness-1/campaign'
        store=Store(source/'memory')
        try:
            self.assertEqual(store.budget()['charged_or_reserved'],1)
            self.assertFalse(any(e['kind'] in {'validation_passed','candidate_promoted'} for e in store.events()))
            self.assertTrue(any(n['type']=='MeasurementGap' for n in store.snapshot()['nodes']))
        finally:store.close()
        context=read_json(self.campaign/'episodes/tool-1/campaign/candidates/project-tool-1/context.json')
        self.assertIn('optimizer_event_hash',context['source'])
        self.assertEqual(context['diagnostic_contract']['entrypoint'],'diagnostic.py:measure(request)')
        self.assertEqual(context['measurement_gap']['experiment_refs'],[])
        final=read_json(self.campaign/'episodes/model-2/campaign/campaign.json')
        self.assertEqual(final['measurement_requirements'][0]['measurement'],'spread')
        self.assertNotIn('active_harness',report['state'])
        project=ResearchProject(self.campaign,agent=agent)
        try:self.assertEqual(len(project.verify()['verified_episodes']),4)
        finally:project.store.close()

    def test_unresolved_data_cannot_route_to_tool(self):
        report=autoresearch(self.campaign,self.configuration(),agent=ReferringAgent(unresolved=True))
        self.assertEqual(report['state']['stop_reason'],'harness_rejected')
        self.assertEqual([e['kind'] for e in report['episodes']],['model','harness'])

    def test_committed_handoff_recovers_before_candidate_projection(self):
        from rec_harness import evolution
        original=evolution.write_json;agent=ReferringAgent();path=self.configuration()
        def interrupt(path,value):
            if Path(path).name=='candidate.json' and value.get('status')=='measurement_required':raise KeyboardInterrupt()
            return original(path,value)
        with patch.object(evolution,'write_json',side_effect=interrupt):
            with self.assertRaises(KeyboardInterrupt):autoresearch(self.campaign,path,agent=agent)
        self.assertEqual(agent.calls.count('optimizer_router'),1)
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=self.cpu):
            report=autoresearch(self.campaign,agent=agent)
        self.assertEqual(agent.calls.count('optimizer_router'),1)
        self.assertEqual([e['kind'] for e in report['episodes']],['model','harness','tool','model'])

    def test_missing_tool_slot_retains_unresolved_gap_without_promotion(self):
        report=autoresearch(self.campaign,self.configuration(tool=False),agent=ReferringAgent())
        self.assertEqual(report['state']['stop_reason'],'measurement_unresolved')
        self.assertIn('pending_gap',report['state'])

    def test_handoff_cannot_be_promoted_or_silently_repair_forged_projection(self):
        agent=ReferringAgent();autoresearch(self.campaign,self.configuration(tool=False),agent=agent)
        root=self.campaign/'episodes/harness-1/campaign';config=read_json(root/'evolution-auto.json')
        evolution=Evolution(root)
        try:
            before=evolution.store.verify();count=len(agent.calls)
            with self.assertRaisesRegex(ContractError,'not a promotable implementation'):
                evolution.promote(config['candidate_id'])
            self.assertEqual(evolution.store.verify(),before)
            marker=root/'candidates'/config['candidate_id']/'candidate.json';saved=read_json(marker)
            committed=[e['payload'] for e in evolution.store.events() if e['kind']=='candidate_proposed'][0]
            for bad in [{**saved,'measurement_gap':{**saved['measurement_gap'],'required_information':'forged'}},
                        {**committed,'validator_hashes':{}}]:
                write_json(marker,bad)
                with self.assertRaisesRegex(ContractError,'projection changed'):
                    evolution.propose(config['candidate_id'],config['kind'],config['context'],config['plan'],agent=agent)
                self.assertEqual(read_json(marker),bad)
                self.assertEqual(evolution.store.verify(),before)
            write_json(marker,saved)
            self.assertEqual(len(agent.calls),count);evolution.verify()
        finally:evolution.store.close()

    def test_corrupt_handoff_archive_blocks_tool_export(self):
        from rec_harness.optimizer_handoff import load_context
        autoresearch(self.campaign,self.configuration(tool=False),agent=ReferringAgent())
        root=self.campaign/'episodes/harness-1/campaign';candidate=read_json(root/'candidates/project-harness-1/candidate.json')
        (root/'objects'/candidate['proposal_ref']['sha256']).write_text('{}')
        with self.assertRaisesRegex(ContractError,'corrupt artifact'):load_context(root)
