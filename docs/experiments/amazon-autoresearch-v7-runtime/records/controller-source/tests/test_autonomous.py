import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rec_harness.autonomous import autoresearch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.runtime import Research
from rec_harness.store import Store
from rec_harness.toy_isolated import samples,score_response
from test_runtime import ScriptedAgent,ROOT


class AutonomousTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.campaign=self.root/'campaign'
        self.config=read_json(ROOT/'examples/toy/campaign.json')
        self.config.update(source=str(ROOT/'examples/toy'),max_rounds=3,max_role_retries=1)

    def start(self,agent):
        path=self.root/'config.json';write_json(path,self.config)
        return autoresearch(self.campaign,path,agent=agent)

    def test_multiple_rounds_stop_at_budget_and_resume_without_new_cost(self):
        result=self.start(ScriptedAgent())
        self.assertEqual(result['state']['round'],2)
        self.assertEqual(result['state']['stop_reason'],'budget')
        again=autoresearch(self.campaign,agent=ScriptedAgent())
        self.assertEqual(again['budget'],result['budget'])
        self.assertEqual(again['state'],result['state'])
        self.assertEqual(again['integrity'],result['integrity'])

    def test_host_effect_is_available_before_agent_selection(self):
        observed=[]
        class Inspect(ScriptedAgent):
            def run(self,role,context,directory):
                if role=='evaluation':observed.append(context['comparison'])
                return super().run(role,context,directory)
        self.config['max_rounds']=1
        result=self.start(Inspect())
        self.assertTrue(observed[0]['effect_gate_passed'])
        self.assertGreater(observed[0]['public_delta'],observed[0]['minimum_effect'])
        self.assertEqual(result['state']['incumbent'],'attempt-1')
        store=Store(self.campaign/'memory')
        try:
            records=[n for n in store.snapshot()['nodes'] if n['type']=='Experiment' and n['id'].startswith('attempt-1-seed-')]
            self.assertEqual(len(records),2)
            self.assertTrue(all(n['attrs']['effect_gate_passed'] for n in records))
        finally:store.close()

    def test_failed_implementation_and_invalid_knowledge_recover_automatically(self):
        class Faults(ScriptedAgent):
            failed=False;bad_knowledge=False
            def run(self,role,context,directory):
                if role=='experiment' and not self.failed:
                    self.failed=True;raise ContractError('simulated generated implementation failure')
                response=super().run(role,context,directory)
                if role=='knowledge' and not self.bad_knowledge:
                    self.bad_knowledge=True;response['patch_json']='{"base_version":-1,"nodes":[],"edges":[]}'
                return response
        self.config['budget']=40
        result=self.start(Faults())
        self.assertEqual(result['state']['round'],3)
        self.assertEqual(result['state']['stop_reason'],'max_rounds')
        store=Store(self.campaign/'memory')
        try:
            events=store.events()
            self.assertEqual(sum(e['kind']=='round_failed' for e in events),1)
            self.assertEqual(sum(e['kind']=='knowledge_rejected' for e in events),1)
            self.assertEqual(sum(e['kind']=='evaluation_completed' for e in events),6)
        finally:store.close()
        self.assertEqual(autoresearch(self.campaign,agent=Faults())['budget'],result['budget'])

    def test_transport_failure_stops_roles_until_explicit_repair(self):
        for failed_role in ['evidence','knowledge','missing_cli']:
            with self.subTest(role=failed_role):
                self.campaign=self.root/failed_role;calls=[]
                class Transport(ScriptedAgent):
                    failed=False
                    def run(self,role,context,directory):
                        calls.append(role)
                        if (role==failed_role or failed_role=='missing_cli' and role=='evidence') and not self.failed:
                            self.failed=True
                            if failed_role=='missing_cli':raise FileNotFoundError('codex executable missing')
                            write_json(Path(directory)/'resources.json',{'status':'crash'})
                            raise ContractError('service unavailable')
                        return super().run(role,context,directory)
                self.config['budget']=100;agent=Transport()
                result=self.start(agent)
                self.assertEqual(result['state']['stop_reason'],'role_transport_failure')
                self.assertEqual(result['state']['round'],1)
                observed_role='evidence' if failed_role=='missing_cli' else failed_role
                self.assertEqual(calls.count(observed_role),1)
                if failed_role=='knowledge':self.assertEqual(result['state']['incumbent'],'attempt-1')
                count=len(calls)
                self.assertEqual(autoresearch(self.campaign,agent=agent),result)
                self.assertEqual(len(calls),count)
                research=Research(self.campaign,agent=agent)
                try:
                    cost=research.store.budget()['charged_or_reserved']
                    if failed_role=='evidence':
                        import io,json
                        from contextlib import redirect_stdout
                        from rec_harness.cli import main
                        output=io.StringIO()
                        with redirect_stdout(output):main(['retry-transport',str(self.campaign),'--reason','service restored'])
                        repair=json.loads(output.getvalue())
                    else:repair=research.reopen_transport('service restored')
                    self.assertEqual(repair['budget']['charged_or_reserved'],cost)
                    # Simulate a lost state projection after the durable reopen event.
                    write_json(self.campaign/'state.json',result['state'])
                finally:research.store.close()
                resumed=autoresearch(self.campaign,agent=agent)
                self.assertEqual(resumed['state']['round'],3)
                self.assertEqual(resumed['state']['stop_reason'],'max_rounds')
                self.assertEqual(calls.count(observed_role),3)
                self.assertGreater(resumed['budget']['charged_or_reserved'],cost)
                self.assertEqual(autoresearch(self.campaign,agent=agent),resumed)

    def test_transport_stop_recovers_when_round_publication_is_interrupted(self):
        calls=[]
        class Transport(ScriptedAgent):
            def run(self,role,context,directory):
                calls.append(role)
                write_json(Path(directory)/'resources.json',{'status':'crash'})
                raise ContractError('service unavailable')
        agent=Transport()
        with patch.object(Research,'_record_attempt',side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):self.start(agent)
        result=autoresearch(self.campaign,agent=agent)
        self.assertEqual(result['state']['stop_reason'],'role_transport_failure')
        self.assertEqual(calls,['evidence'])
        self.assertEqual(autoresearch(self.campaign,agent=agent),result)

    def test_insufficient_baseline_budget_stops_before_execution(self):
        self.config['budget']=1
        result=self.start(ScriptedAgent())
        self.assertIsNone(result['state']['baseline']);self.assertEqual(result['budget']['charged_or_reserved'],0)
        self.assertEqual(result['state']['stop_reason'],'budget')

    def test_unknown_isolation_backend_cannot_silently_execute_directly(self):
        self.config['execution']={'kind':'misspelled-isolated-backend'}
        with self.assertRaisesRegex(ContractError,'implicit fallback'):self.start(ScriptedAgent())
        self.assertFalse(self.campaign.exists())

    def measurement_requirement(self):
        self.config['measurement_requirements']=[{'measurement':'interaction_strength',
            'blocked_decision':'add interaction term or increase data',
            'competing_explanations':['missing interaction','insufficient samples'],
            'cheapest_test':'controlled synthetic interaction probe','development_budget':4,
            'acceptance_contract':'independent reference plus held-out decision utility'}]

    def test_missing_measurement_defers_model_change_and_resume_does_not_repeat_roles(self):
        self.measurement_requirement();roles=[]
        class Observe(ScriptedAgent):
            def run(self,role,context,directory):
                roles.append(role);return super().run(role,context,directory)
        result=self.start(Observe())
        self.assertEqual(roles,['evidence']);self.assertEqual(result['state']['stop_reason'],'measurement_required')
        self.assertEqual(result['state']['incumbent'],'root')
        store=Store(self.campaign/'memory')
        try:
            graph=store.snapshot();self.assertEqual(sum(n['type']=='MeasurementGap' for n in graph['nodes']),1)
            self.assertIn('BLOCKS',{e['relation'] for e in graph['edges']})
            self.assertEqual(sum(e['kind']=='evaluation_completed' for e in store.events()),2)
        finally:store.close()
        self.assertEqual(autoresearch(self.campaign,agent=Observe()),result);self.assertEqual(roles,['evidence'])
        from rec_harness.measurement_gaps import evolution_context
        gap_id=next(n['id'] for n in graph['nodes'] if n['type']=='MeasurementGap')
        context=evolution_context(self.campaign,gap_id)
        self.assertEqual(len(context['observations']),2)
        self.assertEqual(context['measurement_gap']['required_information'],'interaction_strength')
        self.assertNotIn('artifact_refs',str(context['observations']))
        store=Store(self.campaign/'memory')
        try:store.event('future_private_outcome',{'value':'not-visible'})
        finally:store.close()
        self.assertEqual(evolution_context(self.campaign,gap_id),context)
        projection=self.campaign/'memory/events.jsonl'
        projection.write_text('retained incomplete export\n')
        with patch.object(Store,'export_events',side_effect=AssertionError('source projection must not be rewritten')):
            self.assertEqual(evolution_context(self.campaign,gap_id),context)
            self.assertEqual(evolution_context(self.campaign,gap_id),context)
        self.assertEqual(projection.read_text(),'retained incomplete export\n')
        original_verify=Store.verify;inserted=[]
        def concurrent_append(source):
            value=original_verify(source)
            if source.read_only and not inserted:
                inserted.append(True)
                writer=Store(self.campaign/'memory')
                try:writer.event('future_measurement_outcome',{'value':'not-visible'})
                finally:writer.close()
            return value
        with patch.object(Store,'verify',concurrent_append):
            self.assertEqual(evolution_context(self.campaign,gap_id),context)
        self.assertEqual(inserted,[True])



    def test_failed_tool_does_not_create_a_measurement_gap_or_call_model_roles(self):
        self.measurement_requirement()
        with patch('rec_harness.diagnostics.measure',return_value={'status':'invalid','error':'bad invocation'}):
            result=self.start(ScriptedAgent())
        self.assertEqual(result['state']['stop_reason'],'measurement_required')
        store=Store(self.campaign/'memory')
        try:
            self.assertFalse(any(n['type']=='MeasurementGap' for n in store.snapshot()['nodes']))
            self.assertFalse(any(e['kind']=='agent_submitted' for e in store.events()))
        finally:store.close()

    def test_available_required_measurement_allows_model_research(self):
        self.measurement_requirement();self.config['max_rounds']=1
        with patch('rec_harness.diagnostics.measure',return_value={'status':'success','output':{'measurements':{'interaction_strength':1}}}):
            result=self.start(ScriptedAgent())
        self.assertEqual(result['state']['incumbent'],'attempt-1')


class HostScoringTests(unittest.TestCase):
    def test_worker_metrics_cannot_replace_host_computation(self):
        targets=[y for _,y in samples(99,3)]
        response={'weights':[0.],'predictions':[0.,0.,0.]}
        scored=score_response(response,targets)
        self.assertLess(scored['quality'],1)
        with self.assertRaisesRegex(ContractError,'never metrics'):
            score_response({**response,'quality':1},targets)
        for wrong in ([0.],[float('nan')]*3,[1e308]*3,[10**1000]*3):
            with self.assertRaises(ContractError):score_response({**response,'predictions':wrong},targets)
