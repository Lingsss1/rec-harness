import tempfile
import unittest
from unittest.mock import patch
from rec_harness.store import Store
from pathlib import Path
from rec_harness.core import ContractError,read_json
from rec_harness.decision_traces import collect_traces
from rec_harness.runtime import Research
from test_runtime import ROOT,ScriptedAgent


class DecisionTraceTests(unittest.TestCase):
    def test_archived_instruction_identity_reaches_actual_trace(self):
        from rec_harness.agents import schema_for
        from rec_harness.core import write_json
        from rec_harness.instruction_identity import manifest,skill_suffix
        from rec_harness.agents import search_receipt
        class RecordedAgent(ScriptedAgent):
            def run(self,role,context,directory):
                result=super().run(role,context,directory)
                skills=[{'name':'hypothesis-check','content':'Check alternatives.'}] if role=='hypothesis' else []
                prompt='fixture '+role+skill_suffix(skills);schema=schema_for(role)
                request=read_json(Path(directory)/'request.json')
                write_json(Path(directory)/'request.json',{**request,'prompt':prompt,'instruction_identity':manifest(prompt,schema,skills),**({'role_skills':skills} if skills else {})})
                write_json(Path(directory)/'schema.json',schema)
                (Path(directory)/'stdout.jsonl').write_text('{"type":"item.completed","item":{"type":"web_search","query":"fixture"}}\n' if role=='hypothesis' else '')
                search_receipt(directory,'live' if role=='hypothesis' else 'disabled',result)
                return result
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'campaign';config=read_json(ROOT/'examples/toy/campaign.json')
            config.update(source=str(ROOT/'examples/toy'),max_rounds=1)
            Research.init(root,config);research=Research(root,agent=RecordedAgent())
            try:
                research.run();traces=collect_traces(root)['decision_traces']
                self.assertEqual(len(traces),6)
                for trace in traces:
                    value=trace['instruction_identity']
                    skills=[{'name':'hypothesis-check','content':'Check alternatives.'}] if trace['role']=='hypothesis' else []
                    self.assertEqual(value['prompt_hash'],manifest('fixture '+trace['role']+skill_suffix(skills),schema_for(trace['role']),skills)['prompt_hash'])
                    if skills:self.assertEqual(trace['skill_versions']['explicit'],value['explicit_skills'])
                    else:self.assertEqual(trace['skill_versions'],{'availability':'not recorded by transport'})
                    research.artifacts.verify(value['request_ref']);research.artifacts.verify(value['schema_ref'])
                    search=trace['literature_search']
                    self.assertEqual(search['search_observed'],trace['role']=='hypothesis')
                    research.artifacts.verify(search['receipt_ref']);research.artifacts.verify(search['stdout_ref'])
                    self.assertNotIn('literature_search',trace['visible_state'])
            finally:research.store.close()

    def test_prefix_references_actual_inputs_and_excludes_later_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'campaign';config=read_json(ROOT/'examples/toy/campaign.json')
            config.update(source=str(ROOT/'examples/toy'),max_rounds=1)
            Research.init(root,config);research=Research(root,agent=ScriptedAgent())
            try:
                research.run();cutoff=research.store.verify()['events']
                graph=research.store.snapshot()
                self.assertEqual(sum(n['type']=='DecisionTrace' for n in graph['nodes']),6)
                self.assertEqual(sum(e['relation']=='GENERATED_BY' for e in graph['edges']),6)
                package=collect_traces(root,roles=['hypothesis'],cutoff_seq=cutoff)
                trace=package['decision_traces'][0]
                self.assertEqual(trace['role'],'hypothesis');self.assertEqual(trace['source_visibility'],'exact snapshot')
                self.assertIn('model.py',trace['visible_source_snapshot'])
                self.assertNotIn('evaluation',trace['visible_state'])
                self.assertEqual(trace['role_output']['proposal']['base_model_id'],trace['model_version'])
                completed=next(e for e in research.store.events() if e['kind']=='round_completed' and e['payload']['attempt']==trace['episode_id'])
                self.assertEqual(trace['host_feedback']['decision'],completed['payload']['decision'])
                self.assertEqual(trace['host_feedback']['event_hash'],completed['hash'])
                earlier=collect_traces(root,roles=['hypothesis'],cutoff_seq=completed['seq']-1)['decision_traces'][0]
                self.assertEqual(earlier['host_feedback'],{'availability':'not_committed_in_prefix'})
                self.assertEqual(earlier['visible_state'],trace['visible_state'])
                self.assertEqual(earlier['input_hash'],trace['input_hash'])
                # A different episode's outcome must not overwrite this one.
                research.store.event('round_completed',{'attempt':'unrelated','decision':{'selection':'invalid'}})
                latest=collect_traces(root,roles=['hypothesis'])['decision_traces'][0]
                self.assertEqual(latest['host_feedback'],trace['host_feedback'])
                research.store.event('future_outcome',{'result':'FUTURE'})
                self.assertEqual(collect_traces(root,roles=['hypothesis'],cutoff_seq=cutoff),package)
                projection=root/'memory/events.jsonl'
                projection.write_text('preserve partial source projection\n')
                with patch.object(Store,'export_events',side_effect=AssertionError('trace extraction wrote source')):
                    self.assertEqual(collect_traces(root,roles=['hypothesis'],cutoff_seq=cutoff),package)
                self.assertEqual(projection.read_text(),'preserve partial source projection\n')
                original_verify=Store.verify;appended=[]
                def concurrent_append(source):
                    result=original_verify(source)
                    if source.read_only and not appended:
                        appended.append(True)
                        research.store.event('future_concurrent_outcome',{'result':'NOT_IN_PREFIX'})
                    return result
                with patch.object(Store,'verify',concurrent_append):
                    self.assertEqual(collect_traces(root,roles=['hypothesis'],cutoff_seq=cutoff),package)
                self.assertEqual(appended,[True])
                path=research.artifacts.verify(trace['visible_state_ref']);path.write_text('{}')
                with self.assertRaises(ContractError):collect_traces(root,roles=['hypothesis'],cutoff_seq=cutoff)
            finally:research.store.close()
