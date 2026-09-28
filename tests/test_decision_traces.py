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
        from rec_harness.instruction_identity import manifest
        class RecordedAgent(ScriptedAgent):
            def run(self,role,context,directory):
                result=super().run(role,context,directory)
                prompt='fixture '+role;schema=schema_for(role)
                request=read_json(Path(directory)/'request.json')
                write_json(Path(directory)/'request.json',{**request,'prompt':prompt,'instruction_identity':manifest(prompt,schema)})
                write_json(Path(directory)/'schema.json',schema)
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
                    self.assertEqual(value['prompt_hash'],manifest('fixture '+trace['role'],schema_for(trace['role']))['prompt_hash'])
                    research.artifacts.verify(value['request_ref']);research.artifacts.verify(value['schema_ref'])
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
