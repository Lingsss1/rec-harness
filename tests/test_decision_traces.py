import tempfile
import unittest
from pathlib import Path
from rec_harness.core import ContractError,read_json
from rec_harness.decision_traces import collect_traces
from rec_harness.runtime import Research
from test_runtime import ROOT,ScriptedAgent


class DecisionTraceTests(unittest.TestCase):
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
                path=research.artifacts.verify(trace['visible_state_ref']);path.write_text('{}')
                with self.assertRaises(ContractError):collect_traces(root,roles=['hypothesis'],cutoff_seq=cutoff)
            finally:research.store.close()
