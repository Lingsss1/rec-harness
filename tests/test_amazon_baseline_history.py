from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from rec_harness.core import Artifacts,ContractError,file_hash,write_json
from rec_harness.store import Store
from rec_harness.adapters.amazon_baseline_history import baseline_history,verify_visible_history
from rec_harness.adapters.amazon_history_graph import attach_history


class BaselineHistoryTests(unittest.TestCase):
    def test_pre_design_aggregates_are_private_safe_and_resume_with_bound_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/'study';study.mkdir()
            data=study/'tiny_train.txt';data.write_text('987654 22\n987654 23\n')
            cfg={'data':str(study),'data_hashes':{data.name:file_hash(data)},'datasets':['tiny'],
                 'development_seeds':[42],'protocol_hash':'p'}
            ranks=study/'jobs/base-tiny-42/validation_ranks.json'
            write_json(ranks,{'user_ids':[987654],'ranks':[2]})
            store=Store(root/'memory');self.addCleanup(store.close)
            controller=SimpleNamespace(study=study,store=store,artifacts=Artifacts(root/'objects'))
            controller.config={'protocol_hash':'p'}
            with patch('rec_harness.adapters.amazon_baseline_history.load_config',return_value=cfg),patch('rec_harness.adapters.amazon_baseline_history.audit_job',return_value=({'seed':42},{})) as audit:
                value=baseline_history(controller,root/'round-1','base')
                self.assertEqual(value['rows'][0]['slices'][0]['users'],1)
                self.assertEqual(value['rows'][0]['slices'][0]['metrics']['HR@10'],1)
                for forbidden in ['987654','user_ids','input_hashes','checkpoint','candidate',str(root)]:
                    self.assertNotIn(forbidden,str(value))
                before=store.verify()
                verify_visible_history(controller,root/'round-1',value)
                self.assertEqual(store.verify(),before)
                with self.assertRaisesRegex(ContractError,'differs from committed'):
                    verify_visible_history(controller,root/'round-1',{**value,'rows':[]})
                with self.assertRaisesRegex(ContractError,'unique completion'):
                    verify_visible_history(controller,root/'round-2',value)
                projected=root/'round-1/baseline-history.json';original=projected.read_bytes()
                projected.write_bytes(original+b' ')
                with self.assertRaisesRegex(ContractError,'projection changed'):
                    verify_visible_history(controller,root/'round-1',value)
                projected.write_bytes(original)
                graph={'task_id':'research','nodes':[],'edges':[],'graph_version':0}
                first=attach_history(controller,root/'round-1',graph,value)
                self.assertEqual({n['type'] for n in first['nodes']},{'ToolVersion','Observation'})
                self.assertEqual(first['edges'][0]['relation'],'MEASURED_BY')
                self.assertEqual(attach_history(controller,root/'round-1',first,value),first)
                observation=next(n for n in first['nodes'] if n['type']=='Observation')
                self.assertFalse(observation['attrs']['independent_unit'])
                self.assertNotIn('987654',str(first));self.assertNotIn(str(root),str(first))
                with patch('rec_harness.adapters.history_diagnostic.summarize',side_effect=AssertionError('must reuse aggregate')):
                    self.assertEqual(baseline_history(controller,root/'round-1','base'),value)
                self.assertEqual(store.verify(),before)
                self.assertEqual(audit.call_count,2)
                write_json(ranks,{'user_ids':[987654],'ranks':[12]})
                with self.assertRaisesRegex(ContractError,'binding changed'):
                    baseline_history(controller,root/'round-1','base')
