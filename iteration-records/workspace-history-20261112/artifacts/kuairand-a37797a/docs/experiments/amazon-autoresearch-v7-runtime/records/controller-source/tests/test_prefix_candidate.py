import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rec_harness.adapters.amazon_prefix_candidate import OLD, EDITS, admit_plan, expected_sources, plan_guard, plan_remaining
from rec_harness.core import Artifacts, ContractError, file_hash, digest, write_json
from rec_harness.store import Store


class PrefixAdmissionTests(unittest.TestCase):
    def test_extra_generated_code_rejected_before_probe(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'incumbent.py'; source.write_text('# text fixture\n'+OLD+'\n')
            config={'model_versions':{'legacy_incumbent':{'path':str(source),'sha256':file_hash(source)}},'protocol_hash':'p'}
            write_json(root/'study.json',config); write_json(root/'study-lock.json',{'config_hash':digest(config)})
            expected=expected_sources(config); artifacts=Artifacts(root/'artifacts/objects'); versions=[]
            for name,text in expected.items():
                path=root/(name+'.py'); path.write_text(text+'unexpected_execution = True\n')
                versions.append({'model_version':name,'source':artifacts.put(path)})
            store=Store(root/'memory')
            store.event('amazon_research_plan_registered',{'proposal_id':'plan','baseline':'legacy_incumbent',
                'versions':{'candidate':versions[0],'controls':[versions[1]]}}); store.close()
            with patch('rec_harness.adapters.amazon_prefix_candidate.run_gate', side_effect=AssertionError('must not execute')):
                with self.assertRaisesRegex(ContractError,'exceeds reviewed'):
                    admit_plan(root,'plan')

    def test_confirmation_gate_and_plan_cost_include_failed_reservations(self):
        with tempfile.TemporaryDirectory() as d:
            store=Store(Path(d)/'memory'); self.addCleanup(store.close); store.configure_budget(1000)
            names=list(EDITS)
            plan={'proposal_id':'plan','handoff_hash':'hash','requested_budget_gpu_seconds':600,
                  'versions':{'candidate':{'model_version':names[0]},'controls':[{'model_version':names[1]}]}}
            store.event('amazon_research_plan_registered',plan)
            store.event('amazon_prefix_candidate_admitted',{'proposal_id':'plan','plan_handoff_hash':'hash','versions':dict.fromkeys(names)})
            self.assertEqual(plan_guard(store,names,[42],[17,83]),plan)
            with self.assertRaisesRegex(ContractError,'frozen selection'):
                plan_guard(store,names,[17],[17,83])
            store.reserve(names[0]+'-tiny-42',100); store.settle(names[0]+'-tiny-42',None,'failed')
            store.reserve(names[1]+'-tiny-42',100); store.settle(names[1]+'-tiny-42',20,'success')
            self.assertEqual(plan_remaining(store,plan),480)

    def test_reused_control_history_is_not_charged_to_new_proposal_twice(self):
        with tempfile.TemporaryDirectory() as d:
            store=Store(Path(d)/'memory');self.addCleanup(store.close);store.configure_budget(1000)
            store.reserve('baseline-tiny-42',200);store.settle('baseline-tiny-42',200,'success')
            plan={'requested_budget_gpu_seconds':500,'versions':{'candidate':{'model_version':'new'},
                  'controls':[{'model_version':'baseline'}]},'reused_control_versions':['baseline']}
            self.assertEqual(plan_remaining(store,plan),500)
            self.assertEqual(store.budget()['remaining'],800)

    def test_failed_feasibility_prevents_development_launch(self):
        with tempfile.TemporaryDirectory() as d:
            store=Store(Path(d)/'memory');self.addCleanup(store.close)
            name='nextitnet_cosine_context_v1'
            plan={'proposal_id':name,'handoff_hash':'h'}
            store.event('amazon_research_plan_registered',plan)
            store.event('amazon_prefix_candidate_admitted',{'proposal_id':name,
                'plan_handoff_hash':'h','versions':{name:{}}})
            store.event('amazon_research_plan_blocked',{'plan_handoff_hash':'h','reason':'budget infeasible'})
            with self.assertRaisesRegex(ContractError,'research plan blocked'):
                plan_guard(store,[name],[42],[17,83])
