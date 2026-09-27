import copy
from pathlib import Path
import tempfile
import unittest

from rec_harness.core import Artifacts,ContractError,digest,write_json
from rec_harness.adapters.amazon_replay import build_cycle_world
from rec_harness.replay import Replay


class AmazonReplayTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name)
        artifacts=Artifacts(root/'objects');write_json(root/'context.json',{'visible':'before'})
        (root/'base.py').write_text('baseline');(root/'candidate.py').write_text('candidate')
        visible={'protocol_hash':'protocol','bindings':{'base_model_id':'base','candidate_model_version':'candidate','proposal_id':'proposal'},
                 'development_baseline':[{'dataset':'d','seed':42,'metrics':{'HR@10':.5}}]}
        report={'protocol_hash':'protocol','rows':[{'model_version':'base','dataset':'d','seed':42,'metrics':{'HR@10':.5}}],
                'summary':{'metric':'HR@10','means':{'base':.5,'candidate':.501},'eligible_for_confirmation':False}}
        report['hash']=digest(report)
        self.args=dict(artifacts=artifacts,protocol='protocol',proposal='proposal',baseline='base',candidate='candidate',
            visible=visible,report=report,selection='discard',gpu_total=100,candidate_gpu=20,role_calls=11,
            gpu_weight=1,role_weight=2,request_ref=artifacts.put(root/'context.json'),
            workspace_before={'model.py':artifacts.put(root/'base.py')},workspace_after={'model.py':artifacts.put(root/'candidate.py')})

    def test_observed_cycle_replay_charges_once_and_respects_discard(self):
        world=build_cycle_world(**self.args)
        self.assertEqual(world['qualification'],'replay_ready')
        self.assertEqual(world['initial_cost'],80)
        self.assertEqual(world['nodes'][0]['cost'],42)
        replay=Replay(world,122)
        self.assertEqual(replay.view()['observations'],{'root':{'score':.5,'status':'success'}})
        self.assertEqual(replay.step(['root'],action_kind='new_code')['status'],'unsupported')
        self.assertEqual(replay.spent,80)
        result=replay.step(['root'])
        observation=result['outcomes'][0]['observation']
        self.assertEqual(observation['score'],.5)
        self.assertEqual(observation['candidate_score'],.501)
        self.assertEqual(replay.spent,122)
        replay.step(['root']);self.assertEqual(replay.spent,122)
        self.assertFalse(world['online_utility_evidence'])
        limited=Replay(world,121)
        self.assertEqual(limited.step(['root'])['status'],'budget_exhausted')
        self.assertEqual(limited.view()['revealed_node_ids'],['root'])

    def test_unseen_baseline_or_changed_report_rejected(self):
        altered=copy.deepcopy(self.args['visible']);altered['development_baseline'][0]['metrics']['HR@10']=.4
        with self.assertRaisesRegex(ContractError,'not visible'):
            build_cycle_world(**{**self.args,'visible':altered})
        report=copy.deepcopy(self.args['report']);report['summary']['means']['candidate']=.9
        with self.assertRaisesRegex(ContractError,'readout changed'):
            build_cycle_world(**{**self.args,'report':report})
        with self.assertRaisesRegex(ContractError,'frozen gate'):
            build_cycle_world(**{**self.args,'selection':'keep'})

    def test_cost_and_artifact_integrity_are_required(self):
        with self.assertRaisesRegex(ContractError,'exceeds'):
            build_cycle_world(**{**self.args,'candidate_gpu':101})
        with self.assertRaisesRegex(ContractError,'positive replay cost'):
            build_cycle_world(**{**self.args,'gpu_weight':0,'role_weight':0})
        ref=self.args['workspace_after']['model.py']
        self.args['artifacts'].verify(ref).write_text('changed')
        with self.assertRaises(ContractError):build_cycle_world(**self.args)
