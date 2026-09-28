import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from rec_harness.core import Artifacts,ContractError,digest,write_json
from rec_harness.store import Store
from rec_harness.decision_traces import collect_traces
from rec_harness.adapters.amazon_trace_feedback import commit


class AmazonTraceFeedbackTests(unittest.TestCase):
    def test_development_feedback_respects_cutoff_and_excludes_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);store=Store(root/'memory');self.addCleanup(store.close)
            config={'task_id':'fixture','protocol_hash':'p','trace_development_feedback':True}
            write_json(root/'amazon-auto.json',config);write_json(root/'auto-lock.json',{'config_hash':digest(config)})
            objects=Artifacts(root/'artifacts/objects');write_json(root/'visible.json',{'public':'before training'})
            ref=objects.put(root/'visible.json')
            store.event('agent_submitted',{'task_id':'design','role':'hypothesis','episode_id':'round-1',
                'input_hash':digest({'public':'before training'}),'graph_version':0,'visible_state_ref':ref})
            store.event('role_completed',{'task_id':'design','status':'success','wall_seconds':1,'archive':{}})
            cutoff=store.verify()['events']
            before=collect_traces(root,cutoff_seq=cutoff)
            self.assertEqual(before['decision_traces'][0]['host_feedback']['availability'],'not_committed_in_prefix')
            report={'schema':'amazon-development-selection-v1','proposal_id':'p1','plan_handoff_hash':'h',
                'protocol_hash':'p','source_hashes':{},'summary':{'eligible_for_confirmation':True},
                'source_receipts':{},'rows':[{'model_version':'candidate','dataset':'A','seed':42,'metrics':{'HR@10':0.5}}],
                'selection_policy_provenance':'development only'}
            report['hash']=digest(report);folder=root/'rounds/round-1';write_json(folder/'development-readout.json',report)
            controller=SimpleNamespace(root=root,store=store);study={'protocol_hash':'p','development_seeds':[42]}
            commit(controller,folder,report,study);committed=store.verify()
            commit(controller,folder,report,study);self.assertEqual(store.verify(),committed)
            store.event('auto_round_completed',{'result':{'selected':True,'private_result':'PRIVATE_CONFIRMATION'},'state':{'round':1}})
            after=collect_traces(root);trace=after['decision_traces'][0]
            self.assertEqual(trace['host_feedback']['readout'],report)
            self.assertEqual(trace['host_feedback']['partition'],'development')
            self.assertNotIn('PRIVATE_CONFIRMATION',str(after))
            self.assertEqual(trace['visible_state'],before['decision_traces'][0]['visible_state'])
            self.assertEqual(collect_traces(root,cutoff_seq=cutoff),before)
            changed={**report,'rows':[{'seed':83}]};changed['hash']=digest({k:v for k,v in changed.items() if k!='hash'})
            write_json(folder/'development-readout.json',changed)
            with self.assertRaisesRegex(ContractError,'non-development seed'):commit(controller,folder,changed,study)
            changed={**report,'extra':'PRIVATE_CONFIRMATION'};changed['hash']=digest({k:v for k,v in changed.items() if k!='hash'})
            with self.assertRaisesRegex(ContractError,'fields differ'):commit(controller,folder,changed,study)
