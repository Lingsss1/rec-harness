import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from rec_harness.core import Artifacts,ContractError,digest,write_json
from rec_harness.store import Store
from rec_harness.decision_traces import collect_traces
from rec_harness.adapters.amazon_trace_feedback import commit


class AmazonTraceFeedbackTests(unittest.TestCase):
    def test_rejections_are_visible_but_never_admitted_measurement_evidence(self):
        from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
        from rec_harness.optimizer_handoff import request
        for status in ['crash','success']:
            with self.subTest(status=status),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);store=Store(root/'memory');store.configure_budget(5)
                try:
                    config={'task_id':'fixture','protocol_hash':'p'}
                    write_json(root/'amazon-auto.json',config);write_json(root/'auto-lock.json',{'config_hash':digest(config)})
                    runner=object.__new__(AmazonAutoresearch);runner.store=store
                    class BadAgent:
                        def run(self,role,context,target):
                            write_json(target/'request.json',{'role':role})
                            write_json(target/'resources.json',{'status':status,'wall_seconds':2})
                            (target/'response.json').write_text('invalid JSON')
                            raise ContractError('fixture failure')
                    runner.agent=BadAgent();context={'baseline':{'runs':[{'seed':42,'valid':True,'metrics':{'quality':.5}}]}}
                    with self.assertRaises(ContractError):runner._role(root/'rounds/round-1','lead',context)
                    package=collect_traces(root);traces=package['decision_traces']
                    self.assertEqual(len(traces),1 if status=='crash' else 2)
                    self.assertEqual(request(package,10)['allowed_evidence_refs'],[])
                    for trace in traces:
                        self.assertEqual(trace['role_admission'],'rejected');self.assertIsNone(trace['role_output'])
                        self.assertEqual(trace['transport_status'],status)
                        self.assertEqual(trace['resource_cost']['wall_seconds'],2)
                        self.assertIn('baseline',trace['visible_state']);self.assertNotIn('rejection',trace['visible_state'])
                        self.assertIsNotNone(trace['rejection']['raw_response_ref'])
                    events=store.events();first=next(e for e in events if e['kind']=='role_rejected')
                    prefix=collect_traces(root,cutoff_seq=first['seq'])
                    self.assertEqual(len(prefix['decision_traces']),1)
                    with self.assertRaisesRegex(ContractError,'no committed decision traces'):
                        collect_traces(root,cutoff_seq=first['seq']-1)
                    store.event('future_private_result',{'private_value':'FUTURE_NOT_VISIBLE'})
                    self.assertEqual(collect_traces(root,cutoff_seq=first['seq']),prefix)
                    self.assertNotIn('FUTURE_NOT_VISIBLE',str(collect_traces(root)))
                finally:store.close()

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
