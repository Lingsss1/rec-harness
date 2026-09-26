import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from rec_harness.core import Artifacts,ContractError,canonical,digest,file_hash,read_json,write_json
from rec_harness.store import Store
from rec_harness.adapters.independent_scoring import prepare_request,score_response
from rec_harness.adapters.amazon_interaction_context import freeze_interactions
from rec_harness.adapters.amazon_block_context import block_context,cost_summary


class InteractionContextTests(unittest.TestCase):
    def test_freeze_projection_cost_reuse_and_confirmation_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/'study';objects=Artifacts(study/'artifacts/objects');directories=[]
            worker=Path(__file__).resolve().parents[1]/'rec_harness/adapters/score_worker.py'
            cfg={'protocol_hash':'p','datasets':['tiny'],'development_seeds':[42],'data_hashes':{},
                 'model_versions':{'model':{'sha256':'source'}}}
            baseline={'source_hash':'source','checkpoint_sha256':'weights','candidates_hash':'candidates'}
            for index,blocks in enumerate([[],[0],[1],[0,1]]):
                directory=root/str(index);directories.append(str(directory))
                request,key=prepare_request([{'user_id':90210,'history':[101],'candidates':list(range(1,101))}],seed=index)
                values=[0.0]*100;values[key['rows'][0]['positive_index']]=1.0
                response={'schema':'candidate-scores-response-v1','request_hash':digest(request),'rows':[{'row_id':'row-0','scores':values}]}
                ranked=score_response(request,key,response)
                if not blocks:write_json(study/'jobs/model-tiny-42/validation_ranks.json',{'user_ids':[90210],'ranks':[0]})
                for name,data in [('inference-request.json',request),('private-key.json',key),('inference/results/scores.json',response),('host-ranking.json',ranked)]:
                    write_json(directory/name,data)
                summary={'dataset':'tiny','model_version':'model','seed':42,'users':1,'protocol_hash':'p',
                         'source_hash':'source','checkpoint_hash':'weights','candidates_hash':'candidates',
                         'score_worker_hash':file_hash(worker),'host_scorer_hash':'fixture-scorer','rank_disagreements':0,
                         'host_report_hash':ranked['hash'],'host_metrics':ranked['metrics']}
                if blocks:summary['bypass_blocks']=blocks
                summary['hash']=digest(summary);write_json(directory/'acceptance.json',summary)
                write_json(directory/'inference/receipt.json',{'process':{'status':'success'},'wall_seconds':1,
                    'collection':{'artifacts':{'scores.json':objects.put(directory/'inference/results/scores.json')}}})
                write_json(directory/'inference/request.json',{'inputs':{name:{'sha256':value} for name,value in [
                    ('model.py','source'),('checkpoint.pt','weights'),('worker.py',file_hash(worker)),('request.json',file_hash(directory/'inference-request.json'))]},
                    'arguments':sum((['--bypass-block',str(b)] for b in blocks),[])})
            group={'study':str(study),'directories':directories};artifacts=Artifacts(root/'campaign/artifacts/objects')
            with patch('rec_harness.adapters.amazon_interaction_context.load_config',return_value=cfg),patch('rec_harness.adapters.amazon_interaction_context.audit_job',return_value=(baseline,{})) as audit:
                records=freeze_interactions([group],cfg,artifacts)
                public=read_json(artifacts.verify(records[0]['public_ref']))
                self.assertNotIn('90210',canonical(public));self.assertNotIn('weights',canonical(public))
                self.assertEqual(public['measurements']['tiny.blocks-0-1.HR@10.interaction'],0)
                store=Store(root/'campaign/memory')
                try:
                    controller=SimpleNamespace(config={'block_interaction_diagnostics':records},store=store,artifacts=artifacts,task='task',protocol_hash='p')
                    context=block_context(controller,'model',cfg);before=store.verify()
                    self.assertEqual(len([n for n in store.snapshot()['nodes'] if n['type']=='ToolVersion']),1)
                    self.assertEqual([e['relation'] for e in store.snapshot()['edges']],['MEASURED_BY'])
                    self.assertEqual(block_context(controller,'model',cfg),context);self.assertEqual(store.verify(),before)
                    self.assertEqual(cost_summary({'block_interaction_diagnostics':records*2})['retained_interaction_cpu_seconds'],4)
                    receipt_ref=records[0]['source_archives'][0]['inference/receipt.json']
                    receipt_path=artifacts.verify(receipt_ref);original_bytes=receipt_path.read_bytes();receipt_path.write_text('{}')
                    with self.assertRaises(ContractError):block_context(controller,'model',cfg)
                    receipt_path.write_bytes(original_bytes)
                    self.assertEqual(block_context(controller,'model',cfg),context)
                finally:store.close()
                path=Path(directories[-1])/'inference/request.json';request=read_json(path);request['arguments']=['--bypass-block','0'];write_json(path,request)
                with self.assertRaisesRegex(ContractError,'execution arguments'):freeze_interactions([group],cfg,artifacts)
                metadata=read_json(Path(directories[0])/'acceptance.json');metadata['seed']=17;write_json(Path(directories[0])/'acceptance.json',metadata)
                audit.reset_mock()
                with self.assertRaisesRegex(ContractError,'development'):freeze_interactions([group],cfg,artifacts)
                audit.assert_not_called()
