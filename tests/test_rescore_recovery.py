import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import ContractError,digest,file_hash,read_json,write_json
from rec_harness.adapters.amazon_rescore import rescore_job,prepared_request


class RescoreRecoveryTests(unittest.TestCase):
    def test_completed_inference_recovers_without_relaunch_and_rejects_changed_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/'study';study.mkdir();output=root/'scores'
            source=study/'model.py';source.write_text('fixture')
            training=study/'tiny_train.txt';training.write_text('1 101\n')
            job=study/'jobs/model-tiny-42';job.mkdir(parents=True)
            (job/'checkpoint.pt').write_bytes(b'fixture weights')
            write_json(job/'validation_candidates.json',[{'user_id':1,'candidates':list(range(1,101))}])
            write_json(job/'validation_ranks.json',{'user_ids':[1],'ranks':[0]})
            cfg={'datasets':['tiny'],'development_seeds':[42],'confirmation_seeds':[17],
                 'protocol_hash':'protocol','python':'fixture','data':str(study),
                 'data_hashes':{training.name:file_hash(training)},
                 'model_versions':{'model':{'path':str(source),'sha256':file_hash(source)}}}
            result={'source_hash':file_hash(source),'checkpoint_sha256':file_hash(job/'checkpoint.pt'),
                    'candidates_hash':file_hash(job/'validation_candidates.json'),'metrics':{'HR@10':1.0}}
            calls=[]
            def inference(directory,recipe,files,entry,arguments,**kwargs):
                calls.append(directory);directory.mkdir()
                write_json(directory/'request.json',{'inputs':files,'entry':entry,'arguments':arguments,'recipe':recipe})
                write_json(directory/'process/process.json',{'status':'success','wall_seconds':0.1})
                request=read_json(output/'inference-request.json');private=read_json(output/'private-key.json')
                scores=[0.0]*100;scores[private['rows'][0]['positive_index']]=1.0
                write_json(directory/'jail/output/scores.json',{'schema':'candidate-scores-response-v1','request_hash':digest(request),
                    'rows':[{'row_id':'row-0','scores':scores}]})
                raise OSError('interrupted before collection')
            with patch('rec_harness.adapters.amazon_rescore.load_config',return_value=cfg),patch('rec_harness.adapters.amazon_rescore.audit_job',return_value=(result,{})),patch('rec_harness.adapters.amazon_rescore.run_cpu_job',side_effect=inference):
                with self.assertRaisesRegex(OSError,'interrupted'):
                    rescore_job(study,'model','tiny',42,output,{},root/'uids',bypass_blocks=[0,1])
                accepted=rescore_job(study,'model','tiny',42,output,{},root/'uids',bypass_blocks=[0,1],resume=True)
                self.assertEqual(accepted['host_metrics']['HR@10'],1)
                self.assertIsNone(accepted['inference_wall_seconds'])
                self.assertEqual(accepted['inference_process_wall_seconds'],0.1)
                self.assertEqual(rescore_job(study,'model','tiny',42,output,{},root/'uids',bypass_blocks=[1,0],resume=True),accepted)
                self.assertEqual(len(calls),1)
                with self.assertRaisesRegex(ContractError,'plan changed'):
                    rescore_job(study,'model','tiny',42,output,{},root/'uids',bypass_blocks=[0,2],resume=True)
                write_json(output/'host-ranking.json',{'edited':True})
                with self.assertRaisesRegex(ContractError,'host rankings changed'):
                    rescore_job(study,'model','tiny',42,output,{},root/'uids',bypass_blocks=[0,1],resume=True)
                self.assertEqual(len(calls),1)

    def test_private_key_write_interruption_preserves_request_permutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[{'user_id':1,'history':[101],'candidates':list(range(1,101))}]
            request,key=prepared_request(root,rows);(root/'private-key.json').unlink()
            with patch('rec_harness.adapters.amazon_rescore.prepare_request',side_effect=AssertionError('new permutation')):
                self.assertEqual(prepared_request(root,rows),(request,key))
                with self.assertRaisesRegex(ContractError,'history or candidates'):
                    prepared_request(root,[{**rows[0],'history':[102]}])
