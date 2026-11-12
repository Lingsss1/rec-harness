import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rec_harness.core import ContractError,digest,file_hash,read_json,write_json
from rec_harness.store import Store
from rec_harness.adapters.amazon_final_selection import seal_selection
from rec_harness.adapters.amazon_final import run_final


class FinalRunTests(unittest.TestCase):
    def test_complete_comparison_and_exact_resume_without_inference(self):
        self.check_comparison(False)

    def test_recover_completed_process_without_receipt_or_duplicate_inference(self):
        self.check_comparison(True)

    def check_comparison(self,interrupted):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/'study';study.mkdir();output=root/'final'
            models={}
            for name in ['base','candidate']:
                source=root/(name+'.py');source.write_text(name)
                models[name]={'path':str(source),'sha256':file_hash(source)}
            data_hashes={}
            for dataset in ['a','b']:
                for split,items in [('train',[1,2]),('valid',[3]),('test',[4])]:
                    path=root/f'{dataset}_{split}.txt';path.write_text(''.join(f'1 {i}\n' for i in items))
                    if split!='test':data_hashes[path.name]=file_hash(path)
            cfg={'model_versions':models,'datasets':['a','b'],'confirmation_seeds':[17,83],
                'protocol_hash':'p','data':str(root),'data_hashes':data_hashes,'python':'/usr/bin/python3'}
            write_json(study/'study.json',cfg)
            store=Store(study/'memory');store.configure_budget(100);store.close()
            def audit(study,cfg,model,dataset,seed):
                checkpoint=study/'jobs'/f'{model}-{dataset}-{seed}'/'checkpoint.pt'
                checkpoint.parent.mkdir(parents=True,exist_ok=True);checkpoint.write_bytes(b'weights')
                return {'checkpoint_sha256':file_hash(checkpoint),'execution_id':'same'},{'itemnum':200}
            with patch('rec_harness.adapters.amazon_final_selection.load_config',return_value=cfg),patch('rec_harness.adapters.amazon_final_selection.audit_job',side_effect=audit):
                seal_selection(study,output,['base','candidate'],root,eval_seed=7)
            store=Store(study/'memory');sealed=store.verify();store.close()
            calls=[]
            def infer(directory,recipe,files,entry,arguments,**kwargs):
                calls.append(str(directory));directory.mkdir()
                request=read_json(files['request.json']['path'])
                self.assertEqual(set(files),{'model.py','checkpoint.pt','request.json','worker.py'})
                self.assertNotIn('positive_index',request['rows'][0]);self.assertNotIn('user_id',request['rows'][0])
                # Synthetic model scores the known fixture positive last/first.
                candidate=files['model.py']['sha256']==models['candidate']['sha256']
                response={'schema':'candidate-scores-response-v1','request_hash':digest(request),'rows':[
                    {'row_id':r['row_id'],'scores':[1.0 if ((i==4)==candidate) else 0.0 for i in r['candidates']]} for r in request['rows']]}
                scores=directory/'scores.json';write_json(scores,response)
                receipt={'process':{'status':'success','wall_seconds':0.1},'collection':{'artifacts':{'scores.json':kwargs['artifacts'].put(scores)}},'wall_seconds':0.1}
                write_json(directory/'request.json',{'inputs':files,'arguments':arguments,'recipe':recipe,'entry':entry})
                write_json(directory/'process/process.json',receipt['process'])
                write_json(directory/'jail/output/scores.json',response)
                if interrupted and len(calls)==1:raise OSError('interrupted before collection receipt')
                write_json(directory/'receipt.json',receipt);return receipt
            with patch('rec_harness.adapters.amazon_final.run_cpu_job',side_effect=infer):
                if interrupted:
                    with self.assertRaisesRegex(OSError,'before collection'):run_final(output,{},root/'uids')
                report=run_final(output,{},root/'uids')
            self.assertEqual(len(calls),8)
            if interrupted:
                recovered=[r for r in report['records'] if r['inference_recovery']]
                self.assertEqual(len(recovered),1);self.assertIsNone(recovered[0]['inference_wall_seconds'])
                self.assertEqual(recovered[0]['inference_process_wall_seconds'],0.1)
            self.assertEqual(report['paired_effects_by_seed']['17']['candidate']['HR@10'],1)
            self.assertEqual(report['paired_effects_by_seed']['83']['candidate']['HR@10'],1)
            self.assertEqual(len({r['request_hash'] for r in report['records'] if r['dataset']=='a'}),1)
            with patch('rec_harness.adapters.amazon_final.run_cpu_job',side_effect=AssertionError('repeated inference')):
                self.assertEqual(run_final(output,{},root/'uids'),report)
                with self.assertRaisesRegex(ContractError,'configuration changed'):run_final(output,{'changed':True},root/'uids')
                path=next((output/'jobs').glob('*/host-ranking.json'));bad=read_json(path);bad['metrics']['HR@10']=123;write_json(path,bad)
                with self.assertRaisesRegex(ContractError,'ranking report changed'):run_final(output,{},root/'uids')
            store=Store(study/'memory')
            try:self.assertEqual(store.verify(),sealed) # no final results in research memory
            finally:store.close()
