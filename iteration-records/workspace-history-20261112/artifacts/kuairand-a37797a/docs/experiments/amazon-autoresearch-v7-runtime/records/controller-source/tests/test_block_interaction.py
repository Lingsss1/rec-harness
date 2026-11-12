import tempfile
import unittest
from pathlib import Path
from rec_harness.core import ContractError,digest,read_json,write_json
from rec_harness.adapters.independent_scoring import prepare_request,score_response
from rec_harness.adapters.block_interaction import summarize_interaction


class BlockInteractionTests(unittest.TestCase):
    def test_factorial_contrast_recomputes_scores_and_rejects_mixed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directories=[]
            for index,blocks in enumerate([[],[0],[1],[0,1]]):
                directory=root/str(index);directories.append(directory)
                request,key=prepare_request([{'user_id':42,'history':[101],'candidates':list(range(1,101))}],seed=index)
                scores=[0.0]*100;scores[key['rows'][0]['positive_index']]=1.0 if index in [0,3] else -1.0
                response={'schema':'candidate-scores-response-v1','request_hash':digest(request),
                          'rows':[{'row_id':'row-0','scores':scores}]}
                report=score_response(request,key,response)
                value={k:'fixture' for k in ['dataset','model_version','protocol_hash','source_hash','checkpoint_hash',
                       'candidates_hash','score_worker_hash','host_scorer_hash']}
                value.update(seed=42,users=1,host_report_hash=report['hash'],host_metrics=report['metrics'],rank_disagreements=int(index in [1,2]))
                if blocks:value['bypass_blocks']=blocks
                value['hash']=digest(value)
                for name,data in [('acceptance.json',value),('inference-request.json',request),('private-key.json',key),('inference/results/scores.json',response)]:
                    write_json(directory/name,data)
            result=summarize_interaction(directories)
            self.assertEqual(result['metrics']['HR@10']['interaction'],2)
            self.assertEqual(result,summarize_interaction(list(reversed(directories))))
            with self.assertRaisesRegex(ContractError,'duplicate'):
                summarize_interaction(directories[:3]+[directories[0]])
            path=directories[3]/'acceptance.json';original=read_json(path)
            changed={**original,'checkpoint_hash':'different'};changed['hash']=digest({k:v for k,v in changed.items() if k!='hash'})
            write_json(path,changed)
            with self.assertRaisesRegex(ContractError,'unpaired'):summarize_interaction(directories)
            write_json(path,original)
            directory=directories[3]
            saved={name:(directory/name).read_bytes() for name in ['inference-request.json','private-key.json','inference/results/scores.json','acceptance.json']}
            request=read_json(directory/'inference-request.json');request['rows'][0]['history']=[102]
            key=read_json(directory/'private-key.json');key['request_hash']=digest(request)
            scores=read_json(directory/'inference/results/scores.json');scores['request_hash']=digest(request)
            altered=score_response(request,key,scores)
            value={**original,'host_report_hash':altered['hash']};value['hash']=digest({k:v for k,v in value.items() if k!='hash'})
            for name,data in [('inference-request.json',request),('private-key.json',key),('inference/results/scores.json',scores),('acceptance.json',value)]:write_json(directory/name,data)
            with self.assertRaisesRegex(ContractError,'histories or candidate sets'):summarize_interaction(directories)
            for name,data in saved.items():(directory/name).write_bytes(data)
            score_path=directories[3]/'inference/results/scores.json';response=read_json(score_path)
            response['rows'][0]['scores']=[0.0]*100;write_json(score_path,response)
            with self.assertRaisesRegex(ContractError,'scores differ'):summarize_interaction(directories)
