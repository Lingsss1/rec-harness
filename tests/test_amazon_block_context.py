import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rec_harness.core import Artifacts,ContractError,canonical,digest,file_hash,read_json,write_json
from rec_harness.store import Store
from rec_harness.adapters.independent_scoring import prepare_request,score_response
from rec_harness.adapters.amazon_block_context import project_reports,freeze_reports,block_context,cost_summary


class AmazonBlockContextTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.study=self.root/'study';self.directory=self.root/'probe'
        self.cfg={'protocol_hash':'p','datasets':['tiny'],'development_seeds':[42],
            'data_hashes':{'train':'frozen'},'model_versions':{'base':{'sha256':'a'*64}}}

    def prepare(self,bad_metrics=False):
        config={'study':str(self.study),'model_version':'base','dataset':'tiny','seed':42,'blocks':[0]}
        write_json(self.directory/'config.json',config)
        store=Store(self.directory/'memory');objects=Artifacts(self.directory/'objects');scores=Artifacts(self.study/'artifacts/objects')
        self.addCleanup(store.close);store.configure_budget(2);store.event('block_scores_registered',{'config_hash':digest(config)})
        request,key=prepare_request([{'user_id':90210,'history':[1,2],'candidates':list(range(1,101))}],seed=13)
        base=Path(__file__).resolve().parents[1]/'rec_harness/adapters'
        for task,block in [('reference',None),('block-0',0)]:
            values=[0.]*100;position=key['rows'][0]['positive_index'];values[position]=1.
            if block is not None:values[(position+1)%100]=2.
            response={'schema':'candidate-scores-response-v1','request_hash':digest(request),'rows':[{'row_id':'row-0','scores':values}]}
            ranked=score_response(request,key,response);folder=self.directory/task
            write_json(folder/'scores.json',response)
            receipt={'process':{'status':'success'},'collection':{'artifacts':{'scores.json':scores.put(folder/'scores.json')}}}
            summary={'model_version':'base','dataset':'tiny','seed':42,'source_hash':'a'*64,'checkpoint_hash':'b'*64,
                'score_worker_hash':file_hash(base/'score_worker.py'),'host_scorer_hash':file_hash(base/'independent_scoring.py'),
                'host_metrics':ranked['metrics'],'host_report_hash':ranked['hash'],'users':1,'rank_disagreements':int(block is not None),
                'qualification':'SECRET_NOT_PUBLIC'}
            if block is None:
                self.baseline={'source_hash':'a'*64,'checkpoint_sha256':'b'*64,'metrics':dict(ranked['metrics'])}
                write_json(self.study/'jobs/base-tiny-42/validation_ranks.json',{'user_ids':[90210],'ranks':ranked['ranks']})
            else:summary['bypass_block']=block
            if bad_metrics and block is not None:summary['host_metrics']={**ranked['metrics'],'HR@10':0.5}
            summary['hash']=digest(summary)
            contents={'acceptance.json':summary,'host-ranking.json':ranked,'private-key.json':key,'inference-request.json':request,'inference/receipt.json':receipt}
            refs={}
            for name,value in contents.items():write_json(folder/name,value);refs[name]=objects.put(folder/name)
            store.reserve(task,1);store.event('block_score_completed',{'task':task,'artifacts':refs});store.settle(task,1,'success')

    def source_patches(self):
        return patch('rec_harness.adapters.amazon_block_context.load_config',return_value=self.cfg),patch('rec_harness.adapters.amazon_block_context.audit_job',return_value=(self.baseline,{}))

    def test_projection_freeze_context_idempotence_and_stale_model_filter(self):
        self.prepare();first,second=self.source_patches()
        artifacts=Artifacts(self.root/'campaign/artifacts/objects')
        with first,second:
            projected=project_reports(self.directory,self.cfg)
            self.assertNotIn('SECRET_NOT_PUBLIC',canonical(projected));self.assertNotIn('90210',canonical(projected))
            records=freeze_reports([self.directory],self.cfg,artifacts)
            with self.assertRaisesRegex(ContractError,'duplicate'):freeze_reports([self.directory,self.directory],self.cfg,artifacts)
        store=Store(self.root/'campaign/memory');self.addCleanup(store.close)
        controller=SimpleNamespace(config={'block_score_diagnostics':records},artifacts=artifacts,store=store,task='t',protocol_hash='p')
        context=block_context(controller,'base',self.cfg);before=store.verify()
        self.assertEqual(block_context(controller,'base',self.cfg),context);self.assertEqual(store.verify(),before)
        self.assertEqual(block_context(controller,'new-model',self.cfg),{})
        self.assertEqual(cost_summary(controller.config)['retained_diagnostic_bcu'],2)
        self.assertEqual(cost_summary(controller.config)['new_diagnostic_bcu'],0)
        path=artifacts.verify(records[0]['public_ref']);path.write_text('{}')
        with self.assertRaises(ContractError):block_context(controller,'base',self.cfg)

    def test_self_consistent_summary_cannot_override_host_recomputed_scores(self):
        self.prepare(bad_metrics=True);first,second=self.source_patches()
        with first,second,self.assertRaisesRegex(ContractError,'host recomputation'):project_reports(self.directory,self.cfg)

    def test_confirmation_rejected_before_checkpoint_audit(self):
        self.prepare();config=read_json(self.directory/'config.json');config['seed']=17;write_json(self.directory/'config.json',config)
        with patch('rec_harness.adapters.amazon_block_context.load_config',return_value=self.cfg),patch('rec_harness.adapters.amazon_block_context.audit_job') as audit:
            with self.assertRaisesRegex(ContractError,'development'):project_reports(self.directory,self.cfg)
            audit.assert_not_called()
