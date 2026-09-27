from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from rec_harness.core import Artifacts,ContractError,digest,file_hash,read_json,write_json
from rec_harness.store import Store
from rec_harness.adapters.amazon_review_recovery import reuse_design,audit_frozen_rescore


class ReviewReuseTests(unittest.TestCase):
    def test_frozen_scorer_changes_fail_before_process_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);worker=root/'package/adapters/score_worker.py'
            worker.parent.mkdir(parents=True);worker.write_text('original')
            directory=root/'scores'
            write_json(directory/'inference/request.json',{'inputs':{'worker.py':{'path':str(worker)}}})
            execution={'code_hashes':{'adapters/score_worker.py':file_hash(worker)}}
            worker.write_text('changed')
            with patch('rec_harness.adapters.amazon_review_recovery.run_process') as launch:
                with self.assertRaisesRegex(ContractError,'scorer package changed'):
                    audit_frozen_rescore(SimpleNamespace(root=root),directory,{},execution)
                launch.assert_not_called()
                (worker.parent/'extra.py').write_text('unexpected')
                with self.assertRaisesRegex(ContractError,'inventory differs'):
                    audit_frozen_rescore(SimpleNamespace(root=root),directory,{},execution)
                launch.assert_not_called()

    def test_review_only_controller_cannot_enter_training_loop(self):
        from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
        controller=AmazonAutoresearch.__new__(AmazonAutoresearch)
        controller.config={'review_only':True}
        with self.assertRaisesRegex(ContractError,'cannot run training'):
            controller.run()

    def test_event_bound_design_copy_is_idempotent_and_rejects_changed_source_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'original/rounds/round-1';target=root/'new/rounds/round-1'
            cards={'gap':{'gap_id':'g'},'method':{'method_id':'m'},'proposal':{'proposal_id':'p'}}
            model=source/'experiment/model/model.py';model.parent.mkdir(parents=True);model.write_text('candidate')
            versions={'candidate':{'model_version':'candidate','source':{'sha256':file_hash(model)}},
                      'controls':[{'model_version':'base','source':{'sha256':'base'}}]}
            plan={'cards':cards,'versions':versions};write_json(source/'handoff.json',plan)
            store=Store(root/'original/memory');store.configure_budget(10)
            store.reserve('calls',3);store.settle('calls',3,'success')
            objects=Artifacts(root/'original/artifacts/objects')
            for role,response in [('evidence',cards['gap']),('hypothesis',{k:cards[k] for k in ['method','proposal']}),('experiment',{'changed_files':['model.py']})]:
                folder=source/role
                for name,value in [('context.json',{'role':role}),('response.json',response),('request.json',{}),('schema.json',{}),('resources.json',{'status':'success'})]:
                    write_json(folder/name,value)
                accepted={'response_hash':digest(response),'context_hash':digest({'role':role})}
                if role=='experiment':accepted['source_hash']=file_hash(model)
                write_json(folder/'accepted.json',accepted)
                store.event('role_completed',{'task':'round-1-'+role+'-1','role':role,**accepted,
                    'archive':objects.snapshot(folder,['context.json','response.json','resources.json'])})
            before=store.verify();store.close()
            first=reuse_design(None,source,target,plan)
            self.assertEqual(reuse_design(None,source,target,plan),first)
            self.assertEqual(first[1]['charged_or_reserved'],3)
            self.assertEqual(read_json(target/'experiment/reuse.json')['source_response_hash'],digest({'changed_files':['model.py']}))
            store=Store(root/'original/memory',read_only=True)
            self.assertEqual(store.verify(),before);store.close()
            write_json(source/'evidence/resources.json',{'status':'success','wall_seconds':0})
            with self.assertRaisesRegex(ContractError,'committed role archive changed'):
                reuse_design(None,source,target,plan)
            model.write_text('changed')
            with self.assertRaisesRegex(ContractError,'registered candidate'):
                reuse_design(None,source,target,plan)
