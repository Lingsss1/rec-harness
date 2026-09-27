from pathlib import Path
import tempfile
import unittest
from rec_harness.core import Artifacts,ContractError,digest,file_hash,read_json,write_json
from rec_harness.store import Store
from rec_harness.adapters.amazon_review_recovery import reuse_design


class ReviewReuseTests(unittest.TestCase):
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
