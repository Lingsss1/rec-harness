import tempfile
import unittest
import copy
import shutil
from pathlib import Path
from rec_harness.core import Artifacts,ContractError,digest,write_json
from rec_harness.store import Store
from rec_harness.tool_exposure import audit_exposure


class ExposureTests(unittest.TestCase):
    def test_exact_input_and_no_future_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);objects=Artifacts(root/'artifacts/objects')
            (root/'diagnostic.py').write_text('tool fixture')
            source={'diagnostic.py':objects.put(root/'diagnostic.py')}
            tool={'implementation':source,'implementation_hash':digest(source),'candidate_id':'candidate','worker':source['diagnostic.py']}
            config={'task_id':'task','diagnostic_tool':tool}
            write_json(root/'campaign.json',config);write_json(root/'research_archive/manifest.json',{'config_hash':digest(config)})
            value={'measurements':{'x':1},'limitations':[]};write_json(root/'measurement.json',value)
            record={'identity':'measurement','implementation_hash':digest(source),'tool_version':'candidate','model_version':'root',
                'status':'success','output':value,'artifacts':{'measurement.json':objects.put(root/'measurement.json')}}
            request={'task_id':'task','model_version':'root','protocol_hash':'protocol','runs':[]}
            record['input_hash']=digest(request)
            record['identity']=digest({'request':request,'tool':tool['implementation_hash'],'worker':tool['worker']})
            context={'task_id':'task','protocol_hash':'protocol','model_version':'root','diagnostic':record,'baseline':{'runs':[]}}
            write_json(root/'input.json',context)
            submitted={'task_id':'evidence-1','role':'evidence','visible_state_ref':objects.put(root/'input.json'),'input_hash':digest(context)}
            store=Store(root/'memory')
            store.event('diagnostic_completed',record);store.event('agent_submitted',submitted)
            store.event('agent_completed',{'task_id':'evidence-1','status':'success'});store.close()
            result=audit_exposure(root);self.assertTrue(result['exposures'][0]['transport_completed'])
            # The measurement reviewer receives the deliberately compact projection.
            projected={**context,'baseline':[], 'diagnostic':{k:record[k] for k in
                ['status','tool_version','implementation_hash','output']}}
            write_json(root/'review.json',projected)
            review={'task_id':'measurement-1','role':'measurement','visible_state_ref':objects.put(root/'review.json'),'input_hash':digest(projected)}
            store=Store(root/'memory');store.event('agent_submitted',review)
            store.event('agent_completed',{'task_id':'measurement-1','status':'success'});before=store.verify();store.close()
            result=audit_exposure(root)
            self.assertEqual([row['role'] for row in result['exposures']],['evidence','measurement'])
            self.assertEqual(result['exposures'][1]['measurement_identity'],record['identity'])
            self.assertEqual(result['integrity'],before)
            self.assertEqual(audit_exposure(root),result)
            for altered in ['output','baseline']:
                with self.subTest(altered=altered),tempfile.TemporaryDirectory() as other:
                    target=Path(other)/'campaign';shutil.copytree(root,target)
                    corrupted=copy.deepcopy(projected)
                    if altered=='output':corrupted['diagnostic']['output']['measurements']['x']=99
                    else:corrupted['baseline']=[{'seed':7,'valid':True,'metrics':{'quality':0.2}}]
                    write_json(target/'changed.json',corrupted)
                    copied=Artifacts(target/'artifacts/objects');ledger=Store(target/'memory')
                    ledger.event('agent_submitted',{**review,'task_id':'measurement-2',
                        'visible_state_ref':copied.put(target/'changed.json'),'input_hash':digest(corrupted)})
                    ledger.close()
                    with self.assertRaisesRegex(ContractError,'uncommitted or changed'):audit_exposure(target)
            # A different committed input may not introduce an observation that
            # only appears later in the event stream.
            context['diagnostic']={**record,'identity':'future'};write_json(root/'input.json',context)
            store=Store(root/'memory');store.event('agent_submitted',{**submitted,'task_id':'evidence-2','input_hash':digest(context),'visible_state_ref':objects.put(root/'input.json')})
            store.event('diagnostic_completed',context['diagnostic']);store.close()
            with self.assertRaisesRegex(ContractError,'uncommitted'):audit_exposure(root)
