import tempfile
import unittest
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
            # A different committed input may not introduce an observation that
            # only appears later in the event stream.
            context['diagnostic']={**record,'identity':'future'};write_json(root/'input.json',context)
            store=Store(root/'memory');store.event('agent_submitted',{**submitted,'task_id':'evidence-2','input_hash':digest(context),'visible_state_ref':objects.put(root/'input.json')})
            store.event('diagnostic_completed',context['diagnostic']);store.close()
            with self.assertRaisesRegex(ContractError,'uncommitted'):audit_exposure(root)
