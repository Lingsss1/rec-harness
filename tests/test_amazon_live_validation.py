"""Exercise the live-validation harness without paying for model calls in CI."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rec_harness.agents import schema_for,search_receipt
from rec_harness.core import read_json,write_json
from test_amazon_plan import sample

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('live_validation',ROOT/'scripts/validate_evidence_live.py')
live=importlib.util.module_from_spec(spec);spec.loader.exec_module(live)


class ScriptedAgent:
    def __init__(self,root):pass
    def run(self,role,context,target):
        response=sample(schema_for(role));round_no=int(target.parent.name.split('-')[-1])
        if role=='researcher':
            bindings=context['bindings'];refs=[context['evidence_graph']['nodes'][0]['id']]
            response['gap'].update(gap_id='gap-'+str(round_no),model_version=bindings['base_model_id'],
                protocol_hash=bindings['train_protocol_hash'],observation_refs=refs)
            response['method'].update(method_id='method-'+str(round_no),origin='agent_hypothesis',sources=[])
            response['proposal'].update({k:v for k,v in bindings.items() if k!='candidate_model_version'})
            response['proposal'].update(gap_ids=[response['gap']['gap_id']],method_ids=[response['method']['method_id']],
                budget=1000,insertion_contract=json.dumps({'schema':'nextitnet-method-scope-v1',
                    'allowed_methods':['NextItNetModel.encode_sequence'],'rationale':'simulation source change'}))
            response['evidence_refs']=refs
        if role=='experiment':
            source=(target/'model/model.py').read_text()
            a=.6+round_no*.01;b=.4+round_no*.01
            (target/'model/model.py').write_text(source.replace('0.85 * x + 0.15 * context',f'{a} * x + {1-a} * context'))
            (target/'model/control-1.py').write_text(source.replace('0.85 * x + 0.15 * context',f'{b} * x + {1-b} * context'))
            response['changed_files']=['model.py','control-1.py']
        if role=='reviewer':response.update(verdict='approve',checks=[],blocking_findings=[],evidence_refs=[context['implementation_hash']])
        if role=='analyst':
            graph=context['graph'];fact=next(n['id'] for n in graph['nodes'] if n['attrs'].get('kind')=='derived_development_comparison')
            response['decision'].update(selection='discard',conclusion_status='unknown',evidence_refs=[context['development']['hash']])
            attrs={'task_id':graph['task_id'],'protocol_hash':context['protocol_hash'],
                   'origin':'agent_hypothesis','status':'provisional',
                   'scope':'simulation only','evidence_refs':[fact],'description':'Real measurements still required.'}
            response['knowledge']['patch_json']=json.dumps({'base_version':graph['graph_version'],
                'nodes':[{'id':'runtime-gap-'+str(round_no),'type':'MeasurementGap','attrs':attrs}],'edges':[]})
        for name,value in [('context.json',context),('request.json',{'role':role}),('response.json',response),
                           ('schema.json',schema_for(role)),('resources.json',{'status':'success'}),('process.json',{'status':'success'})]:
            write_json(target/name,value)
        (target/'stdout.jsonl').write_text('{"type":"scripted_test"}\n')
        search_receipt(target,'disabled',response)
        return response


class LiveValidationTests(unittest.TestCase):
    def test_actual_controller_graph_roundtrip_and_interruption_with_compute_stub(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(live,'LiveAgent',ScriptedAgent):
            report=live.run(Path(tmp))
            self.assertEqual(report['audit']['verified_cycles'],2)
            self.assertEqual(report['audit']['verified_roles'],8)
            self.assertTrue(report['interrupted']);self.assertTrue(report['terminal_resume_no_new_calls'])
            second=report['rounds'][1]
            self.assertIn('runtime-gap-1',second['native_measurement_gaps'])
            self.assertEqual(second['graph_hash'],second['analyst_history_hash'])
            self.assertTrue(report['rounds'][0]['imported_interpretations'])
            self.assertFalse(report['result']['failed_rounds'])
            probe=live.probe_readback(Path(tmp))
            self.assertIn('runtime-gap-2',probe['native_measurement_gaps_visible'])
            self.assertEqual(probe['role_cost']['charged_or_reserved'],1)
