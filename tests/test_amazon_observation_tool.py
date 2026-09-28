"""Amazon project -> generated observation tool -> worker -> successor role fixtures."""
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
from rec_harness.autonomous import autoresearch
from rec_harness.core import Artifacts,ContractError,digest,read_json,write_json
from rec_harness.diagnostic_inputs import OBSERVATION_CONTRACT
from rec_harness.runtime import node
from rec_harness.store import Store
from rec_harness.tool_exposure import audit_exposure
import test_project
import test_amazon_project


class AmazonObservationToolTests(unittest.TestCase):
    setUp=test_project.ProjectTests.setUp

    def test_failed_observation_collection_prevents_generated_measurement(self):
        path=test_amazon_project.AmazonProjectTests.config(self,'tool')
        config=read_json(path)['model'];target=self.root/'direct-model'
        AmazonAutoresearch.init(target,config)
        controller=AmazonAutoresearch(target,agent=self.agent)
        try:
            # Exercise only context assembly order; no active tool or private probe is executed.
            controller.config['diagnostic_tool']={'contract':OBSERVATION_CONTRACT}
            with patch('rec_harness.adapters.amazon_autonomous.audit_job',return_value=({'metrics':{'HR@10':0.5}},{})), \
                 patch('rec_harness.adapters.amazon_activation.activation_context',return_value={'measurement_failure':'probe failed'}), \
                 patch('rec_harness.adapters.amazon_diagnostics.diagnostic_context',side_effect=AssertionError('must not measure')):
                context=controller._context({'incumbent':'base'},target/'round-1')
            self.assertEqual(context['measurement_failure'],'probe failed')
        finally:controller.store.close()

    def test_generated_contract_worker_current_observations_and_role_delivery(self):
        path=test_amazon_project.AmazonProjectTests.config(self,'tool');config=read_json(path)
        config['tool']['plan']['tool_contract']=OBSERVATION_CONTRACT;write_json(path,config)
        calls=[]
        class Agent(test_project.ProjectAgent):
            def run(agent,role,context,directory):
                if role!='tool':return super().run(role,context,directory)
                contract=context['context']['diagnostic_contract']
                assert contract['contract']==OBSERVATION_CONTRACT
                assert 'observations' in contract['request']
                source=Path(directory)/'model/diagnostic.py';source.parent.mkdir(parents=True,exist_ok=True)
                source.write_text('def measure(request):\n'
                    '    row=request["observations"]["activation_diagnostics"][0]\n'
                    '    values=list(row["measurements"].values())\n'
                    '    return {"measurements":{"spread":max(values)-min(values)},"limitations":row["limitations"]}\n')
                return {'changed_files':['diagnostic.py'],'limitations':['fixture']}
        def observations(controller,base,study):
            row={'observation_id':'fixture-activation','model_version':base,'dataset':'A','seed':42,
                 'measurements':{'norm-a':2,'norm-b':5},'limitations':['synthetic histories; no causal inference'],
                 'scope':'public development activation fixture'}
            if row['observation_id'] not in {n['id'] for n in controller.store.snapshot()['nodes']}:
                controller.store.patch({'base_version':controller.store.version,'nodes':[
                    node(row['observation_id'],'Observation',{**row,'valid':True},controller.task,controller.protocol_hash)],'edges':[]})
            return {'activation_diagnostics':[row]}
        def cpu(directory,recipe,files,*args,**kwargs):
            from rec_harness.diagnostic_worker import main
            root=Path(directory)/'inputs';out=Path(directory)/'output';out.mkdir(parents=True)
            for name,ref in files.items():
                destination=root/name;destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(Path(ref['path']).read_bytes())
            calls.append(read_json(root/'request.json'))
            main(root,out)
            return {'process':{'status':'success'},'collection':{'artifacts':{
                'measurement.json':kwargs['artifacts'].put(out/'measurement.json')}}}
        with patch.object(AmazonAutoresearch,'_round',test_amazon_project.AmazonProjectTests.round), \
             patch('rec_harness.adapters.amazon_autonomous.audit_job',return_value=({'metrics':{'HR@10':0.5}},{})), \
             patch('rec_harness.adapters.amazon_autonomous.run_phase',side_effect=AssertionError('no training')), \
             patch('rec_harness.adapters.amazon_activation.activation_context',side_effect=observations), \
             patch('rec_harness.diagnostics.run_cpu_job',side_effect=cpu):
            report=autoresearch(self.campaign,path,agent=Agent())
            self.assertEqual([r['kind'] for r in report['episodes']],['model','tool','model'])
            self.assertEqual(autoresearch(self.campaign,agent=Agent()),report)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0]['observations']['activation_diagnostics'][0]['measurements'],{'norm-a':2,'norm-b':5})
        child=self.campaign/'episodes/model-2/campaign';frozen=read_json(child/'amazon-auto.json')
        self.assertEqual(frozen['diagnostic_tool']['contract'],OBSERVATION_CONTRACT)
        exposure=audit_exposure(child);self.assertEqual(len(exposure['exposures']),1)
        self.assertTrue(exposure['exposures'][0]['transport_completed'])
        objects=Artifacts(child/'artifacts/objects');store=Store(child/'memory')
        try:
            submitted=next(e['payload'] for e in store.events() if e['kind']=='agent_submitted')
            context=read_json(objects.verify(submitted['visible_state_ref']))
            self.assertEqual(context['diagnostic']['output']['measurements'],{'spread':3})
            review={k:context[k] for k in ['task_id','model_version','protocol_hash','activation_diagnostics']}
            review['baseline']=context['measurement_baseline']['runs']
            review['diagnostic']={k:v for k,v in context['diagnostic'].items() if k in
                ['status','tool_version','implementation_hash','output','tool_registration']}
            write_json(child/'review.json',review)
            store.event('agent_submitted',{'task_id':'review','role':'measurement',
                'visible_state_ref':objects.put(child/'review.json'),'input_hash':digest(review)})
        finally:store.close()
        self.assertEqual(len(audit_exposure(child)['exposures']),2)
        # Omitting the input observation cannot pass a self-consistent role archive audit.
        review.pop('activation_diagnostics');write_json(child/'wrong-review.json',review)
        store=Store(child/'memory')
        try:store.event('agent_submitted',{'task_id':'wrong-review','role':'measurement',
            'visible_state_ref':objects.put(child/'wrong-review.json'),'input_hash':digest(review)})
        finally:store.close()
        with self.assertRaises(ContractError):audit_exposure(child)
