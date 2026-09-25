"""Amazon project routing fixtures, not GPU or online evolution utility evidence."""
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
from rec_harness.autonomous import autoresearch
from rec_harness.core import digest,file_hash,read_json,write_json
from rec_harness.measurement_gaps import block_model_proposal
from rec_harness.store import Store
import test_project


class AmazonProjectTests(unittest.TestCase):
    setUp=test_project.ProjectTests.setUp
    cpu=test_project.ProjectTests.cpu

    def config(self,kind):
        path=test_project.ProjectTests.config(self,kind);config=read_json(path)
        study=self.root/'study';study.mkdir();source=study/'base.py';source.write_text('# public model fixture\n')
        cfg={'model_versions':{'base':{'path':str(source),'sha256':file_hash(source)}},'protocol_hash':'p',
            'protocol':{'metric':'HR@10'},'datasets':['A'],'development_seeds':[42],'confirmation_seeds':[17,83]}
        write_json(study/'study.json',cfg);write_json(study/'study-lock.json',{'config_hash':digest(cfg)})
        store=Store(study/'memory');store.configure_budget(10000);store.close()
        old=config['model'];config['model']={'adapter':'amazon-nextitnet-v1','study':str(study),'task_id':'initial',
            'base_model_id':'base','max_rounds':1,'role_call_limit':12,'rescore_recipe':{'fixture':True},
            'uid_registry':'unused','agent_backend':{},'diagnostic_budget':4}
        if 'measurement_requirements' in old:config['model']['measurement_requirements']=old['measurement_requirements']
        write_json(path,config);return path

    @staticmethod
    def round(instance,state,folder):
        folder.mkdir(parents=True,exist_ok=True);context=instance._context(state,folder)
        source=Path(read_json(instance.study/'study.json')['model_versions'][state['incumbent']]['path'])
        gap=instance._role(folder,'evidence',context,source)
        block_model_proposal(instance,folder.name,gap,context.get('measurement_gaps',[]))
        readout={'schema':'amazon-development-selection-v1','protocol_hash':'p',
            'rows':[{'dataset':'A','seed':42,'metrics':{'HR@10':0.5}}]};readout['hash']=digest(readout)
        write_json(folder/'development-readout.json',readout)
        write_json(folder/'private-confirmation.json',{'result':'DO_NOT_EXPORT'})
        return {'selected':False,'status':'fixture-only'}

    def run_project(self,kind):
        with patch.object(AmazonAutoresearch,'_round',self.round), \
             patch('rec_harness.adapters.amazon_autonomous.audit_job',return_value=({'metrics':{'HR@10':0.5}},{})), \
             patch('rec_harness.adapters.amazon_autonomous.run_phase',side_effect=AssertionError('unit test cannot train')), \
             patch('rec_harness.diagnostics.run_cpu_job',side_effect=self.cpu):
            report=autoresearch(self.campaign,self.config(kind),agent=self.agent)
            self.assertEqual([r['kind'] for r in report['episodes']],['model',kind,'model'])
            self.assertEqual(autoresearch(self.campaign,agent=self.agent),report)
        child=read_json(self.campaign/'episodes/model-2/campaign/amazon-auto.json')
        self.assertEqual(child['base_model_id'],'base')
        self.assertNotIn('DO_NOT_EXPORT',str(child));return child

    def test_tool_update_reenters_amazon_with_measurement_requirement(self):
        child=self.run_project('tool')
        self.assertIn('diagnostic_tool',child)
        first=read_json(self.campaign/'episodes/model-1/campaign/run-report.json')
        self.assertEqual(first['state']['stop_reason'],'measurement_required')
        second=read_json(self.campaign/'episodes/model-2/campaign/run-report.json')
        self.assertEqual(second['role_calls']['charged_or_reserved'],1)

    def test_harness_update_uses_amazon_prefix_and_only_development_memory(self):
        child=self.run_project('harness')
        self.assertEqual(child['research_policy'],{'retrieval_limit':25})
        self.assertTrue(child['development_memory'][0]['entries'])
        optimizer=read_json(self.campaign/'episodes/harness-1/campaign/evolution-auto.json')
        trace=optimizer['context']['decision_traces'][0]
        self.assertEqual(trace['source_visibility'],'exact snapshot')
        self.assertNotIn('DO_NOT_EXPORT',str(optimizer['context']))
