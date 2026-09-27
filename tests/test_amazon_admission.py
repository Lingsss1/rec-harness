import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rec_harness.agents import schema_for
from rec_harness.core import ContractError,digest,file_hash,write_json
from rec_harness.store import Store
from rec_harness.adapters.amazon_plan import register_plan
from rec_harness.adapters.amazon_admission import admit_plan
from rec_harness.adapters.amazon_controls import load_config
from test_amazon_plan import sample
from test_model_source_contract import SOURCE


class MultiModelAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.base=self.source('base',SOURCE)
        config={'model_versions':{'base':self.base['source']},'protocol_hash':'p',
                'datasets':['tiny'],'development_seeds':[42],'confirmation_seeds':[17,83],'python':'unused'}
        write_json(self.root/'study.json',config)
        write_json(self.root/'study-lock.json',{'config_hash':digest(config)})
        store=Store(self.root/'memory');store.configure_budget(100);store.close()
        gap=sample(schema_for('evidence'));generated=sample(schema_for('hypothesis'))
        gap.update(gap_id='gap',model_version='base',protocol_hash='p')
        generated['method']['method_id']='method'
        generated['proposal'].update(proposal_id='proposal',base_model_id='base',train_protocol_hash='p',
            gap_ids=['gap'],method_ids=['method'],controls=['base','control'],seeds=[42,17,83],budget=50)
        self.cards={'gap':gap,**generated}
        self.versions={'candidate':self.source('candidate',SOURCE.replace('return self.block(x)','return self.block(x) * 0.9')),
                       'controls':[self.base,self.source('control',SOURCE.replace('return self.block(x)','return self.block(x) * 0.8'))]}

    def source(self,name,text):
        path=self.root/(name+'.py');path.write_text(text)
        return {'model_version':name,'source':{'path':str(path),'sha256':file_hash(path)}}

    def gate(self,python,source,name,directory,env,timeout):
        write_json(directory/'report.json',{'checks':{'parameter_gradients':{'weight':{'finite':True}}}})
        return {'status':'success'}

    def smoke(self,python,source,directory,recipe,artifacts,registry):
        return {'status':'success','report_ref':artifacts.put(source['path'])}

    def test_all_new_sources_checked_once_and_committed_atomically(self):
        self.cards['proposal']['insertion_contract']=json.dumps({'schema':'nextitnet-method-scope-v1','allowed_methods':['NextItNetModel.forward'],'rationale':'scale forward only'})
        register_plan(self.root,self.cards,self.versions)
        with patch('rec_harness.adapters.amazon_admission.run_gate',side_effect=self.gate) as gate, \
             patch('rec_harness.adapters.amazon_admission.run_training_smoke',side_effect=self.smoke) as smoke:
            first=admit_plan(self.root,'proposal',recipe={},uid_registry='unused')
            self.assertEqual(set(first['versions']),{'candidate','control'})
            for item in first['versions'].values():
                self.assertEqual(item['source_contract']['insertion_scope']['changed_methods'],['NextItNetModel.forward'])
            self.assertEqual([call.args[2] for call in gate.call_args_list],['candidate','control'])
            self.assertEqual(smoke.call_count,2)
            self.assertEqual(admit_plan(self.root,'proposal',recipe={},uid_registry='unused'),first)
            self.assertEqual(gate.call_count,2);self.assertEqual(smoke.call_count,2)
        config=load_config(self.root)
        self.assertEqual(set(config['model_versions']),{'base','candidate','control'})
        store=Store(self.root/'memory',read_only=True)
        try:self.assertEqual(sum(e['kind']=='amazon_model_candidate_admitted' for e in store.events()),1)
        finally:store.close()

    def test_generated_control_outside_insertion_scope_rejected_before_cpu(self):
        self.cards['proposal']['insertion_contract']=json.dumps({'schema':'nextitnet-method-scope-v1',
            'allowed_methods':['NextItNetModel.forward'],'rationale':'readout only'})
        self.versions['controls'][1]=self.source('control',SOURCE.replace('return self.linear(x)','return self.linear(x)*0.8'))
        register_plan(self.root,self.cards,self.versions)
        with patch('rec_harness.adapters.amazon_admission.run_gate',side_effect=AssertionError('CPU must not start')):
            with self.assertRaisesRegex(ContractError,'outside frozen insertion scope'):
                admit_plan(self.root,'proposal',recipe={},uid_registry='unused')
        self.assertEqual(set(load_config(self.root)['model_versions']),{'base'})

    def test_failing_control_never_partially_registers_candidate(self):
        register_plan(self.root,self.cards,self.versions)
        def gate(*args):
            return {'status':'failed'} if args[2]=='control' else self.gate(*args)
        with patch('rec_harness.adapters.amazon_admission.run_gate',side_effect=gate), \
             patch('rec_harness.adapters.amazon_admission.run_training_smoke',side_effect=self.smoke):
            with self.assertRaisesRegex(ContractError,'numerical contract failed'):
                admit_plan(self.root,'proposal',recipe={},uid_registry='unused')
        self.assertEqual(set(load_config(self.root)['model_versions']),{'base'})
        self.assertTrue(list((self.root/'model-admissions').glob('proposal/*/contract/report.json')))

    def test_invalid_control_source_rejected_before_any_cpu_execution(self):
        self.versions['controls'][1]=self.source('control',SOURCE+'\nprint("outside model")\n')
        register_plan(self.root,self.cards,self.versions)
        with patch('rec_harness.adapters.amazon_admission.run_gate',side_effect=AssertionError('CPU must not start')):
            with self.assertRaisesRegex(ContractError,'frozen imports'):
                admit_plan(self.root,'proposal',recipe={},uid_registry='unused')
        self.assertEqual(set(load_config(self.root)['model_versions']),{'base'})
