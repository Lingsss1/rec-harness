import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
from rec_harness.agents import schema_for
from rec_harness.core import ContractError,digest,file_hash,read_json,write_json
from rec_harness.store import Store
from test_amazon_plan import sample


class AmazonValidationToolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);study=self.root/'study';study.mkdir()
        source=study/'model.py';source.write_text('# public model fixture\n')
        cfg={'model_versions':{'base':{'path':str(source),'sha256':file_hash(source)}},
             'protocol_hash':'p','protocol':{'metric':'HR@10'},'datasets':['A','B'],
             'development_seeds':[42],'confirmation_seeds':[17,83]}
        write_json(study/'study.json',cfg);write_json(study/'study-lock.json',{'config_hash':digest(cfg)})
        store=Store(study/'memory');store.configure_budget(10000);self.study_head=store.verify()['head'];store.close()
        self.config={'study':str(study),'task_id':'validation','base_model_id':'base',
                     'max_rounds':1,'role_call_limit':6,'rescore_recipe':{'fixture':True},
                     'uid_registry':'unused','agent_backend':{},'diagnostic_budget':4,
                     'structured_insertion_contract':False}
        tool=self.root/'tool';tool.mkdir()
        (tool/'diagnostic.py').write_text("def measure(request):\n values=[r['metrics']['HR@10'] for r in request['runs']]\n return {'measurements':{'spread':max(values)-min(values)},'limitations':['aggregate only']}\n")
        self.tool={'source':str(tool),'files':['diagnostic.py'],'cost':2,'timeout':10,
                   'execution':{'recipe':{},'python':'unused','uid_registry':'unused'}}

    def test_unpromoted_tool_executes_frozen_source_and_reaches_role_once(self):
        campaign=self.root/'campaign';AmazonAutoresearch.init(campaign,self.config,_validation_tool=self.tool)
        frozen=read_json(campaign/'amazon-auto.json')
        self.assertTrue(frozen['validation_only']);self.assertTrue(frozen['diagnostic_tool']['validation_only'])
        self.assertNotIn('registration',frozen['diagnostic_tool'])
        self.assertNotIn('diagnostic_registry',frozen)
        # The archived implementation, not the mutable author workspace, runs.
        (self.root/'tool/diagnostic.py').write_text('raise AssertionError("mutable draft executed")\n')
        calls=[];roles=[]
        class Agent:
            def run(_,role,context,target):
                roles.append(context)
                response=sample(schema_for(role));response.update(model_version='base',protocol_hash='p')
                write_json(target/'response.json',response)
                return response
        controller=AmazonAutoresearch(campaign,agent=Agent())
        try:
            def cpu(directory,recipe,files,*args,**kwargs):
                from rec_harness.diagnostic_worker import main
                inputs=Path(directory)/'inputs';output=Path(directory)/'output';output.mkdir(parents=True)
                for name,ref in files.items():
                    target=inputs/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ref['path'],target)
                calls.append(read_json(inputs/'request.json'));main(inputs,output)
                return {'process':{'status':'success'},'collection':{'artifacts':{'measurement.json':controller.artifacts.put(output/'measurement.json')}}}
            def audited(study,cfg,model,dataset,seed):
                return {'metrics':{'HR@10':0.75 if dataset=='A' else 0.25}},{}
            folder=campaign/'rounds/round-1';folder.mkdir(parents=True)
            with patch('rec_harness.adapters.amazon_autonomous.audit_job',side_effect=audited),patch('rec_harness.diagnostics.run_cpu_job',side_effect=cpu):
                with controller.lock():
                    first=controller._context({'incumbent':'base','round':0},folder)
                    second=controller._context({'incumbent':'base','round':0},folder)
                    self.assertEqual(first,second)
                    controller._role(folder,'evidence',first)
                    controller._role(folder,'evidence',first)
            self.assertEqual(len(calls),1);self.assertEqual(len(roles),1)
            self.assertEqual(roles[0]['diagnostic']['output']['measurements'],{'spread':0.5})
            self.assertEqual({r['seed'] for r in calls[0]['runs']},{42})
            self.assertEqual(controller.store.budget()['charged_or_reserved'],1)
        finally:controller.close()
        costs=Store(campaign/'diagnostic-budget',read_only=True)
        try:self.assertEqual(costs.budget()['charged_or_reserved'],2)
        finally:costs.close()
        study=Store(self.root/'study/memory',read_only=True)
        try:self.assertEqual(study.verify()['head'],self.study_head)
        finally:study.close()

    def test_validation_cannot_override_registry_or_review_or_self_assert_marker(self):
        for changes,options in [({'diagnostic_registry':{}},{}),({}, {'review_only':True}),({'validation_only':True},{})]:
            with self.subTest(changes=changes,options=options):
                with self.assertRaises(ContractError):
                    AmazonAutoresearch.init(self.root/'blocked',{**self.config,**changes},_validation_tool=self.tool,**options)
                self.assertFalse((self.root/'blocked').exists())

    def test_explicit_diagnostic_budget_and_implementation_required(self):
        config=dict(self.config);config.pop('diagnostic_budget')
        with self.assertRaises(ContractError):AmazonAutoresearch.init(self.root/'no-budget',config,_validation_tool=self.tool)
        with self.assertRaises(ContractError):
            AmazonAutoresearch.init(self.root/'no-entry',self.config,_validation_tool={**self.tool,'files':[]})
        self.assertFalse((self.root/'no-entry/amazon-auto.json').exists())

    def test_ordinary_campaign_does_not_gain_validation_tool(self):
        AmazonAutoresearch.init(self.root/'ordinary',self.config)
        config=read_json(self.root/'ordinary/amazon-auto.json')
        self.assertNotIn('diagnostic_tool',config);self.assertNotIn('validation_only',config)
