import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from rec_harness.autonomous import autoresearch
from rec_harness.agents import schema_for
from rec_harness.core import Artifacts,ContractError,digest,file_hash,read_json,write_json
from rec_harness.store import Store
from test_amazon_plan import sample


class AmazonDiagnosticTests(unittest.TestCase):
    def test_missing_measurement_stops_before_proposal_and_gpu_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/'study';study.mkdir();source=study/'model.py';source.write_text('# public fixture\n')
            cfg={'model_versions':{'base':{'path':str(source),'sha256':file_hash(source)}},'protocol_hash':'p',
                'protocol':{'metric':'HR@10'},'datasets':['A','B'],'development_seeds':[42],'confirmation_seeds':[17,83]}
            write_json(study/'study.json',cfg);write_json(study/'study-lock.json',{'config_hash':digest(cfg)})
            store=Store(study/'memory');store.configure_budget(10000);store.close()
            config={'adapter':'amazon-nextitnet-v1','study':str(study),'task_id':'test','base_model_id':'base',
                'max_rounds':3,'role_call_limit':18,'rescore_recipe':{'fixture':True},'uid_registry':'unused','agent_backend':{},
                'measurement_requirements':[{'measurement':'category_spread','blocked_decision':'shared or per-category change',
                    'competing_explanations':['shared deficit','category effect'],'cheapest_test':'aggregate probe',
                    'development_budget':10,'acceptance_contract':'independent utility'}]}
            write_json(root/'config.json',config);calls=[]
            class Agent:
                def run(self,role,context,target):
                    calls.append(role);assert role=='evidence'
                    response=sample(schema_for(role));response.update(model_version='base',protocol_hash='p')
                    write_json(target/'context.json',context);write_json(target/'response.json',response)
                    return response
            with patch('rec_harness.adapters.amazon_autonomous.audit_job',return_value=({'metrics':{'HR@10':0.5}},{})), \
                 patch('rec_harness.adapters.amazon_autonomous.run_phase',side_effect=AssertionError('GPU must not run')):
                result=autoresearch(root/'campaign',root/'config.json',agent=Agent())
                self.assertEqual(result['state']['stop_reason'],'measurement_required');self.assertEqual(calls,['evidence'])
                self.assertEqual(result['role_calls']['charged_or_reserved'],1)
                self.assertEqual(autoresearch(root/'campaign',agent=Agent()),result)
            store=Store(root/'campaign/memory')
            try:
                graph=store.snapshot();self.assertEqual(sum(n['type']=='MeasurementGap' for n in graph['nodes']),1)
                self.assertEqual(sum(n['type']=='Experiment' for n in graph['nodes']),2)
                self.assertIn('BLOCKS',{e['relation'] for e in graph['edges']})
            finally:store.close()
            from rec_harness.measurement_gaps import evolution_context
            gap=next(n['id'] for n in graph['nodes'] if n['type']=='MeasurementGap')
            projection=root/'campaign/memory/events.jsonl'
            projection.write_text('preserve source projection\n')
            with patch.object(Store,'export_events',side_effect=AssertionError('measurement source mutated')):
                context=evolution_context(root/'campaign',gap)
                self.assertEqual(context,evolution_context(root/'campaign',gap))
            self.assertEqual(projection.read_text(),'preserve source projection\n')
            self.assertEqual({o['dataset'] for o in context['observations']},{'A','B'})
            self.assertEqual({o['seed'] for o in context['observations']},{42})

    def test_diagnostic_receives_categories_and_separate_cost_ledger(self):
        from rec_harness.adapters.amazon_diagnostics import diagnostic_context
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=Store(root/'memory');store.configure_budget(18)
            self.addCleanup(store.close);artifacts=Artifacts(root/'artifacts/objects')
            source=root/'tool.py';source.write_text('# fixture\n');ref=artifacts.put(source)
            implementation={'diagnostic.py':ref}
            config={'task_id':'task','protocol_hash':'p','diagnostic_budget':4,'diagnostic_tool':{
                'candidate_id':'tool','implementation':implementation,'implementation_hash':digest(implementation),
                'worker':ref,'execution':{'recipe':{},'python':'unused','uid_registry':'unused'},'cost':2,'timeout':5}}
            controller=SimpleNamespace(root=root,store=store,artifacts=artifacts,config=config)
            rows=[{'dataset':'A','seed':42,'metrics':{'HR@10':0.6}},{'dataset':'B','seed':42,'metrics':{'HR@10':0.3}}]
            def execute(directory,recipe,files,*args,**kwargs):
                request=read_json(files['request.json']['path'])
                self.assertEqual({row['dataset'] for row in request['runs']},{'A','B'})
                self.assertEqual({row['seed'] for row in request['runs']},{42})
                output=Path(directory)/'results/measurement.json'
                write_json(output,{'measurements':{'category_spread':0.3},'limitations':['aggregate only']})
                return {'process':{'status':'success'},'collection':{'artifacts':{'measurement.json':artifacts.put(output)}}}
            with patch('rec_harness.diagnostics.run_cpu_job',side_effect=execute) as worker:
                first=diagnostic_context(controller,'base',rows)
                self.assertEqual(diagnostic_context(controller,'base',rows),first);self.assertEqual(worker.call_count,1)
            self.assertEqual(store.budget()['charged_or_reserved'],0)
            costs=Store(root/'diagnostic-budget')
            try:self.assertEqual(costs.budget()['charged_or_reserved'],2)
            finally:costs.close()
            rows[0]['metrics']['HR@10']=0.8
            with self.assertRaisesRegex(ContractError,'aggregate changed'):diagnostic_context(controller,'base',rows)
