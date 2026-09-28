"""Committed observation shadow inputs; reference jobs and CPU launch are fixtures."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import Artifacts,ContractError,digest,read_json,write_json
from rec_harness.diagnostic_inputs import OBSERVATION_CONTRACT
from rec_harness.runtime import node
from rec_harness.store import Store
from rec_harness.tool_shadow import freeze_suite,audit_inputs,run_validation
import test_tool_shadow


class ObservationShadowTests(unittest.TestCase):
    setUp=test_tool_shadow.ShadowTests.setUp
    job=test_tool_shadow.ShadowTests.job

    def source(self):
        campaign=self.root/'observations';objects=Artifacts(campaign/'artifacts/objects')
        config={'task_id':'source-task','study':str(self.root/'study'),'protocol_hash':'p',
                'study_config_hash':digest(self.config)}
        write_json(campaign/'amazon-auto.json',config);write_json(campaign/'auto-lock.json',{'config_hash':digest(config)})
        store=Store(campaign/'memory');store.configure_budget(0)
        try:
            for version,value in [('A',0.25),('B',0.75)]:
                row={'observation_id':'activation-'+version,'model_version':version,'dataset':'D','seed':42,
                     'measurements':{'norm':value},'limitations':['synthetic histories'],'scope':'public development activation'}
                path=campaign/(version+'.json');write_json(path,row);ref=objects.put(path)
                store.patch({'base_version':store.version,'nodes':[
                    node(row['observation_id'],'Observation',{**row,'valid':True,'summary_ref':ref},'source-task','p')],'edges':[]},
                    accompanying_events=[('activation_summary_recorded',{'observation_id':row['observation_id'],'summary_ref':ref})])
        finally:store.close()
        spec=read_json(self.root/'spec.json');spec['tool_contract']=OBSERVATION_CONTRACT
        for case in spec['conditions']:
            case['observation_source']={'campaign':'observations','observation_ids':['activation-'+case['model_version']]}
        write_json(self.root/'spec.json',spec);return campaign

    def execute(self,directory,recipe,files,*args,**kwargs):
        request=read_json(files['request.json']['path'])
        self.assertNotIn(str(self.root),str(request));self.assertNotIn('host-only-hash',str(request))
        value=request['observations']['activation_diagnostics'][0]['measurements']['norm']
        output=Path(directory)/'results/measurement.json'
        write_json(output,{'measurements':{'mean':value},'limitations':['fixture']})
        return {'process':{'status':'success'},'wall_seconds':0.1,'collection':{'artifacts':{
            'measurement.json':kwargs['artifacts'].put(output)}}}

    def test_two_conditions_bind_exact_observations_and_ignore_later_append(self):
        campaign=self.source();suite=freeze_suite(self.root/'spec.json',self.root/'suite')
        self.assertEqual(suite['tool_contract'],OBSERVATION_CONTRACT)
        for case,value in zip(suite['cases'],[0.25,0.75]):
            self.assertEqual(case['input']['observations']['activation_diagnostics'][0]['measurements']['norm'],value)
            self.assertEqual(case['observation_source']['campaign'],str(campaign))
        store=Store(campaign/'memory')
        try:store.event('later_annotation',{'note':'outside selected prefix'})
        finally:store.close()
        audit_inputs(suite)
        with patch('rec_harness.tool_shadow.run_cpu_job',side_effect=self.execute) as worker:
            result=run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json')
            self.assertTrue(result['passed']);self.assertEqual(result['total_cost'],4)
            self.assertEqual(run_validation(self.root/'suite/suite.json',self.root/'candidate',self.root/'result.json'),result)
            self.assertEqual(worker.call_count,4)

    def test_wrong_model_cutoff_or_missing_source_rejected_before_suite_creation(self):
        self.source();original=read_json(self.root/'spec.json')
        for change in ['wrong_model','early_cutoff','missing','old_contract']:
            with self.subTest(change=change):
                spec=copy.deepcopy(original);source=spec['conditions'][0]['observation_source']
                if change=='wrong_model':source['observation_ids']=['activation-B']
                elif change=='early_cutoff':source['cutoff_seq']=1
                elif change=='missing':spec['conditions'][0].pop('observation_source')
                else:spec.pop('tool_contract')
                write_json(self.root/'spec.json',spec)
                with self.assertRaises(ContractError):freeze_suite(self.root/'spec.json',self.root/'suite')
                self.assertFalse((self.root/'suite').exists())

    def test_rehashed_suite_cannot_change_observations_or_prefix_provenance(self):
        self.source();original=freeze_suite(self.root/'spec.json',self.root/'suite')
        for change in ['value','head','task','omit']:
            with self.subTest(change=change):
                suite=copy.deepcopy(original);case=suite['cases'][0]
                if change=='value':case['input']['observations']['activation_diagnostics'][0]['measurements']['norm']=0.99
                elif change=='head':case['observation_source']['event_head']='0'*64
                elif change=='task':case['observation_source']['task_id']='other'
                else:case['input']['observations']['activation_diagnostics']=[]
                suite['hash']=digest({k:v for k,v in suite.items() if k!='hash'})
                with self.assertRaisesRegex(ContractError,'prefix changed'):audit_inputs(suite)

    def test_source_summary_corruption_and_lock_changes_rejected(self):
        campaign=self.source();suite=freeze_suite(self.root/'spec.json',self.root/'suite')
        config=read_json(campaign/'amazon-auto.json');write_json(campaign/'amazon-auto.json',{**config,'task_id':'changed'})
        with self.assertRaisesRegex(ContractError,'campaign changed'):audit_inputs(suite)
        write_json(campaign/'amazon-auto.json',config)
        ref=suite['cases'][0]['observation_source']['receipts']['activation-A']['summary_ref']
        Artifacts(campaign/'artifacts/objects').verify(ref).write_text('{}')
        with self.assertRaises(ContractError):audit_inputs(suite)

    def test_rollback_cannot_reveal_later_observations_in_earlier_prefix(self):
        campaign=self.source();store=Store(campaign/'memory')
        try:store.rollback(1)
        finally:store.close()
        with self.assertRaisesRegex(ContractError,'not committed'):freeze_suite(self.root/'spec.json',self.root/'suite')

    def test_preflight_requires_plan_and_shadow_to_share_input_contract(self):
        from rec_harness.validation_preflight import preflight
        self.source();suite=freeze_suite(self.root/'spec.json',self.root/'suite')
        plan={'identity':suite['identity'],'tool_contract':OBSERVATION_CONTRACT,'stages':{'shadow':{'command':[
            '{python}','-m','rec_harness.tool_shadow','run','--suite',str(self.root/'suite/suite.json')]}}}
        self.assertEqual(preflight(plan)['status'],'passed')
        plan['tool_contract']='public-aggregate-measurement-v1'
        with self.assertRaisesRegex(ContractError,'input contract'):preflight(plan)
