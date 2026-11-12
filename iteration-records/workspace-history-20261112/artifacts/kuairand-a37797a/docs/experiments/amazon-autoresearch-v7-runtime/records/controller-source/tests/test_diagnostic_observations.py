"""Public observation input contract; no model internals or external role calls."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.diagnostic_inputs import CONTRACT,OBSERVATION_CONTRACT,measurement_request
from rec_harness.diagnostics import measure,freeze_validation_tool,runtime_contract
from rec_harness.runtime import node
import test_diagnostics


class ObservationInputTests(unittest.TestCase):
    setUp=test_diagnostics.DiagnosticTests.setUp
    execute=test_diagnostics.DiagnosticTests.execute
    receipt_execute=test_diagnostics.DiagnosticTests.receipt_execute

    def observations(self):
        self.research.config['diagnostic_tool']['contract']=OBSERVATION_CONTRACT
        self.state['baseline']['runs'][0]['dataset']='A'
        row={'observation_id':'activation-test','model_version':'root','dataset':'A','seed':42,
             'measurements':{'norm':2},'limitations':['synthetic histories'],'scope':'development aggregate'}
        self.research.store.patch({'base_version':self.research.store.version,'nodes':[
            node(row['observation_id'],'Observation',{**row,'valid':True},'task','protocol')],'edges':[]})
        return {'activation_diagnostics':[row]}

    def test_verified_inputs_archived_and_cached_without_execution_or_recharge(self):
        observations=self.observations()
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=self.execute) as runner:
            record=measure(self.research,self.state,observations=observations)
            self.assertEqual(measure(self.research,self.state,observations=observations),record)
            self.assertEqual(runner.call_count,1)
        request=read_json(self.research.artifacts.verify(record['request_ref']))
        self.assertEqual(request['observations']['activation_diagnostics'],observations['activation_diagnostics'])
        self.assertEqual(request['observations']['block_score_diagnostics'],[])
        self.assertNotIn('private_path',str(request))
        self.assertEqual(self.research.store.budget()['charged_or_reserved'],1)
        self.research.artifacts.verify(record['request_ref']).write_text('{}')
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=AssertionError('must not execute')):
            with self.assertRaises(ContractError):measure(self.research,self.state,observations=observations)

    def test_uncommitted_stale_private_or_wrong_condition_rejected_before_spending(self):
        original=self.observations()
        invalid=[]
        for key,value in [('model_version','other'),('seed',17),('dataset','B'),
                          ('measurements',{'norm':99}),('checkpoint','/private/weights'),
                          ('observation_id','not-committed'),('scope','altered'),('seed',True)]:
            changed=copy.deepcopy(original);changed['activation_diagnostics'][0][key]=value;invalid.append(changed)
        duplicate=copy.deepcopy(original);duplicate['block_score_diagnostics']=duplicate['activation_diagnostics'];invalid.append(duplicate)
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=AssertionError('must not execute')):
            for observations in invalid:
                with self.subTest(observations=observations):
                    with self.assertRaises(ContractError):measure(self.research,self.state,observations=observations)
        self.assertEqual(self.research.store.budget()['charged_or_reserved'],0)
        self.assertFalse((self.research.root/'research_archive/diagnostics').exists())

    def test_old_contract_remains_exact_and_new_contract_is_explicit(self):
        runs=self.state['baseline']['runs']
        expected={'task_id':'task','model_version':'root','protocol_hash':'protocol',
                  'runs':[{'seed':42,'valid':True,'metrics':{'quality':0.5}}]}
        self.assertEqual(measurement_request('task','root','protocol',runs,CONTRACT,{'activation_diagnostics':['not exposed']}),expected)
        self.assertNotIn('observations',runtime_contract()['request'])
        self.assertIn('observations',runtime_contract(OBSERVATION_CONTRACT)['request'])
        options={'source':str(self.research.root),'files':['diagnostic.py'],'execution':{'recipe':{},'python':'python3','uid_registry':'unused'},
                 'cost':1,'timeout':5,'contract':OBSERVATION_CONTRACT}
        frozen=freeze_validation_tool(options,self.research.artifacts)
        self.assertEqual(frozen['contract'],OBSERVATION_CONTRACT)
        with self.assertRaises(ContractError):freeze_validation_tool({**options,'contract':'unknown'},self.research.artifacts)

    def test_observation_request_recovers_completed_worker_without_second_charge(self):
        observations=self.observations();original=self.research.store.event
        def interrupt(kind,value):
            if kind=='diagnostic_completed':raise KeyboardInterrupt()
            return original(kind,value)
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=self.receipt_execute) as runner:
            with patch.object(self.research.store,'event',side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):measure(self.research,self.state,observations=observations)
            self.assertEqual(runner.call_count,1)
        budget=self.research.store.budget()
        with patch('rec_harness.diagnostics.run_cpu_job',side_effect=AssertionError('must not execute')):
            result=measure(self.research,self.state,observations=observations)
        self.assertEqual(result['status'],'success')
        self.assertEqual(read_json(self.research.artifacts.verify(result['request_ref']))['observations']['activation_diagnostics'],observations['activation_diagnostics'])
        self.assertEqual(self.research.store.budget(),budget)

    def test_failed_cross_protocol_or_nonlocal_observations_are_not_tool_inputs(self):
        original=self.observations()['activation_diagnostics'][0]
        for index,(key,value) in enumerate([('valid',False),('origin','agent_hypothesis'),('task_id','other'),('protocol_hash','other')]):
            row={**original,'observation_id':'bad-'+str(index)}
            entry=node(row['observation_id'],'Observation',{**row,'valid':True},'task','protocol')
            entry['attrs'][key]=value
            self.research.store.patch({'base_version':self.research.store.version,'nodes':[entry],'edges':[]})
            with self.assertRaises(ContractError):measure(self.research,self.state,observations={'activation_diagnostics':[row]})
        self.assertEqual(self.research.store.budget()['charged_or_reserved'],0)
