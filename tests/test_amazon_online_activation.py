import unittest
from pathlib import Path
from unittest.mock import patch
import test_amazon_tool_online
from rec_harness.amazon_tool_online import freeze_suite,load_suite,create_study
from rec_harness.amazon_tool_cost import ledger_axis
from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
from rec_harness.adapters.amazon_activation import audit_costs
from rec_harness.core import Artifacts,ContractError,digest,read_json,write_json
from rec_harness.store import Store


class AmazonOnlineActivationTests(unittest.TestCase):
    setUp=test_amazon_tool_online.AmazonToolOnlineTests.setUp
    freeze=test_amazon_tool_online.AmazonToolOnlineTests.freeze

    def enable(self):
        options={'settings':{'HIDDEN_UNITS':8,'NUM_BLOCKS':1,'KERNEL_SIZE':3,'DROPOUT_RATE':0,'MAXLEN':16},
                 'execution':{'recipe':{'runtime_dirs':['/usr'],'runtime_files':['/bin/true']},
                              'python':'/usr/bin/python3','uid_registry':str(self.root/'probe-uids')},
                 'budget':2,'cost':1,'timeout':20}
        for path in self.configs:
            config=read_json(path);config['activation_diagnostic']=options;write_json(path,config)
        self.spec['cost_weights']['activation_units']=3
        return options

    def test_activation_requires_declared_weight_and_development_only_seeds(self):
        self.enable();self.spec['cost_weights'].pop('activation_units')
        with self.assertRaisesRegex(ContractError,'activation cost weight'):self.freeze()
        self.spec['cost_weights']['activation_units']=3
        config=read_json(self.configs[0]);config['activation_diagnostic']['seeds']=[101];write_json(self.configs[0],config)
        write_json(self.root/'spec.json',self.spec)
        with self.assertRaisesRegex(ContractError,'development seeds only'):
            freeze_suite(self.root/'spec.json',self.root/'other-suite')

    def setup_probe(self):
        options=self.enable();suite,objects=load_suite(self.freeze());case=suite['cases'][0]
        root=self.root/'group';execution=create_study(case,root/'study',objects)
        config={**case['controller'],'task_id':'activation','study':str(root/'study'),'execution_id':execution}
        AmazonAutoresearch.init(root/'campaign',config)
        cfg=read_json(root/'study/study.json')
        probe={**{k:options[k] for k in ['settings','execution','cost','timeout']},'budget':options['cost'],
               'source':cfg['model_versions']['base'],
               'checkpoint_source':{'study':str(root/'study'),'model_version':'base','dataset':'A','seed':100}}
        folder=root/'campaign/activation-probes'/digest(probe)
        write_json(folder/'config.json',probe);binding={'config_hash':digest(probe),'checkpoint_hash':'checkpoint'}
        write_json(folder/'binding.json',binding)
        measurement={'measurements':{'block_0.activation_rms':2},'limitations':['fixture']}
        write_json(folder/'measurement.json',measurement);objects=Artifacts(folder/'objects')
        store=Store(folder/'memory')
        try:
            store.configure_budget(1);store.reserve('activation-probe',1);store.settle('activation-probe',1,'success')
            result={'status':'success','binding':binding,'measurement':measurement,
                    'artifacts':{'measurement.json':objects.put(folder/'measurement.json')}}
            store.event('activation_probe_completed',result);head=store.verify()
        finally:store.close()
        write_json(folder/'result.json',result)
        return root,options,folder,head

    def test_activation_ledger_and_receipts_reconciled_without_new_probe(self):
        root,options,folder,head=self.setup_probe()
        with patch('rec_harness.activation_checkpoint.resolve_checkpoint',return_value={'sha256':'checkpoint'}), \
             patch('rec_harness.adapters.amazon_activation.run_probe',side_effect=AssertionError('must not execute')):
            result=audit_costs(root/'campaign',options)
            self.assertEqual(result['charged_or_reserved'],1);self.assertEqual(result['pending'],0)
            self.assertEqual(result['ledgers'][0]['cost']['integrity'],head)
            self.assertEqual(audit_costs(root/'campaign',options),result)
            self.assertEqual(ledger_axis(folder/'memory'),('activation_units',1))
            changed=read_json(folder/'result.json');changed['measurement']['measurements']['block_0.activation_rms']=999
            write_json(folder/'result.json',changed)
            with self.assertRaisesRegex(ContractError,'committed result'):audit_costs(root/'campaign',options)

    def test_changed_probe_config_and_checkpoint_are_rejected(self):
        root,options,folder,_=self.setup_probe()
        with patch('rec_harness.activation_checkpoint.resolve_checkpoint',return_value={'sha256':'other'}):
            with self.assertRaisesRegex(ContractError,'checkpoint binding'):audit_costs(root/'campaign',options)
        config=read_json(folder/'config.json');config['cost']=0;write_json(folder/'config.json',config)
        with self.assertRaisesRegex(ContractError,'frozen options'):audit_costs(root/'campaign',options)

    def test_online_report_prices_activation_axis_for_all_groups(self):
        from rec_harness.amazon_tool_online import run_validation
        self.enable();path=self.freeze()
        def run(case,root,objects,tool,**kwargs):
            return {'score':0.5,'protocol_hash':'p','evaluation_conditions':['same'],
                'diagnostic_calls':int(tool is not None),'diagnostic_failures':0,
                'costs':{'role_calls':{'charged_or_reserved':6},'gpu_seconds':{'charged_or_reserved':10},
                         'diagnostic_units':{'charged_or_reserved':int(tool is not None)},
                         'activation_units':{'charged_or_reserved':1,'pending':0,'ledgers':[]}}}
        with patch('rec_harness.amazon_tool_online.episode',side_effect=run):
            report=run_validation(path,self.root/'tool',self.root/'online/result.json','utility',agent_factory=lambda *_:object())
        self.assertEqual(report['cost_axes']['activation_units'],3)
        self.assertAlmostEqual(report['total_cost'],29.3)

    def test_terminal_online_readout_includes_audited_activation_costs(self):
        from rec_harness.amazon_tool_online import read_result
        root,options,folder,_=self.setup_probe()
        state={'round':1,'incumbent':'base','status':'ready'}
        store=Store(root/'campaign/memory')
        try:
            store.event('agent_submitted',{'task_id':'fixture','role':'evidence'})
            store.reserve('fixture',1);store.settle('fixture',1,'success')
            store.event('auto_round_completed',{'state':state,'result':{'selected':False}})
        finally:store.close()
        write_json(root/'campaign/run-report.json',{'state':{**state,'status':'stopped','stop_reason':'max_rounds'},'failed_rounds':0})
        with patch.object(AmazonAutoresearch,'verify',return_value={'fixture':'audit'}),              patch('rec_harness.amazon_tool_online.progress_report',return_value={'rows':[{'development_score':0.5}]}),              patch('rec_harness.amazon_tool_online.audit_job',return_value=({'metrics':{'HR@10':0.5}},{'candidates_hash':'fixed','data_hashes':{'train':'t','validation':'v'}})),              patch('rec_harness.activation_checkpoint.resolve_checkpoint',return_value={'sha256':'checkpoint'}):
            result=read_result(root)
        self.assertEqual(result['costs']['activation_units']['charged_or_reserved'],1)
        self.assertEqual(result['costs']['activation_units']['ledgers'][0]['directory'],str(folder/'memory'))

    def test_insufficient_probe_envelope_rejected_before_training(self):
        self.enable()
        config=read_json(self.configs[0]);config['max_rounds']=3;write_json(self.configs[0],config)
        with self.assertRaisesRegex(ContractError,'cannot cover frozen research rounds'):self.freeze()
