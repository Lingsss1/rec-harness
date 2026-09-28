import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rec_harness.core import ContractError,digest,read_json,write_json
from rec_harness.store import Store
from rec_harness.adapters.amazon_progress import progress_report,round_costs


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.study=self.root/'study';self.campaign=self.root/'campaign'
        self.controller=Store(self.campaign/'memory');self.ledger=Store(self.study/'memory')
        self.addCleanup(self.controller.close);self.addCleanup(self.ledger.close)
        self.controller.configure_budget(20);self.ledger.configure_budget(100)
        cfg={'study':str(self.study)}
        write_json(self.campaign/'amazon-auto.json',cfg);write_json(self.campaign/'auto-lock.json',{'config_hash':digest(cfg)})
        study={'datasets':['a','b'],'development_seeds':[42],'protocol_hash':'p','protocol':{'metric':'HR@10'}}
        write_json(self.study/'study.json',study);write_json(self.study/'study-lock.json',{'config_hash':digest(study)})

    def complete(self,index,selected=False,legacy=False,broken_cost=False):
        self.controller.reserve('call-'+str(index),1);self.controller.settle('call-'+str(index),1,'success')
        self.ledger.reserve('job-'+str(index),10);self.ledger.settle('job-'+str(index),4,'success')
        report={'protocol_hash':'p','rows':[{'model_version':version,'dataset':dataset,'seed':42,'metrics':{'HR@10':value}}
            for version,value in [('base',.5),('candidate',.9)] for dataset in ['a','b']]}
        report['hash']=digest(report)
        path=self.campaign/'rounds'/('round-'+str(index))/'development-readout.json';write_json(path,report)
        self.ledger.event('amazon_research_cycle_completed',{'identity':'cycle-'+str(index),'readout_hash':report['hash']})
        result={'candidate':'candidate','selected':selected,'cycle_identity':'cycle-'+str(index),'development_hash':report['hash']}
        payload={'state':{'round':index,'incumbent':'candidate' if selected else 'base'},'result':result}
        if not legacy:
            payload['cost_prefixes']=round_costs(SimpleNamespace(store=self.controller,study=self.study))
            if broken_cost:payload['cost_prefixes']['study_gpu_seconds']['charged_or_reserved']+=1
        self.controller.event('auto_round_completed',payload)
        return path

    def test_selected_model_curve_retains_exact_cost_prefix_and_legacy_unknown(self):
        self.complete(1);self.complete(2,selected=True);self.complete(3,legacy=True)
        self.controller.reserve('later-call',1);self.ledger.reserve('later-job',9)
        before=(self.controller.verify(),self.ledger.verify())
        with patch.object(Store,'export_events',side_effect=AssertionError('readonly query')):
            report=progress_report(self.campaign)
            self.assertEqual(report,progress_report(self.campaign))
        rows=report['rows']
        self.assertEqual([row['development_score'] for row in rows],[.5,.9,.5])
        self.assertEqual([row['costs']['role_calls_charged_or_reserved'] for row in rows],[1,2,3])
        self.assertEqual([row['costs']['study_gpu_seconds_charged_or_reserved'] for row in rows],[4,8,None])
        self.assertEqual((self.controller.verify(),self.ledger.verify()),before)
        self.assertEqual(report['partition'],'development')

    def test_changed_public_readout_and_forged_cost_prefix_rejected(self):
        path=self.complete(1);report=read_json(path)
        report['rows'][0]['metrics']['HR@10']=1
        report['hash']=digest({k:v for k,v in report.items() if k!='hash'});write_json(path,report)
        with self.assertRaisesRegex(ContractError,'readout differs'):progress_report(self.campaign)

    def test_forged_study_cost_rejected(self):
        self.complete(1,broken_cost=True)
        with self.assertRaisesRegex(ContractError,'prefix amount changed'):progress_report(self.campaign)

    def test_failed_round_is_retained_without_fabricating_score(self):
        self.controller.reserve('failed-call',1);self.controller.settle('failed-call',1,'crash')
        self.controller.event('auto_round_completed',{'state':{'round':1,'incumbent':'base'},
            'result':{'status':'failed','failure_kind':'role_transport'}})
        row=progress_report(self.campaign)['rows'][0]
        self.assertIsNone(row['development_score']);self.assertEqual(row['status'],'failed')
        self.assertEqual(row['costs']['role_calls_charged_or_reserved'],1)
