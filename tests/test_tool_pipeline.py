import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import Artifacts,ContractError,digest,read_json,write_json
from rec_harness.store import Store
from rec_harness.tool_pipeline import freeze_suite,load_suite,run_validation


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        common={'identity':{'suite':'fixture'},'tool_files':['diagnostic.py']}
        reference={**common,'budget':10,'case_cost':1,'timeout':1,'cases':[
            {'id':stage,'stage':stage,'input':{},'expected':{'x':1},'absolute_tolerance':0}
            for stage in ['correctness','synthetic','robustness']]}
        for kind,body in [('reference',reference),('shadow',{**common,'schema':'amazon-tool-shadow-v1'}),('online',{**common,'schema':'online-tool-suite-v1'})]:
            write_json(self.root/(kind+'.json'),body)
        write_json(self.root/'spec.json',{kind:kind+'.json' for kind in ['reference','shadow','online']})
        freeze_suite(self.root/'spec.json',self.root/'suite')
        self.candidate=self.root/'candidate';self.candidate.mkdir();(self.candidate/'diagnostic.py').write_text('fixture')

    def test_stage_routes_and_freezes_all_suite_inputs(self):
        report={'identity':{'suite':'fixture'}}
        for stage,module in [('correctness','tool_validation'),('synthetic','tool_validation'),('robustness','tool_validation'),('shadow','tool_shadow'),('utility','tool_online')]:
            with patch('rec_harness.'+module+'.run_validation',return_value=report) as run:
                self.assertEqual(run_validation(self.root/'suite/suite.json',stage,self.candidate,self.root/stage/'result.json'),report)
                self.assertEqual(run.call_count,1)
        write_json(self.root/'shadow.json',{})
        with self.assertRaisesRegex(ContractError,'stage suite changed'):load_suite(self.root/'suite/suite.json')

    def test_cost_stage_includes_all_offline_and_abandoned_ledgers(self):
        directory=self.root/'evolution/candidates/candidate/validation'
        snapshot=Artifacts(self.root/'objects').snapshot(self.candidate,['diagnostic.py'])
        report={'identity':{'suite':'fixture'},'total_cost':3}
        for stage in ['correctness','synthetic','robustness','shadow','utility']:
            write_json(directory/stage/'accepted.json',{'report':report,'report_hash':digest(report),'implementation_hash':digest(snapshot)})
        write_json(directory/'utility/result.json',report)
        abandoned=directory.parent/'validation-attempts/utility-1/case/memory';store=Store(abandoned)
        store.configure_budget(2);store.reserve('interrupted-job',1);store.close()
        with patch('rec_harness.tool_cost.run_validation',return_value=report) as cost:
            result=run_validation(self.root/'suite/suite.json','cost_regression',self.candidate,directory/'cost_regression/result.json','prefix.json')
            self.assertEqual(result,report);ledgers=cost.call_args.args[3]
            self.assertEqual(len(ledgers),5);self.assertIn(abandoned,ledgers)
            self.assertEqual(cost.call_args.args[4],'prefix.json')
        store=Store(abandoned);self.addCleanup(store.close)
        self.assertEqual(store.budget()['charged_or_reserved'],1)
        self.assertEqual(store.db.execute('SELECT status FROM budget').fetchone()[0],'interrupted')
        write_json(directory/'utility/result.json',{'changed':True})
        with self.assertRaisesRegex(ContractError,'differs from accepted'):
            run_validation(self.root/'suite/suite.json','cost_regression',self.candidate,directory/'cost_regression/result.json','prefix.json')

    def test_inconsistent_tool_identity_cannot_be_frozen(self):
        shadow=read_json(self.root/'shadow.json');shadow['identity']={'different':'fixture'};write_json(self.root/'shadow.json',shadow)
        with self.assertRaisesRegex(ContractError,'share frozen identity'):
            freeze_suite(self.root/'spec.json',self.root/'bad-suite')

    def test_observation_contract_must_match_all_validation_stages(self):
        from rec_harness.diagnostic_inputs import OBSERVATION_CONTRACT
        shadow=read_json(self.root/'shadow.json');shadow['tool_contract']=OBSERVATION_CONTRACT
        write_json(self.root/'shadow.json',shadow)
        with self.assertRaisesRegex(ContractError,'input contract'):
            freeze_suite(self.root/'spec.json',self.root/'mismatched')
        reference=read_json(self.root/'reference.json');reference['tool_contract']=OBSERVATION_CONTRACT
        write_json(self.root/'reference.json',reference)
        online=read_json(self.root/'online.json');online['tool_runtime']={'contract':OBSERVATION_CONTRACT}
        write_json(self.root/'online.json',online)
        result=freeze_suite(self.root/'spec.json',self.root/'matched')
        self.assertEqual(result['tool_contract'],OBSERVATION_CONTRACT)
