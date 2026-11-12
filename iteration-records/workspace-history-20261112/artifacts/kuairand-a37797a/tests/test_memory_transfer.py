import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import ContractError,read_json,write_json
from rec_harness.online_validation import freeze_suite,load_suite,episode
from rec_harness.memory_transfer import freeze,load_study,run_validation
from rec_harness.runtime import Research
import test_online_validation
from test_online_validation import PairedAgent


class TransferAgent(PairedAgent):
    def run(self,role,context,directory):
        from rec_harness.memory_policy import LIST_FIELDS
        result=super().run(role,context,directory)
        if role=='hypothesis':
            for card in [result['method'],result['proposal']]:
                for key in LIST_FIELDS & card.keys():
                    if not isinstance(card[key],list):card[key]=[]
            write_json(Path(directory)/'response.json',result)
        return result


class MemoryTransferTests(unittest.TestCase):
    setUp=test_online_validation.OnlineValidationTests.setUp
    evaluate=test_online_validation.OnlineValidationTests.evaluate

    def prepare_source(self):
        freeze_suite(self.root/'spec.json',self.root/'source-suite')
        suite,inputs=load_suite(self.root/'source-suite/suite.json')
        with patch('rec_harness.toy_isolated.evaluate',side_effect=self.evaluate):
            episode(suite['cases'][0],{},self.root/'source',inputs,agent=TransferAgent())

    def prepare_targets(self):
        for stage,seed in [('development',901),('confirmation',902)]:
            config=read_json(self.root/(stage+'.json'))
            config['task_id']='target-'+stage;config['protocol']['seeds']=[seed]
            write_json(self.root/(stage+'.json'),config)
        freeze_suite(self.root/'spec.json',self.root/'target-suite')

    def test_frozen_source_exposure_independent_overlays_cost_and_resume(self):
        self.prepare_source();self.prepare_targets()
        frozen=freeze(self.root/'source',self.root/'target-suite/suite.json',self.root/'study')
        self.assertEqual(frozen['source_cost'],8);self.assertFalse(frozen['source_codex_requests'])
        args=(self.root/'study/study.json',self.root/'result.json','development')
        with patch('rec_harness.toy_isolated.evaluate',side_effect=self.evaluate):
            report=run_validation(*args,agent_factory=lambda *_:TransferAgent())
            self.assertFalse(report['online']);self.assertEqual(report['gain'],0)
            self.assertEqual(report['source_cost'],8);self.assertEqual(report['target_cost'],16)
            self.assertEqual(report['total_attributable_cost'],24)
            row=report['rows'][0]
            self.assertEqual(row['fresh']['visible_source_nodes'],[])
            self.assertTrue(row['experienced']['visible_source_nodes'])
            self.assertNotEqual(row['fresh']['graph_directory'],row['experienced']['graph_directory'])
            class NoCalls:
                def run(self,*args):raise AssertionError('unexpected role call')
            self.assertEqual(run_validation(*args,agent_factory=lambda *_:NoCalls()),report)
        self.assertEqual(load_study(self.root/'study/study.json')[0],frozen)
        object_path=self.root/'study/objects'/frozen['source_memory']['sha256'];object_path.write_text('{}')
        with self.assertRaises(ContractError):load_study(self.root/'study/study.json')

    def test_overlap_agent_change_and_unsettled_source_fail_before_target_runs(self):
        self.prepare_source()
        with self.assertRaisesRegex(ContractError,'reuses source'):
            freeze(self.root/'source',self.root/'source-suite/suite.json',self.root/'overlap')
        self.prepare_targets()
        research=Research(self.root/'source',agent=object())
        try:research.store.reserve('pending',1)
        finally:research.store.close()
        with self.assertRaisesRegex(ContractError,'unsettled'):
            freeze(self.root/'source',self.root/'target-suite/suite.json',self.root/'pending-study')
        research=Research(self.root/'source',agent=object())
        try:research.store.settle('pending',0,'cancelled')
        finally:research.store.close()
        cfg=read_json(self.root/'development.json');cfg['codex']['model']='different-model'
        write_json(self.root/'development.json',cfg)
        freeze_suite(self.root/'spec.json',self.root/'changed-suite')
        with self.assertRaisesRegex(ContractError,'base agent differ'):
            freeze(self.root/'source',self.root/'changed-suite/suite.json',self.root/'changed-study')
