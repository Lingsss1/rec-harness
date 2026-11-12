from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
from rec_harness.core import ContractError,write_json


class AutonomousRescoreRecoveryTests(unittest.TestCase):
    def test_partial_and_completed_planned_rescores_use_verified_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'round'
            controller=AmazonAutoresearch.__new__(AmazonAutoresearch)
            controller.study=root/'study';controller.config={'rescore_recipe':{},'uid_registry':root/'uids'}
            directory=folder/'independent-rescores/model-tiny-42'
            write_json(directory/'rescore-plan.json',{'frozen':'plan'})
            write_json(directory/'inference/receipt.json',{'status':'fixture'})
            result={'rank_disagreements':0,'positive_tied_users':0}
            with patch('rec_harness.adapters.amazon_autonomous.load_config',return_value={'datasets':['tiny']}), \
                 patch('rec_harness.adapters.amazon_autonomous.rescore_job',return_value=result) as score:
                self.assertEqual(controller._rescore(folder,'model',[42]),[result])
                self.assertTrue(score.call_args.kwargs['resume'])
                self.assertEqual(list(directory.parent.iterdir()),[directory])
                write_json(directory/'acceptance.json',{'unverified':'local projection'})
                score.side_effect=ContractError('cached rescore acceptance changed')
                with self.assertRaisesRegex(ContractError,'acceptance changed'):
                    controller._rescore(folder,'model',[42])
                self.assertEqual(score.call_count,2)

    def test_unplanned_interruption_is_not_renamed_and_reexecuted(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);controller=AmazonAutoresearch.__new__(AmazonAutoresearch)
            controller.study=root/'study';controller.config={'rescore_recipe':{},'uid_registry':root/'uids'}
            directory=root/'round/independent-rescores/model-tiny-42'
            write_json(directory/'inference/receipt.json',{'legacy':'interrupted'})
            with patch('rec_harness.adapters.amazon_autonomous.load_config',return_value={'datasets':['tiny']}), \
                 patch('rec_harness.adapters.amazon_autonomous.rescore_job',side_effect=ContractError('legacy rescore requires original recovery')) as score:
                with self.assertRaisesRegex(ContractError,'original recovery'):
                    controller._rescore(root/'round','model',[42])
                self.assertTrue(score.call_args.kwargs['resume'])
                self.assertEqual(list(directory.parent.iterdir()),[directory])
