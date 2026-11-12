import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.activation_histories import resolve_histories
from rec_harness.core import ContractError,file_hash


class HistorySamplingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.path=self.root/'A_train.txt'
        self.path.write_text(''.join(f'{user} {item}\n' for user,length in [(901,3),(902,10)] for item in range(1,length+1)))
        self.study={'data':str(self.root),'data_hashes':{'A_train.txt':file_hash(self.path)},
            'datasets':['A'],'development_seeds':[42],'confirmation_seeds':[17]}
        self.config={'history_sampling':{'sample_size':2,'seed':13},'settings':{'MAXLEN':16},
            'checkpoint_source':{'study':str(self.root),'dataset':'A','seed':42}}

    def resolve(self):
        with patch('rec_harness.activation_histories.load_config',return_value=self.study):
            return resolve_histories(self.config)

    def test_deterministic_right_alignment_counts_and_no_user_ids(self):
        first=self.resolve();self.assertEqual(first,self.resolve())
        self.assertEqual(first['slice_counts'],{'all':2,'short':1,'long':1})
        self.assertEqual(first['valid_positions'],13)
        self.assertNotIn('901',str(first));self.assertNotIn('902',str(first))
        self.assertEqual(sorted(row[-1] for row in first['payload']['sequences']),[3,10])
        self.config['settings']['MAXLEN']=8
        self.assertEqual(self.resolve()['slice_counts'],{'all':2,'short':2})

    def test_changed_data_confirmation_and_oversized_sample_rejected(self):
        self.config['checkpoint_source']['seed']=17
        with self.assertRaisesRegex(ContractError,'development'):self.resolve()
        self.config['checkpoint_source']['seed']=42
        self.config['history_sampling']['sample_size']=129
        with self.assertRaises(ContractError):self.resolve()
        self.config['history_sampling']['sample_size']=2
        self.path.write_text('901 5\n')
        with self.assertRaisesRegex(ContractError,'changed'):self.resolve()

    def test_sampling_requires_checkpoint_and_sufficient_histories(self):
        self.config['history_sampling']['sample_size']=3
        with self.assertRaisesRegex(ContractError,'insufficient'):self.resolve()
        del self.config['checkpoint_source']
        with self.assertRaisesRegex(ContractError,'checkpoint'):self.resolve()
