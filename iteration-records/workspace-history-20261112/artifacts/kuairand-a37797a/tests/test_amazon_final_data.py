import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from rec_harness.core import ContractError,file_hash
from rec_harness.adapters.amazon_final_data import dataset_rows,prepare_final_data,read_interactions


class FinalDataTests(unittest.TestCase):
    def test_unknown_items_are_rejected_before_sampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.txt';path.write_text('1 201\n')
            with self.assertRaisesRegex(ContractError,'vocabulary'):read_interactions(path,200)
            path.write_text('1 4 extra\n')
            with self.assertRaisesRegex(ContractError,'invalid'):read_interactions(path,200)

    def test_shared_sampling_and_history(self):
        train={i:[1,2] for i in range(1,21)};valid={i:[3] for i in train};test={i:[4,5] for i in train}
        rows=dataset_rows(train,valid,test,itemnum=200,count=7,seed=19)
        self.assertEqual(rows,dataset_rows(train,valid,test,itemnum=200,count=7,seed=19))
        self.assertEqual(len(rows),7)
        for row in rows:
            self.assertEqual(row['history'],[1,2,3]);self.assertEqual(row['candidates'][0],4)
            self.assertEqual(len(set(row['candidates'])),100)
            self.assertFalse(set(row['candidates'])&{1,2,3})
        # Even an invalid user outside the sampled population must not be dropped.
        test[999]=[4]
        with self.assertRaisesRegex(ContractError,'lacks'):dataset_rows(train,valid,test,itemnum=200,count=1,seed=19)
        with self.assertRaisesRegex(ContractError,'insufficient'):
            dataset_rows({1:[1,2]},{1:[3]},{1:[4]},itemnum=100,count=1,seed=19)

    def test_private_request_resume_and_data_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);study=root/'study';study.mkdir();output=root/'final';output.mkdir()
            for split,items in [('train',[1,2]),('valid',[3]),('test',[4])]:
                (root/f'tiny_{split}.txt').write_text(''.join(f'1 {item}\n' for item in items))
            plan={'hash':'sealed','request':{'study':str(study),'test_directory':str(root),'eval_seed':9,'eval_users':1000},
                  'datasets':['tiny'],'jobs':[{'dataset':'tiny','itemnum':200}], 'data':str(root),
                  'data_hashes':{f'tiny_{s}.txt':file_hash(root/f'tiny_{s}.txt') for s in ['train','valid']}}
            with patch('rec_harness.adapters.amazon_final_data.load_selection',return_value=plan):
                record=prepare_final_data(output)
                self.assertEqual(prepare_final_data(output),record)
                row=record['datasets']['tiny'];self.assertEqual(row['users'],1)
                self.assertNotIn('user_id',row['request']['rows'][0]);self.assertNotIn('positive_index',row['request']['rows'][0])
                self.assertEqual((output/'private-evaluation.json').stat().st_mode&0o777,0o600)
                (root/'tiny_test.txt').write_text('1 5\n')
                with self.assertRaisesRegex(ContractError,'data changed'):prepare_final_data(output)

    def test_unsealed_selection_cannot_open_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch('rec_harness.adapters.amazon_final_data.load_selection',side_effect=ContractError('not sealed')):
                with patch('rec_harness.adapters.amazon_final_data.read_interactions',side_effect=AssertionError('test opened')):
                    with self.assertRaisesRegex(ContractError,'not sealed'):prepare_final_data(tmp)
