"""Causal data and real model/scorer acceptance on explicitly synthetic fixtures."""
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from baseline import Model
from data import prepare,Data,PROFILE
from train import check_source


def make_raw(root,extra_future=False):
    root.mkdir()
    def save(name,header,rows):
        with (root/(name+'.csv')).open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(header);writer.writerows(rows)
    start=1494000000  # 2017-05-06 00:00 Asia/Shanghai
    samples=[];behaviors=[]
    for day in range(8):
        for user in [1,2,3,4]:
            for number in range(4):
                timestamp=start+day*86400+100+number*100
                label=(user+number)%2
                samples.append([user,timestamp,1+number%2,'430548_1007',1-label,label])
                behaviors += [[user,timestamp-1,['pv','cart','fav','buy'][number],10+number%2,100+number%2],
                              [user,timestamp,'pv',99,999]]
    if extra_future:behaviors.append([1,start+7*86400+500,'buy',123,456])
    save('raw_sample',['user','time_stamp','adgroup_id','pid','nonclk','clk'],samples)
    save('ad_feature',['adgroup_id','cate_id','campaign_id','customer','brand','price'],[[1,10,1,1,100,50],[2,11,2,2,101,100]])
    save('user_profile',['userid',*PROFILE],[[user,*([1]*len(PROFILE))] for user in [1,2,3]]) # missing profile retained
    save('behavior_log',['user','time_stamp','btag','cate','brand'],behaviors)


class Acceptance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        make_raw(cls.root/'raw')
        prepare(cls.root/'raw',cls.root/'prepared',fixture=True)
        cls.data=Data(cls.root/'prepared',8)

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_population_split_and_label_exclusion(self):
        self.assertEqual([len(self.data.indices(s)) for s in ['train','validation','test']],[96,16,16])
        self.assertNotIn('clk',self.data.meta['fields']);self.assertNotIn('nonclk',self.data.meta['fields'])
        self.assertEqual(len(self.data.static),128)

    def test_strict_history_and_same_timestamp_ties(self):
        for i in range(self.data.meta['rows']):
            start,end=int(self.data.history_start[i]),int(self.data.history_end[i])
            if end>start:self.assertTrue(np.all(self.data.history[start:end,3]<self.data.times[i]))

    def test_future_mutation_cannot_change_past_features(self):
        make_raw(self.root/'future_raw',extra_future=True)
        prepare(self.root/'future_raw',self.root/'future_prepared',fixture=True)
        future=Data(self.root/'future_prepared',8)
        indices=self.data.indices('train')[:30]
        a=self.data.batch(indices,torch.device('cpu'));b=future.batch(indices,torch.device('cpu'))
        for field in a:self.assertTrue(torch.equal(a[field],b[field]),field)

    def test_all_reference_models_handle_missing_profiles_and_empty_masks(self):
        batch=self.data.batch(np.arange(8),torch.device('cpu'))
        batch['history_mask'][:]=False
        meta={key:self.data.meta[key] for key in ['fields','buckets','category_field_index','brand_field_index','history_columns']}
        for kind in ['static','pooled','din']:
            model=Model(meta,{'kind':kind,'embedding_dim':4,'hidden':[8]})
            values=model(batch)
            self.assertEqual(tuple(values.shape),(8,));self.assertTrue(torch.isfinite(values).all())
            loss=torch.nn.functional.binary_cross_entropy_with_logits(values,torch.zeros(8));loss.backward()

    def test_candidate_io_is_rejected(self):
        path=self.root/'unsafe.py';path.write_text('import os\n')
        with self.assertRaises(ValueError):check_source(path)


if __name__=='__main__':unittest.main()
