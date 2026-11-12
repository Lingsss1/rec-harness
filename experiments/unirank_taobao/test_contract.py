"""Meaningful DIN and host information-boundary checks, without research calls."""
from collections import OrderedDict
import json
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(PROJECT/'UniRank'))

import numpy as np
import pandas as pd
import torch
from unirank.features import FeatureMap
from unirank.pytorch.dataloaders.unirank_dataloader import BlockedBatchCollator
from baseline import Model
from train import without_labels,check_source,predict
from visibility import mature_actions,visible_prefix_length,MaturedCollator


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.labels=['is_click','cart','fav','buy']
        self.fmap=FeatureMap('fixture',str(self.root));self.fmap.labels=self.labels
        self.fmap.features=OrderedDict([
            ('user_id',{'type':'categorical','vocab_size':8}),
            ('item_id',{'type':'categorical','vocab_size':8,'source':'item'}),
            ('cate_id',{'type':'categorical','vocab_size':5,'source':'item'}),
            ('action',{'type':'categorical','vocab_size':17,'source':'action'})])
        self.fmap.num_fields=4;self.fmap.default_emb_dim=4
    def tearDown(self):self.temp.cleanup()

    def model(self):
        return Model(self.fmap,gpu=-1,embedding_dim=4,attention_hidden_units=[8,4],tower_hidden_units=[8,4],
                     verbose=0,metrics=['AUC'],model_root=str(self.root))

    def batch(self):
        b={'user_id':torch.tensor([1,2]),**{label:torch.tensor([0.,1.]) for label in self.labels}}
        items={'item_id':torch.tensor([[1,2,3],[2,0,4]]),
               'cate_id':torch.tensor([[1,2,3],[2,0,3]]),'action':torch.tensor([[9,6,0],[6,0,0]])}
        mask=torch.tensor([[1.,1.],[1.,0.]])
        return b,items,mask,[mask for _ in self.labels]

    def test_forward_backward_and_no_target_labels_in_model_inputs(self):
        model=self.model();raw=self.batch();features=without_labels(raw,self.labels,torch.device('cpu'))
        self.assertTrue(set(self.labels).isdisjoint(features[0]))
        values=predict(model,features,self.labels)
        loss=torch.nn.functional.binary_cross_entropy(values,torch.tensor([[0.,0.,0.,0.],[1.,1.,1.,1.]]))
        loss.backward()
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        check_source(Path(__file__).with_name('baseline.py'))

    def test_masked_history_changes_do_not_change_eval_predictions(self):
        model=self.model().eval();raw=self.batch();first=without_labels(raw,self.labels,torch.device('cpu'))
        altered=self.batch();altered[1]['item_id'][1,1]=7;altered[1]['cate_id'][1,1]=4;altered[1]['action'][1,1]=16
        second=without_labels(altered,self.labels,torch.device('cpu'))
        with torch.no_grad():torch.testing.assert_close(predict(model,first,self.labels),predict(model,second,self.labels))

    def test_same_time_events_and_unmatured_outcomes_are_excluded(self):
        self.assertEqual(visible_prefix_length([100,110,110,90000],2),1)
        click=np.array([0,6,6,6,6,6,6,6,6,9,9,9,9,9,9,9,9])
        result=mature_actions(np.array([11,11,11]),np.array([10,86400,86401]),click)
        np.testing.assert_array_equal(result,[9,9,11])

    def test_official_and_strict_collators_expose_the_documented_difference(self):
        folder=self.root/'train';(folder/'user_info').mkdir(parents=True);(folder/'item_info').mkdir()
        vocabulary={'exposure':6,'is_click':9,'is_click|cart':11,'cart':2,'fav':7,'buy':1}
        (self.root/'meta_data.json').write_text(json.dumps({'action_vocab':vocabulary,
            'behavior_log_usage':{'behavior_label_window_seconds':86400}}))
        user=folder/'user_info/part-00000.parquet';item=folder/'item_info/part-00000.parquet'
        pd.DataFrame({'user_index':[0],'full_item_seq':[[1,2,3,4]],'full_action_seq':[[11,6,6,9]],
                      'full_timestamp_seq':[[100,110,110,90000]]}).to_parquet(user,index=False)
        pd.DataFrame({'item_index':[0,1,2,3,4],'item_id':[0,1,2,3,4],'cate_id':[0,1,1,2,3]}).to_parquet(item,index=False)
        columns={name:index for index,name in enumerate(['user_index','item_index','seq_len','user_id']+self.labels)}
        common=dict(feature_map=self.fmap,max_len=3,column_index=columns,user_info=str(folder/'user_info'),item_info=str(folder/'item_info'))
        # The target is the second exposure tied at t=110; the official prefix has two events.
        payload={'rows':np.array([[0,3,2,1,0,0,0,0]],dtype=np.float64),'user_info_file':str(user),'item_info_file':str(item)}
        # Upstream collators only emit columns included in the feature map.
        for name in ['user_index','item_index','seq_len']:self.fmap.features[name]={'type':'meta'}
        official=BlockedBatchCollator(**common)(payload);strict=MaturedCollator(**common)(payload)
        self.assertEqual(int(official[2].sum()),2);self.assertEqual(int(strict[2].sum()),1)
        self.assertIn(11,official[1]['action']);self.assertNotIn(11,strict[1]['action'])
        self.assertIn(9,strict[1]['action'])


if __name__=='__main__':unittest.main()
