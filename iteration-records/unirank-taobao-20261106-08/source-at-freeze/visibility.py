"""Optional conservative input policy, kept separate from official offline inputs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from unirank.pytorch.dataloaders import unirank_dataloader as upstream
from unirank.utils import find_meta_data_json

OriginalCollator=upstream.BlockedBatchCollator


def visible_prefix_length(timestamps,target_index):
    """Exclude the target, future events and every event tied at target time."""
    timestamps=np.asarray(timestamps,dtype=np.int64)
    return int(np.searchsorted(timestamps,timestamps[int(target_index)],side='left'))


def mature_actions(actions,ages,click_only,window=86400):
    """Until the complete outcome window is strictly past, retain only click."""
    actions=np.asarray(actions,dtype=np.int64)
    return np.where(np.asarray(ages)>window,actions,click_only[actions])


class MaturedCollator(OriginalCollator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        meta=json.loads(Path(find_meta_data_json(kwargs['user_info'])).read_text())
        vocabulary=meta['action_vocab'];self.window=int(meta['behavior_log_usage']['behavior_label_window_seconds'])
        exposure=int(vocabulary['exposure']);click=int(vocabulary.get('click',vocabulary.get('is_click',0)))
        if not click:raise ValueError('No click-only action token')
        self.click_only=np.full(len(self.action_task_table),exposure,dtype=np.int64);self.click_only[0]=0
        for name,code in vocabulary.items():
            if 'click' in name.split('|') or 'is_click' in name.split('|'):self.click_only[int(code)]=click

    def _load_block_side_info(self,user_info_file,item_info_file):
        side=super()._load_block_side_info(user_info_file,item_info_file)
        if 'timestamps' not in side:
            times=pd.read_parquet(user_info_file,columns=['user_index','full_timestamp_seq']).set_index('user_index').sort_index()
            side['timestamps']=times['full_timestamp_seq'].to_numpy()
        return side

    def __call__(self,payload):
        side=self._load_block_side_info(payload['user_info_file'],payload['item_info_file'])
        rows=np.asarray(payload['rows']).copy()
        positions=rows[:,self.column('user_index')].astype(np.int64)-side['user_index_min']
        user_rows=side['user_row_lookup'][torch.from_numpy(positions)].numpy()
        ages=np.zeros((len(rows),self.max_len),dtype=np.int64)
        for index,user_row in enumerate(user_rows):
            times=np.asarray(side['timestamps'][user_row],dtype=np.int64)
            target=int(rows[index,self.column('seq_len')]);timestamp=times[target]
            length=visible_prefix_length(times,target);rows[index,self.column('seq_len')]=length
            selected=times[max(0,length-self.max_len):length];size=len(selected)
            if size:ages[index,-size:]=timestamp-selected
        result=super().__call__({**payload,'rows':rows})
        batch,items,mask,_=result
        action=items['action'].numpy().copy()
        action[:,:-1]=mature_actions(action[:,:-1],ages,self.click_only,self.window)
        action[:,:-1]*=mask.numpy().astype(np.int64)
        items['action']=torch.from_numpy(action)
        task_mask=self.action_task_table[items['action'][:,:-1]]*mask.unsqueeze(-1)
        return batch,items,mask,[task_mask[:,:,index] for index in range(len(self.task_labels))]

    def column(self,name):return next(index for feature,index in self.batch_cols if feature==name)


def select_policy(policy):
    if policy not in ['official_offline','matured_24h']:raise ValueError('Choose an explicit history visibility policy')
    upstream.BlockedBatchCollator=OriginalCollator if policy=='official_offline' else MaturedCollator
