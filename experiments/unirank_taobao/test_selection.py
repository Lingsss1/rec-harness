"""Selection paths disagree with click-only ranking in these regression cases."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parents[1]
os.environ.setdefault('REC_CTR_PROJECT',str(PROJECT))
sys.path.append(str(PROJECT/'experiments/taobao_ctr_dual'))
spec=importlib.util.spec_from_file_location('legacy_controller',PROJECT/'experiments/taobao_ctr_dual/controller.py')
legacy=importlib.util.module_from_spec(spec);sys.modules['legacy_controller']=legacy;spec.loader.exec_module(legacy)
import controller
from common import read,write,sha
from selection import POLICY,TASKS,aggregate,best_trial,checkpoint_key
from topics import TopicPool
from assess import selection as freeze_selection
from audit_selection import audit


def metric(values,loss=.2):
    tasks={task:{'auc':value,'logloss':loss,'samples':100,'positives':10} for task,value in zip(TASKS,values)}
    return {'per_task':tasks,**aggregate(tasks),'samples':100}


class MultiTaskSelection(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.source=self.root/'model.py';self.source.write_text('class Model: pass\n')
        self.checkpoint=self.root/'checkpoint.pt';self.checkpoint.write_bytes(b'checkpoint identity fixture only')
        self.ref=self.row('din',[.6,.8,.8,.8]);self.ref.update(baseline_kind='din',role='baseline')
        self.click=self.row('click_winner',[.7,.65,.65,.65])
        self.balanced=self.row('balanced',[.61,.82,.82,.82])
        self.protocol={'identity':'p','data_identity':'d','selection_policy':POLICY,'primary':'four-task macro ROC-AUC',
            'source_sha256':{},'train_seed':42,'estimated_unit_seconds':1,'epochs':1,'batch_size':8,'history_limit':100}

    def tearDown(self):self.temp.cleanup()

    def row(self,name,values):
        return {'id':name,'status':'success','role':'candidate','metrics':{
            'validation':metric(values),'parameters':100,'data_identity':'d','effective_config':{},
            'checkpoint_sha256':sha(self.checkpoint),'train_seconds':1},'seed':42,'config':{},
            'source_sha256':sha(self.source),'model_path':str(self.source),'output':str(self.root),'wall_seconds':1,'round':0}

    def pool(self):
        topics=[{'title':str(i),'question':'question'+str(i),'hypothesis':'hypothesis'} for i in range(2)]
        pool=TopicPool(self.root,[0,1],topics,[self.ref],self.protocol,time.time()+1000)
        for gpu,row in enumerate([self.balanced,self.click]):
            pool.apply_decision(gpu,{'action':'continue','reason':'fixture','evidence_refs':['din']},self.ref)
            row=copy.deepcopy(row);row['topic_id']=pool.states[gpu]['active_topic_id'];pool.states[gpu]['trials'].append(row)
        return pool

    def test_checkpoint_and_incumbent_prefer_macro_over_click(self):
        self.assertLess(checkpoint_key(self.balanced['metrics']['validation']),checkpoint_key(self.click['metrics']['validation']))
        # Even a stale click-only alias cannot change the canonical decision.
        self.balanced['metrics']['validation']['auc']=.1
        self.click['metrics']['validation']['auc']=.99
        self.assertEqual(best_trial({'trials':[self.click,self.balanced]})['id'],'balanced')
        self.assertEqual(controller.best_trial({'trials':[self.click,self.balanced]})['id'],'balanced')

    def test_missing_task_and_nonfinite_metrics_are_rejected(self):
        per_task=copy.deepcopy(self.balanced['metrics']['validation']['per_task']);per_task.pop('buy')
        with self.assertRaises(ValueError):aggregate(per_task)
        per_task=copy.deepcopy(self.balanced['metrics']['validation']['per_task']);per_task['buy']['auc']=float('nan')
        with self.assertRaises(ValueError):aggregate(per_task)
        per_task['buy']['auc']=None
        with self.assertRaises(ValueError):aggregate(per_task)
        self.assertIsNone(aggregate(per_task,allow_undefined=True)['macro_auc'])

    def test_ties_use_macro_logloss_then_parameters_then_identity(self):
        other=copy.deepcopy(self.balanced);other['id']='other';other['metrics']['parameters']=99
        self.assertEqual(best_trial({'trials':[self.balanced,other]})['id'],'other')
        other['metrics']['validation']=metric([.61,.82,.82,.82],loss=.3)
        self.assertEqual(best_trial({'trials':[self.balanced,other]})['id'],'balanced')

    def test_topic_parent_gains_merge_gate_and_light_context_use_all_tasks(self):
        pool=self.pool();board={row['gpu']:row for row in pool.board()['topics']}
        self.assertEqual(pool.call_state(0)['incumbent'],'balanced')
        self.assertEqual(pool.call_state(1)['incumbent'],'din')
        self.assertAlmostEqual(board[0]['incremental_gain'],.0175)
        self.assertLess(board[1]['incremental_gain'],0)
        snap=pool.decision_snapshot(0)
        row=next(row for row in snap['light_context']['topics'] if row['gpu']==0)
        self.assertEqual(set(row['best_per_task']),set(TASKS))
        self.assertAlmostEqual(row['task_deltas_vs_reference']['buy']['auc'],.02)
        self.assertEqual(snap['light_context']['protocol']['selection_policy'],POLICY)
        decision={'action':'merge','reason':'test','evidence_refs':['balanced','click_winner'],
            'topic':{'title':'merge','question':'merge question','hypothesis':'merge hypothesis'},
            'source_topic_ids':['g0-t000','g1-t000'],'merge_control_refs':['balanced','click_winner']}
        with self.assertRaisesRegex(ValueError,'positive increments'):pool.validate_decision(0,{'topic_decision':decision},snap)

    def test_final_freeze_selects_macro_and_audit_never_reads_test(self):
        write(self.root/'protocol.json',self.protocol);write(self.root/'references.json',{'trials':[self.ref]})
        write(self.root/'dataset-source-manifest.json',{'revision':'fixture'})
        write(self.root/'status.json',{'phase':'topic_research_stopped'})
        for gpu in range(4):write(self.root/f'topics/gpu-{gpu}/state.json',{
            'phase':'research_stopped','trials':[self.click,self.balanced] if gpu==0 else [],'completed_research_rounds':1})
        frozen=freeze_selection(self.root)
        self.assertEqual(next(x for x in frozen['suite'] if x['key']=='selected')['id'],'balanced')
        paths=[]
        def checked(path):
            paths.append(str(path));return read(path)
        with patch('audit_selection.read',side_effect=checked):report=audit(self.root,self.root/'audit')
        self.assertEqual(report['audit_validation_best']['id'],'balanced')
        self.assertFalse(report['test_accessed'])
        self.assertFalse(any('/test/' in path or '/final-assessment/' in path for path in paths))

    def test_multitask_decisions_must_acknowledge_every_task(self):
        pool=self.pool();snapshot=pool.decision_snapshot(0)
        decision={'action':'continue','reason':'inspect macro gains and task deltas','evidence_refs':['balanced']}
        with self.assertRaisesRegex(ValueError,'every task'):
            pool.validate_decision(0,{'topic_decision':decision},snapshot)
        decision['task_tradeoffs']={task:'observed validation gain; repetition remains needed' for task in TASKS}
        self.assertEqual(pool.validate_decision(0,{'topic_decision':decision},snapshot)['action'],'continue')


if __name__=='__main__':unittest.main()
