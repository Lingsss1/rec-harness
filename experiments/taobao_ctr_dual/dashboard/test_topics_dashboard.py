"""Observer boundaries: only committed, matching, full-protocol validation."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from rich.console import Console
from collect import Collector,Reader
from topics_app import TopicCollector,render


class TopicDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.put('budget.json',{'gpus':[0],'start_unix':time.time()-30,'deadline_unix':time.time()+86400,
                              'decision_every':6,'model':'gpt-6.1-sol','reasoning_effort':'high'})
        self.put('protocol.json',{'data_identity':'d','epochs':3,'batch_size':4096})
        self.reference={'id':'baseline-din','baseline_kind':'din','status':'success','output':str(self.root/'reference'),
                        'wall_seconds':1,'metrics':{'data_identity':'d','validation':{'auc':.6,'logloss':.3}}}
        self.put('references.json',{'trials':[self.reference]})
        self.state={'gpu':0,'phase':'propose','completed_research_rounds':5,'active_topic_id':'g0-t000',
                    'topics':[{'id':'g0-t000','title':'fixture','question':'question','hypothesis':'hypothesis','entry_auc':.6}],
                    'trials':[],'agent_calls':[],'decisions':[]}
        self.put('topics/gpu-0/state.json',self.state)
    def tearDown(self):self.temp.cleanup()
    def put(self,name,value):
        path=self.root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value))
    def snapshot(self):
        with patch.object(Collector,'resources',return_value={'gpus':[]}):return TopicCollector(self.root).snapshot()

    def test_only_committed_matching_validation_and_no_test_archive_reads(self):
        good=copy.deepcopy(self.reference)
        good.update(id='g0-good',baseline_kind=None,topic_id='g0-t000',role='candidate')
        good['metrics']['validation']['auc']=.62
        rows=[good]
        for name,extra in [('test',{'split':'test'}),('calibration',{'calibration_only':True}),('foreign',{'data_identity':'other'})]:
            row=copy.deepcopy(good);row['id']=name;row['metrics'].update(extra);row['metrics']['validation']['auc']=.99
            rows.append(row)
        self.state['trials']=rows;self.put('topics/gpu-0/state.json',self.state)
        self.put('topics/gpu-0/trials/uncommitted/result.json',{'validation':{'auc':.999}})
        self.put('assessment/result.json',{'secret':'test score'})
        self.put('final.json',{'secret':'test score'})
        paths=[];original=Reader.json
        def track(reader,path,default=None):paths.append(str(path));return original(reader,path,default)
        with patch.object(Reader,'json',track):snapshot=self.snapshot()
        self.assertEqual(snapshot['lanes'][0]['success'],1)
        self.assertEqual(snapshot['lanes'][0]['best']['id'],'g0-good')
        self.assertFalse(any('/assessment/' in path or path.endswith('/final.json') or 'uncommitted' in path for path in paths))

    def test_six_completed_rounds_drive_displayed_decision_milestones(self):
        self.assertEqual(self.snapshot()['lanes'][0]['next_decision_round'],6)
        self.state['completed_research_rounds']=6;self.put('topics/gpu-0/state.json',self.state)
        self.assertEqual(self.snapshot()['lanes'][0]['next_decision_round'],12)

    def test_views_render_before_first_training_result(self):
        snapshot=self.snapshot()
        for name in ['overview','resources','lane-0','experiments','decisions','costs','slices']:
            console=Console(width=140,height=45,record=True)
            with console.capture():console.print(render(name,snapshot,140,45))

    def test_48h_budget_can_render_before_dataset_ready(self):
        self.put('budget.json',{'gpus':[0],'start_unix':None,'deadline_unix':None,'seconds':172800,
            'start_event':'dataset_download_and_checksum_verification_completed','pre_budget':{}})
        snapshot=self.snapshot();console=Console(width=140,height=45,record=True)
        with console.capture() as output:
            console.print(render('overview',snapshot,140,45))
            console.print(render('costs',snapshot,140,45))
        self.assertIn('48h',output.get())
        self.assertNotIn('24h',output.get())

    def test_multitask_board_labels_and_ranking_ignore_click_only_winner(self):
        policy={'id':'macro-fixture','primary':'macro_auc'}
        self.put('protocol.json',{'data_identity':'d','selection_policy':policy})
        self.reference['metrics']['validation'].update(auc=.8,selection_policy_id='macro-fixture',selection_metric='macro_auc')
        self.put('references.json',{'trials':[self.reference]})
        rows=[]
        for name,mean,click,policy_id in [('balanced',.81,.61,'macro-fixture'),('click',.7,.9,'macro-fixture'),('foreign',.99,.99,None)]:
            row=copy.deepcopy(self.reference);row.update(id=name,baseline_kind=None,topic_id='g0-t000')
            row['metrics']['validation'].update(auc=mean,click_auc=click,selection_policy_id=policy_id)
            rows.append(row)
        self.state['trials']=rows;self.put('topics/gpu-0/state.json',self.state)
        snapshot=self.snapshot()
        self.assertEqual(snapshot['lanes'][0]['best']['id'],'balanced')
        self.assertEqual(snapshot['lanes'][0]['success'],2)
        console=Console(width=140,height=45)
        with console.capture() as output:console.print(render('overview',snapshot,140,45))
        self.assertIn('四任务平均',output.get())


if __name__=='__main__':unittest.main()
