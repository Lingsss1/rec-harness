import copy
import unittest
from pathlib import Path

from rec_harness.core import ContractError,digest,write_json
from rec_harness.adapters.amazon_replay import build_cycle_world
from rec_harness.adapters.amazon_replay_chain import compose_chain,validate_lineage
from rec_harness.replay import Replay
import test_amazon_replay


class ChainReplayTests(unittest.TestCase):
    def test_lineage_requires_exact_predecessors_heads_costs_and_no_pending_work(self):
        previous={'source':{'controller_head':'role-head','study_head':'gpu-head'},
                  'resources':{'campaign_cli_calls':11,'retained_gpu_seconds':100}}
        lineage={'role_predecessor':'/prior/campaign','gpu_predecessor':'/prior/study',
                 'role_cost':{'integrity':{'head':'role-head'},'charged_or_reserved':11,'pending':0},
                 'gpu_cost':{'integrity':{'head':'gpu-head'},'charged_or_reserved':100,'pending':0}}
        args=(previous,Path('/prior/campaign'),Path('/prior/study'))
        validate_lineage(lineage,*args)
        for field in ['role_predecessor','gpu_predecessor']:
            with self.assertRaises(ContractError):validate_lineage({**lineage,field:'/different'},*args)
        for kind in ['role','gpu']:
            for field,value in [('integrity',{'head':'changed'}),('charged_or_reserved',0),('pending',1)]:
                bad=copy.deepcopy(lineage);bad[kind+'_cost'][field]=value
                with self.assertRaises(ContractError):validate_lineage(bad,*args)

    def setUp(self):
        fixture=test_amazon_replay.AmazonReplayTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.artifacts=fixture.args['artifacts']
        self.first=build_cycle_world(**fixture.args)
        args=dict(fixture.args)
        visible=copy.deepcopy(args['visible']);visible['bindings']['proposal_id']='second'
        visible['history']=[args['report']]
        path=self.artifacts.root.parent/'second-context.json';write_json(path,visible)
        args.update(proposal='second',visible=visible,request_ref=self.artifacts.put(path),gpu_total=150,role_calls=18)
        self.second=build_cycle_world(**args)

    def test_sequential_revelation_and_cumulative_cost_without_double_charge(self):
        world=compose_chain([self.first,self.second],self.artifacts)
        self.assertEqual(world['nodes'][1]['cost'],64)
        replay=Replay(world,186)
        self.assertNotIn('second',str(replay.view()))
        with self.assertRaises(ContractError):replay.step(['proposal'])
        replay.step(['root'])
        self.assertEqual(replay.spent,122)
        self.assertNotIn('second',str(replay.view()))
        replay.step(['proposal']);self.assertEqual(replay.spent,186)
        self.assertEqual(replay.report()['reveals'],2)
        limited=Replay(world,185);limited.step(['root'])
        self.assertEqual(limited.step(['proposal'])['status'],'budget_exhausted')
        self.assertNotIn('second',limited.view()['revealed_node_ids'])

    def test_rejects_unlinked_context_or_changed_incumbent_and_cost(self):
        for change,error in [('context','predecessor readout'),('baseline','baseline differs'),('cost','cost decreased')]:
            second=copy.deepcopy(self.second)
            if change=='context':
                path=self.artifacts.root.parent/'unlinked.json';write_json(path,{'history':[]})
                second['nodes'][0]['request_ref']=self.artifacts.put(path)
            elif change=='baseline':second['root_observation']['score']=.6
            else:second['resources']['campaign_cli_calls']=1
            second['hash']=digest({k:v for k,v in second.items() if k!='hash'})
            with self.assertRaisesRegex(ContractError,error):compose_chain([self.first,second],self.artifacts)

    def test_development_keep_does_not_imply_confirmed_promotion(self):
        first=copy.deepcopy(self.first)
        first['nodes'][0]['observation'].update(selection='keep',eligible_for_confirmation=True,score=.501)
        first['hash']=digest({k:v for k,v in first.items() if k!='hash'})
        # Next observed campaign still uses the baseline: confirmation may reject.
        world=compose_chain([first,self.second],self.artifacts)
        self.assertEqual(world['nodes'][1]['workspace_before'],first['nodes'][0]['workspace_before'])
        promoted=copy.deepcopy(self.second)
        promoted['nodes'][0]['workspace_before']=first['nodes'][0]['workspace_after']
        promoted['root_observation']['score']=.501
        promoted['hash']=digest({k:v for k,v in promoted.items() if k!='hash'})
        self.assertEqual(len(compose_chain([first,promoted],self.artifacts)['nodes']),2)
