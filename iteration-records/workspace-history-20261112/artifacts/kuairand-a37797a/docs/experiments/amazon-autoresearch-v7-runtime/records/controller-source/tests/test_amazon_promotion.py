import copy
import unittest
from rec_harness.adapters.amazon_selection import summarize_development
from rec_harness.adapters.amazon_promotion import confirm_promotion
from rec_harness.core import ContractError


class PromotionTests(unittest.TestCase):
    def fixture(self, candidate=.51, control=.53):
        config={'datasets':['a','b'],'development_seeds':[42],'confirmation_seeds':[17,83],
                'protocol':{'metric':'HR@10','minimum_effect':.005}}
        proposal={'mechanism':'recent context','insertion_contract':'candidate recent three; control earliest three',
                  'falsification_condition':'recent does not beat early','alternative_explanations':['context length']}
        versions={'candidate':{'model_version':'candidate','source':{'sha256':'c'}},
                  'controls':[{'model_version':'control','source':{'sha256':'k'}}]}
        plan={'baseline':'base','versions':versions,'selection_policy':'baseline-positive-v1',
              'cards':{'proposal':proposal,'gap':{'predicted_observations':['recent beats early']}}}
        rows=[{'model_version':v,'dataset':d,'seed':42,'metrics':{'HR@10':score}}
              for v,score in [('base',.5),('candidate',candidate),('control',control)] for d in ['a','b']]
        return config,plan,rows

    def test_control_wins_without_vetoing_candidate(self):
        config,plan,rows=self.fixture()
        summary=summarize_development(config,plan,rows)
        self.assertEqual(summary['promotion_order'],['control','candidate'])
        self.assertTrue(summary['eligible_for_confirmation'])
        comparison=summary['mechanism_comparisons'][0]
        self.assertLess(comparison['candidate_minus_control'],0)
        self.assertEqual(comparison['mechanism_status'],'unknown')
        self.assertIn('earliest three',comparison['design']['insertion_contract'])

    def test_positive_below_old_threshold_and_null_contrast(self):
        config,plan,rows=self.fixture(.501,.501)
        summary=summarize_development(config,plan,rows)
        self.assertEqual(summary['promotion_order'],['candidate','control'])
        self.assertEqual(summary['mechanism_comparisons'][0]['candidate_minus_control'],0)

    def test_only_control_positive_and_both_negative_still_recorded(self):
        for c,k,order in [(.49,.51,['control']),(.49,.48,[])]:
            config,plan,rows=self.fixture(c,k)
            summary=summarize_development(config,plan,rows)
            self.assertEqual(summary['promotion_order'],order)
            self.assertEqual(len(summary['mechanism_comparisons']),1)

    def test_confirmation_selects_by_frozen_development_order_and_requires_each_seed(self):
        config,plan,rows=self.fixture()
        summary=summarize_development(config,plan,rows)
        scores={'base':{17:.5,83:.5},'candidate':{17:.8,83:.8},'control':{17:.51,83:.51}}
        def audit(root,cfg,v,d,s):
            return {'metrics':{'HR@10':scores[v][s]}},{'candidates_hash':'same','data_hashes':{}}
        result=confirm_promotion(None,config,'base',summary,audit)
        self.assertEqual(result['promoted_model'],'control')
        scores['control'][83]=.5
        result=confirm_promotion(None,config,'base',summary,audit)
        self.assertEqual(result['promoted_model'],'candidate')
        scores['candidate'][83]=.49
        self.assertFalse(confirm_promotion(None,config,'base',summary,audit)['selected'])

    def test_confirmation_rejects_mismatched_identity_and_no_seeds(self):
        config,plan,rows=self.fixture();summary=summarize_development(config,plan,rows)
        def audit(root,cfg,v,d,s):
            return {'metrics':{'HR@10':.6}},{'candidates_hash':v,'data_hashes':{}}
        with self.assertRaisesRegex(ContractError,'pairing'):
            confirm_promotion(None,config,'base',summary,audit)
        config['confirmation_seeds']=[]
        with self.assertRaisesRegex(ContractError,'seeds'):
            confirm_promotion(None,config,'base',summary,audit)

    def test_reused_controls_not_promotable_and_old_policy_unchanged(self):
        config,plan,rows=self.fixture(.49,.6)
        plan['reused_control_versions']=['control']
        self.assertEqual(summarize_development(config,plan,rows)['promotion_order'],[])
        del plan['selection_policy']
        old=summarize_development(config,plan,rows)
        self.assertNotIn('promotion_order',old)
        self.assertFalse(old['eligible_for_confirmation'])
