import copy
import unittest

from rec_harness.adapters.amazon_selection import summarize_development,release_confirmation
from rec_harness.core import ContractError


class DevelopmentSelectionTests(unittest.TestCase):
    def fixture(self):
        config={'datasets':['a','b'],'development_seeds':[42],'protocol':{'metric':'HR@10','minimum_effect':0.005}}
        plan={'baseline':'base','versions':{'candidate':{'model_version':'candidate'},'controls':[{'model_version':'control'}]}}
        rows=[{'model_version':v,'dataset':d,'seed':42,'metrics':{'HR@10':score}}
              for v,score in [('base',0.5),('candidate',0.52),('control',0.51)] for d in config['datasets']]
        return config,plan,rows

    def test_requires_both_baseline_effect_and_control_advantage(self):
        config,plan,rows=self.fixture()
        self.assertTrue(summarize_development(config,plan,rows)['eligible_for_confirmation'])
        for row in rows:
            if row['model_version']=='control':row['metrics']['HR@10']=0.53
        self.assertFalse(summarize_development(config,plan,rows)['eligible_for_confirmation'])
        for row in rows:
            if row['model_version']=='control':row['metrics']['HR@10']=0.49
            if row['model_version']=='candidate':row['metrics']['HR@10']=0.504
        self.assertFalse(summarize_development(config,plan,rows)['eligible_for_confirmation'])

    def test_missing_duplicate_or_confirmation_rows_cannot_release(self):
        config,plan,rows=self.fixture()
        for changed in [rows[:-1],rows+[rows[0]],rows+[dict(rows[0],seed=17)]]:
            with self.assertRaisesRegex(ContractError,'exactly'):
                summarize_development(config,plan,changed)

    def test_unconfirmed_scientific_claim_or_wrong_evidence_cannot_release(self):
        decision={'selection':'keep','conclusion_status':'supported','rationale':'fixture',
                  'evidence_refs':['readout'],'unresolved_explanations':[]}
        with self.assertRaisesRegex(ContractError,'provisional'):
            release_confirmation('/never-accessed','plan',decision,'readout')
        decision['conclusion_status']='provisional'; decision['evidence_refs']=['other']
        with self.assertRaisesRegex(ContractError,'exact'):
            release_confirmation('/never-accessed','plan',decision,'readout')

    def test_incumbent_control_reuse_does_not_duplicate_evidence(self):
        config,plan,rows=self.fixture()
        plan['versions']['controls']=[{'model_version':'base'}]
        rows=[row for row in rows if row['model_version']!='control']
        result=summarize_development(config,plan,rows)
        self.assertEqual(set(result['means']),{'base','candidate'})
        self.assertEqual(len(result['slices']),2)
        self.assertTrue(result['eligible_for_confirmation'])

    def test_newly_promoted_incumbent_confirmation_can_be_reused(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from rec_harness.store import Store
        from rec_harness.core import write_json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=Store(root/'memory')
            store.event('amazon_research_plan_registered',{'proposal_id':'plan','baseline':'promoted-model',
                'reused_control_versions':['promoted-model']});store.close()
            config={'datasets':['tiny'],'confirmation_seeds':[17,83]}
            report={'hash':'readout','summary':{'eligible_for_confirmation':True},
                    'source_hashes':{'promoted-model':'basehash','new-candidate':'newhash'},'plan_handoff_hash':'planhash'}
            (root/'jobs/promoted-model-tiny-17').mkdir(parents=True)
            decision={'selection':'keep','conclusion_status':'provisional','rationale':'fixture',
                      'evidence_refs':['readout'],'unresolved_explanations':[]}
            with patch('rec_harness.adapters.amazon_selection.load_config',return_value=config), patch('rec_harness.adapters.amazon_selection.development_readout',return_value=report):
                record=release_confirmation(root,'plan',decision,'readout')
                self.assertEqual(record['proposal_id'],'plan')
                self.assertEqual(release_confirmation(root,'plan',decision,'readout'),record)
