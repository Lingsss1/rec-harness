import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rec_harness.adapters.amazon_campaign import continue_campaign,activate_campaign,campaign_guard
from rec_harness.adapters.amazon_study import init_study,DATASETS
from rec_harness.core import ContractError,read_json,write_json,digest
from rec_harness.store import Store


class CampaignContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.project=self.root/'project'
        for name in ['results/nextitnet-fixedbase-project-12h-20260921/fixed_baseline/best.py',
                     'workspace/nextitnet-fixedbase-project-12h-20260921/nextitnet_perdataset/best.py']:
            p=self.project/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('# source fixture\n')
        for dataset in DATASETS:
            for split in ['train','valid']:
                p=self.project/f'input/amazon-project-defaults/trainval/{dataset}_{split}.txt'
                p.parent.mkdir(parents=True,exist_ok=True);p.write_text('fixture\n')
        self.old=self.root/'old';self.new=self.root/'new';init_study(self.old,self.project,budget=100)
        st=Store(self.old/'memory');st.reserve('historical',60);st.settle('historical',60,'success');st.close()
        cfg=read_json(self.old/'study.json')
        self.assertEqual(cfg['development_seeds'],[42])
        self.assertEqual(cfg['confirmation_seeds'],[])
        self.assertEqual(cfg['selection_policy'],'baseline-single-seed-v1')
        for dataset in DATASETS:
            for seed in cfg['development_seeds']+cfg['confirmation_seeds']:
                job=f'legacy_incumbent-{dataset}-{seed}'
                write_json(self.old/'jobs'/job/'completed.json',{'job_id':job})
        self.audit=patch('rec_harness.adapters.amazon_campaign.audit_job',side_effect=lambda root,c,v,d,s:
            (read_json(root/'jobs'/f'{v}-{d}-{s}'/'completed.json'),{'fixture':'manifest'}))
        self.audit.start();self.addCleanup(self.audit.stop)

    def test_extension_retains_cost_and_seals_predecessor(self):
        continue_campaign(self.new,self.old,self.project,40,'user authorized additional 40')
        st=Store(self.new/'memory')
        try:
            self.assertEqual(st.budget(),{'limit':140.,'charged_or_reserved':60.,'remaining':80.})
            campaign_guard(read_json(self.new/'study.json'),st)
            self.assertEqual(len(list((self.new/'jobs').glob('*/completed.json'))),4)
        finally:st.close()
        st=Store(self.old/'memory')
        try:
            self.assertEqual(st.budget()['limit'],100)
            with self.assertRaisesRegex(ContractError,'successor'):campaign_guard(read_json(self.old/'study.json'),st)
        finally:st.close()
        with self.assertRaisesRegex(ContractError,'already continued'):
            continue_campaign(self.root/'fork',self.old,self.project,40,'same authorization')

    def test_legacy_continuation_retains_its_frozen_seeds(self):
        cfg=read_json(self.old/'study.json')
        cfg.update(confirmation_seeds=[17,83],selection_policy='baseline-positive-v1')
        write_json(self.old/'study.json',cfg)
        write_json(self.old/'study-lock.json',{'config_hash':digest(cfg)})
        for dataset in DATASETS:
            for seed in [17,83]:
                job=f'legacy_incumbent-{dataset}-{seed}'
                write_json(self.old/'jobs'/job/'completed.json',{'job_id':job})
        continue_campaign(self.new,self.old,self.project,40,'authorized continuation')
        new=read_json(self.new/'study.json')
        self.assertEqual(new['confirmation_seeds'],[17,83])
        self.assertEqual(new['selection_policy'],'baseline-positive-v1')
        self.assertEqual(len(list((self.new/'jobs').glob('*/completed.json'))),12)

    def test_crash_after_sealing_cannot_spend_until_activation_recovers(self):
        with patch('rec_harness.adapters.amazon_campaign.activate_campaign',side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):
                continue_campaign(self.new,self.old,self.project,40,'user authorized additional 40')
        st=Store(self.new/'memory')
        try:
            with self.assertRaisesRegex(ContractError,'not activated'):campaign_guard(read_json(self.new/'study.json'),st)
        finally:st.close()
        first=activate_campaign(self.new);self.assertEqual(first,activate_campaign(self.new))
        st=Store(self.new/'memory')
        try:campaign_guard(read_json(self.new/'study.json'),st);self.assertEqual(st.budget()['charged_or_reserved'],60.)
        finally:st.close()
