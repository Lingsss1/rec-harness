import tempfile
import unittest
from pathlib import Path
from rec_harness.core import ContractError,file_hash,write_json
from rec_harness.adapters.amazon_role_sources import development_sources,verify_new_sources


class NoveltyTests(unittest.TestCase):
    def test_prior_study_history_is_frozen_and_protocol_bound(self):
        from unittest.mock import patch
        from rec_harness.store import Store
        from rec_harness.adapters.amazon_role_sources import freeze_source_history
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=Store(root/'memory');store.close()
            study={'protocol_hash':'p','data_hashes':{'d':'h'},'development_seeds':[42],
                   'datasets':['d'],'model_versions':{'old':{'sha256':'source'}}}
            write_json(root/'jobs/old-d-42/completed.json',{'source_hash':'source'})
            config={'source_history_studies':[str(root)]}
            with patch('rec_harness.adapters.amazon_controls.load_config',return_value=study):
                freeze_source_history(config,study)
                self.assertEqual(config['prior_development_sources'][0]['sources'],
                                 [{'model_version':'old','source_hash':'source'}])
                with self.assertRaisesRegex(ContractError,'changed protocol_hash'):
                    freeze_source_history({'source_history_studies':[str(root)]},{**study,'protocol_hash':'other'})
            with self.assertRaisesRegex(ContractError,'controller-owned'):freeze_source_history(config,study)

    def test_duplicate_is_repaired_inside_experiment_role_and_costs_retained(self):
        from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
        from rec_harness.store import Store
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'baseline.py';source.write_text('candidate')
            old=root/'old.py';old.write_text('old control')
            prior=[{'model_version':'prior','source_hash':file_hash(old)}]
            calls=[]
            class Agent:
                def run(self,role,context,directory):
                    calls.append(context)
                    write_json(directory/'request.json',{'role':role,'context':context})
                    (directory/'model/control-1.py').write_text('old control' if len(calls)==1 else 'new control')
                    write_json(directory/'resources.json',{'status':'success'})
                    response={'changed_files':['model.py','control-1.py'],'implementation_checks':['fixture'],'limitations':['fixture']}
                    write_json(directory/'response.json',response)
                    return response
            controller=object.__new__(AmazonAutoresearch)
            controller.store=Store(root/'memory');controller.store.configure_budget(3);controller.agent=Agent()
            try:
                controller._role(root/'round-1','experiment',{'generated_controls':[{'file':'control-1.py'}]},source,
                    lambda response,directory:verify_new_sources(directory/'model',response['changed_files'],prior))
                self.assertEqual(len(calls),2)
                self.assertIn('duplicates previously evaluated',calls[1]['validation_error'])
                self.assertEqual(controller.store.budget()['charged_or_reserved'],2)
                self.assertTrue((root/'round-1/experiment-rejected-1/rejection.json').is_file())
            finally:controller.store.close()

    def test_only_development_identities_and_exclude_current_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cfg={'model_versions':{name:{'sha256':name} for name in ['old','current','confirmation-only']},
                 'datasets':['d'],'development_seeds':[42]}
            for name,seed in [('old',42),('current',42),('confirmation-only',83)]:
                write_json(root/f'jobs/{name}-d-{seed}/completed.json',{'source_hash':name,'metrics':{'private':seed}})
            # Must not even parse a malformed confirmation receipt.
            (root/'jobs/confirmation-only-d-83/completed.json').write_text('not JSON')
            self.assertEqual(development_sources(root,cfg,exclude=['current']),
                             [{'model_version':'old','source_hash':'old'}])
            write_json(root/'jobs/old-d-42/completed.json',{'source_hash':'tampered'})
            with self.assertRaisesRegex(ContractError,'differs'):development_sources(root,cfg)

    def test_duplicate_control_reports_actionable_error_before_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'model.py').write_text('new candidate')
            (root/'control-1.py').write_text('old implementation')
            prior=[{'model_version':'old-model','source_hash':file_hash(root/'control-1.py')}]
            with self.assertRaisesRegex(ContractError,'control-1.py duplicates previously evaluated model old-model'):
                verify_new_sources(root,['model.py','control-1.py'],prior)
            (root/'control-1.py').write_text('distinct control')
            verify_new_sources(root,['model.py','control-1.py'],prior)
            (root/'model.py').write_text('old implementation')
            with self.assertRaisesRegex(ContractError,'model.py duplicates'):verify_new_sources(root,['model.py'],prior)
