import copy
from pathlib import Path
import tempfile
import unittest

from rec_harness.agents import schema_for
from rec_harness.adapters.amazon_plan import compile_plan, register_plan
from rec_harness.core import ContractError, digest, file_hash, write_json
from rec_harness.store import Store


def sample(schema):
    kind = schema['type']
    if 'enum' in schema: return schema['enum'][0]
    if kind == 'object': return {k: sample(v) for k, v in schema['properties'].items()}
    if kind == 'array': return [sample(schema['items'])]
    return 'fixture' if kind == 'string' else 1


class AmazonPlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        baseline = self.root / 'baseline.py'; baseline.write_text('raise RuntimeError("never import")\n')
        self.config = {'model_versions': {'baseline': {'path': str(baseline), 'sha256': file_hash(baseline)}},
                       'protocol_hash': 'protocol', 'datasets': ['a', 'b'], 'development_seeds': [42], 'confirmation_seeds': [17, 83]}
        write_json(self.root / 'study.json', self.config)
        write_json(self.root / 'study-lock.json', {'config_hash': digest(self.config)})
        store = Store(self.root / 'memory'); store.configure_budget(100); store.close()
        gap = sample(schema_for('evidence')); generated = sample(schema_for('hypothesis'))
        gap.update(gap_id='gap', model_version='baseline', protocol_hash='protocol')
        generated['method']['method_id'] = 'method'
        generated['proposal'].update(proposal_id='proposal', base_model_id='baseline', train_protocol_hash='protocol',
            gap_ids=['gap'], method_ids=['method'], controls=['control'], seeds=[42,17,83], budget=50)
        self.cards = {'gap': gap, **generated}
        self.versions = {'candidate': self.source('candidate'), 'controls': [self.source('control')]}

    def source(self, name):
        path = self.root / (name + '.py'); path.write_text(f'raise RuntimeError("must not execute {name}")\n')
        return {'model_version': name, 'source': {'path': str(path), 'sha256': file_hash(path)}}

    def test_registration_freezes_sources_and_preserves_confirmation_gate(self):
        before = (self.root / 'study.json').read_bytes()
        report = register_plan(self.root, self.cards, self.versions)
        self.assertFalse(report['candidate_admitted'])
        self.assertEqual([len(p['jobs']) for p in report['phases']], [6, 12])
        self.assertEqual(report['phases'][1]['release_condition'], 'frozen_selection_after_development')
        self.assertNotIn('path', report['versions']['candidate']['source'])
        from rec_harness.adapters.amazon_controls import load_config
        self.assertEqual(set(load_config(self.root)['model_versions']), {'baseline'})
        self.assertEqual(register_plan(self.root, self.cards, self.versions), report)
        store = Store(self.root / 'memory')
        try:
            self.assertEqual(store.budget()['charged_or_reserved'], 0)
            self.assertEqual(sum(e['kind'] == 'amazon_research_plan_registered' for e in store.events()), 1)
        finally: store.close()
        self.assertEqual((self.root / 'study.json').read_bytes(), before)
        self.cards['proposal']['mechanism'] = 'changed after registration'
        with self.assertRaisesRegex(ContractError, 'rewritten'):
            register_plan(self.root, self.cards, self.versions)

    def test_protocol_controls_and_budget_cannot_be_changed(self):
        for field, value, message in [('seeds',[1],'protocol'), ('controls',['imaginary'],'controls'), ('budget',101,'budget')]:
            cards = copy.deepcopy(self.cards); cards['proposal'][field] = value
            with self.assertRaisesRegex(ContractError, message):
                compile_plan(self.config, cards, self.versions, 100)

    def test_source_cannot_be_renamed_to_hide_prior_outcomes(self):
        write_json(self.root / 'attempt-history/old-name/completed.json',
                   {'source_hash': self.versions['candidate']['source']['sha256']})
        with self.assertRaisesRegex(ContractError, 'precede preregistration'):
            register_plan(self.root, self.cards, self.versions)

    def test_another_plan_cannot_reuse_version_names(self):
        register_plan(self.root, self.cards, self.versions)
        self.cards['proposal']['proposal_id'] = 'another-proposal'
        with self.assertRaisesRegex(ContractError, 'another registered plan'):
            register_plan(self.root, self.cards, self.versions)

    def test_existing_baseline_control_is_explicitly_reused_without_duplicate_jobs(self):
        self.cards['proposal']['controls'] = ['baseline']
        self.versions['controls'] = [{'model_version':'baseline','source':self.config['model_versions']['baseline']}]
        write_json(self.root/'jobs/baseline-a-42/completed.json',
                   {'source_hash':self.config['model_versions']['baseline']['sha256']})
        plan = register_plan(self.root,self.cards,self.versions)
        self.assertEqual(plan['reused_control_versions'],['baseline'])
        self.assertEqual([len(p['jobs']) for p in plan['phases']],[4,8])
        bad = copy.deepcopy(self.versions); bad['controls'][0]['source']['sha256']='other'
        with self.assertRaisesRegex(ContractError,'registered source'):
            compile_plan(self.config,self.cards,bad,100)
