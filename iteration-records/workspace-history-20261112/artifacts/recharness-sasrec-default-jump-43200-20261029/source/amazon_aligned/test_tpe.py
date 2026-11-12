import ast
import os
from pathlib import Path
import pickle
import re
import tempfile
import time
import unittest
from unittest.mock import patch

from .core import read, save
from .prepare import ROOT, REPO
from .tpe import (BASELINE, SAMPLER, SPACE, TEMPLATES, TPEController,
                  optuna_module, source, suggest, study_for, save_sampler, tell)
from .tpe_launch import smoke_params


class TPETests(unittest.TestCase):
    def test_full_async_driver_respects_budget_and_reaches_tpe_sampling(self):
        from . import tpe
        optuna = optuna_module()
        class SyntheticController(TPEController):
            # Exercise the real ask/tell scheduler and ledger without GPU training.
            def _trial(self, key, source, gpu, requested, files, smoke):
                folder = self.root / 'trials' / key
                folder.mkdir(parents=True)
                (folder / 'train.py').write_text(source)
                allowance = self.ledger.reserve(key, requested or self.c['trial_timeout_seconds'],
                    minimum=self.c['minimum_trial_seconds'], fair_slots=None if requested else len(self.c['gpus']))
                time.sleep(.01)
                elapsed = min(1., allowance)
                self.ledger.settle(key, elapsed)
                failed = key == 'tpe-00002'
                record = {'id': key, 'status': 'crash' if failed else 'success',
                    'score': None if failed else .505 if key == 'baseline' else .6,
                    'allocated_gpu_seconds': elapsed, 'output': str(folder)}
                save(folder / 'receipt.json', record)
                with self.event_lock:
                    if record['status'] == 'success' and (self.best is None or record['score'] > self.best['score']):
                        self.best = record
                return record
            def finish(self):
                return {'selected': self.brief(self.best), 'test': {'synthetic': True},
                        'budget': read(self.ledger.path)}
        with tempfile.TemporaryDirectory() as root:
            config = {**read(ROOT / 'config.json'), 'root': root, 'sampler_seed': 42,
                      'hpo_models': ['sasrec'], 'search_budget_gpu_seconds': 40,
                      'trial_timeout_seconds': 8, 'minimum_trial_seconds': 1,
                      'llm_model': None, 'llm_effort': None}
            with patch.object(tpe, 'TPEController', SyntheticController):
                result = tpe.run(config)
            self.assertLessEqual(result['budget']['spent_gpu_seconds'], 40)
            self.assertEqual(result['budget']['reservations'], {})
            self.assertGreater(result['sampling_phase_counts']['tpe'], 0)
            self.assertIn('tpe-00002', result['budget']['settled'])
            study = optuna.load_study(study_name='amazon-tpe', storage='sqlite:///' + str(Path(root) / 'tpe/study.sqlite3'))
            self.assertFalse(any(t.state == optuna.trial.TrialState.RUNNING for t in study.trials))
            self.assertEqual(study.trials[2].state, optuna.trial.TrialState.FAIL)

    def test_fixed_sasrec_has_no_model_selection_dimension(self):
        optuna = optuna_module()
        with tempfile.TemporaryDirectory() as root:
            c = TPEController({**read(ROOT / 'config.json'), 'root': root,
                               'sampler_seed': 42, 'hpo_models': ['sasrec']})
            study = study_for(c)
            for i in range(9):
                trial = study.ask()
                params = suggest(trial, ['sasrec'])
                self.assertEqual(params['model'], 'sasrec')
                self.assertNotIn('model', trial.params)
                self.assertFalse(any(k.startswith(('gru4rec_', 'bert4rec_', 'nextitnet_', 'hstu_')) for k in trial.params))
                if i == 0:
                    self.assertEqual(params, BASELINE)
                study.tell(trial, .5 + i * .001)

    def test_parameter_binding_preserves_all_native_model_and_training_functions(self):
        for model, (folder, _) in TEMPLATES.items():
            original = ast.parse((REPO / 'gagc/templates' / folder / 'train.py').read_text())
            generated = ast.parse(source(smoke_params(model)))
            def definitions(tree):
                return {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                        if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            self.assertEqual(definitions(original), definitions(generated), model)
            # HSTU's relocated standalone predictor must also remain valid Python.
            if model == 'hstu':
                writer = generated.body[-1].body[-1].value
                ast.parse(writer.args[0].value)

    def test_all_model_parameters_and_smoke_limits_reach_the_native_environment(self):
        for model, (_, prefix) in TEMPLATES.items():
            p = smoke_params(model)
            code = source(p)
            binding = re.search(r'# Frozen TPE parameter binding[^\n]*\n(.*?)\n\n', code, re.S).group(1)
            for smoke in [False, True]:
                with patch.dict(os.environ, {'SASREC_EPOCHS': '5' if smoke else '200',
                                             'SASREC_BATCHES': '2' if smoke else '200'}, clear=True):
                    exec(binding, {})
                    self.assertEqual(os.environ[prefix + '_HIDDEN'], str(p['hidden']))
                    self.assertEqual(os.environ[prefix + '_MAXLEN'], str(p['maxlen']))
                    self.assertEqual(os.environ[prefix + '_EPOCHS'], '5' if smoke else '200')
                    self.assertEqual(os.environ[prefix + '_BATCHES'], '2' if smoke else '200')
                    self.assertEqual(os.environ[prefix + '_BATCH'], '32' if smoke else str(p['batch']))
                    self.assertEqual(os.environ['GAGC_SEED'], '42')

    def test_suggestions_are_valid_and_have_no_per_seed_or_ensemble_dimension(self):
        optuna = optuna_module()
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42, **SAMPLER))
        seen = set()
        for _ in range(30):
            trial = study.ask()
            p = suggest(trial, list(TEMPLATES))
            seen.add(p['model'])
            if p['model'] + '_heads' in p:
                self.assertEqual(p['hidden'] % p[p['model'] + '_heads'], 0)
            self.assertEqual('weight_decay' in p, p['weight_decay_mode'] == 'positive')
            self.assertNotIn('seed', p)
            self.assertNotIn('ensemble', p)
            study.tell(trial, p['lr'])
        self.assertEqual(seen, set(TEMPLATES))

    def test_baseline_is_enqueued_and_sampler_rng_survives_checkpoint(self):
        optuna = optuna_module()
        with tempfile.TemporaryDirectory() as root:
            config = {**read(ROOT / 'config.json'), 'root': root, 'sampler_seed': 42}
            controller = TPEController(config)
            study = study_for(controller)
            baseline = study.ask()
            self.assertEqual(suggest(baseline, list(TEMPLATES)), BASELINE)
            study.tell(baseline, .505)
            save_sampler(controller, study.sampler)
            resumed = study_for(controller)
            self.assertEqual(len(resumed.trials), 1)
            self.assertEqual(resumed.best_value, .505)
            sampler_bytes = pickle.dumps(resumed.sampler)
            old_trials = resumed.get_trials()
            def next_parameters():
                copy = optuna.create_study(sampler=pickle.loads(sampler_bytes))
                copy.add_trials(old_trials)
                return suggest(copy.ask(), list(TEMPLATES))
            self.assertEqual(next_parameters(), next_parameters())
            self.assertEqual(read(controller.ledger.path)['spent_gpu_seconds'], 0)

    def test_failed_trials_remain_charged_without_a_fabricated_objective(self):
        optuna = optuna_module()
        with tempfile.TemporaryDirectory() as root:
            c = TPEController({**read(ROOT / 'config.json'), 'root': root, 'sampler_seed': 42})
            study = study_for(c)
            trial = study.ask()
            suggest(trial, list(TEMPLATES))
            c.ledger.reserve('failed', 100, minimum=1)
            c.ledger.settle('failed', 17.25)
            tell(study, trial.number, {'id': 'failed', 'status': 'timeout', 'score': None})
            frozen = study.trials[0]
            self.assertEqual(frozen.state, optuna.trial.TrialState.FAIL)
            self.assertIsNone(frozen.value)
            self.assertEqual(read(c.ledger.path)['spent_gpu_seconds'], 17.25)
            with self.assertRaises(RuntimeError):
                c.llm('must never be called')


if __name__ == '__main__':
    unittest.main()
