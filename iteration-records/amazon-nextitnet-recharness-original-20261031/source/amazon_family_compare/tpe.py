"""Traditional Optuna TPE over pinned single-model RecHarness templates."""
import ast
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import pickle
import sys

from .core import BudgetExhausted, Controller, read, save
from .prepare import WORKSPACE, REPO, sha
from .rerun_controller import SingleSelectionMixin

DESTINATION = WORKSPACE / 'artifacts/amazon-tpe-sasrec-20261028'
# Reuse only the isolated software installation from the unlaunched proposal.
RUNTIME = WORKSPACE / 'artifacts/amazon-tpe-20261028/runtime-packages'
TEMPLATES = {
    'sasrec': ('perdataset', 'SASREC'),
    'gru4rec': ('gru4rec_perdataset', 'GRU4REC'),
    'nextitnet': ('nextitnet_perdataset', 'NEXTITNET'),
    'bert4rec': ('bert4rec_perdataset', 'BERT4REC'),
    'hstu': ('hstu_perdataset', 'HSTU'),
}
BASELINE = {'model': 'sasrec', 'hidden': 64, 'maxlen': 128, 'batch': 128,
            'dropout': .5, 'lr': .001, 'weight_decay_mode': 'zero',
            'sasrec_blocks': 2, 'sasrec_heads': 1}
SPACE = {
    'model': list(TEMPLATES), 'hidden': [32, 64, 128], 'maxlen': [32, 64, 128],
    'batch': [64, 128, 256], 'dropout': {'low': 0., 'high': .6},
    'lr': {'low': 1e-4, 'high': 1e-2, 'log': True},
    'weight_decay_mode': ['zero', 'positive'],
    'weight_decay': {'low': 1e-8, 'high': 1e-3, 'log': True},
    'sasrec_blocks': [1, 2, 3], 'sasrec_heads': [1, 2, 4],
    'gru4rec_layers': [1, 2, 3],
    'nextitnet_blocks': [2, 4, 8], 'nextitnet_kernel': [3, 5],
    'bert4rec_blocks': [1, 2, 3], 'bert4rec_heads': [1, 2, 4],
    'hstu_blocks': [1, 2, 3], 'hstu_heads': [1, 2, 4],
}
SAMPLER = {'n_startup_trials': 5, 'n_ei_candidates': 24,
           'multivariate': True, 'group': True, 'constant_liar': True}


def optuna_module():
    if str(RUNTIME) not in sys.path:
        sys.path.insert(0, str(RUNTIME))
    import optuna
    if optuna.__version__ != '4.6.0':
        raise RuntimeError('The preregistered Optuna 4.6.0 is required')
    return optuna


def suggest(trial, models):
    model = models[0] if len(models) == 1 else trial.suggest_categorical('model', models)
    params = {'model': model}
    for name in ['hidden', 'maxlen', 'batch']:
        params[name] = trial.suggest_categorical(name, SPACE[name])
    for name in ['dropout', 'lr']:
        params[name] = trial.suggest_float(name, **SPACE[name])
    mode = trial.suggest_categorical('weight_decay_mode', SPACE['weight_decay_mode'])
    params['weight_decay_mode'] = mode
    if mode == 'positive':
        params['weight_decay'] = trial.suggest_float('weight_decay', **SPACE['weight_decay'])
    keys = ['layers'] if model == 'gru4rec' else ['blocks', 'kernel'] if model == 'nextitnet' else ['blocks', 'heads']
    for key in keys:
        name = model + '_' + key
        params[name] = trial.suggest_categorical(name, SPACE[name])
    return params


def source(params):
    """Set public template hyperparameters; preserve every model/training function."""
    model = params['model']
    folder, prefix = TEMPLATES[model]
    original = (REPO / 'gagc/templates' / folder / 'train.py').read_text()
    values = {'HIDDEN': params['hidden'], 'MAXLEN': params['maxlen'], 'BATCH': params['batch'],
              'DROPOUT': params['dropout'], 'LR': params['lr'],
              'WD': params.get('weight_decay', 0.) if params['weight_decay_mode'] == 'positive' else 0.,
              'PATIENCE': 20, 'EVAL_EVERY': 5, 'EVAL_USERS': 200}
    for key in ['blocks', 'heads', 'layers', 'kernel']:
        name = model + '_' + key
        if name in params:
            values[key.upper()] = params[name]
    # The common worker's smoke mode changes SASREC_EPOCHS/BATCHES only.
    # Formal runs retain the identical 200-epoch, 200-batch ceiling for all models.
    injection = ['import os as _hpo_os',
        '_hpo_smoke = _hpo_os.environ.get("SASREC_EPOCHS") == "5" and _hpo_os.environ.get("SASREC_BATCHES") == "2"',
        '_hpo_epochs = "5" if _hpo_smoke else "200"',
        '_hpo_batches = "2" if _hpo_smoke else "200"',
        '_hpo_os.environ["GAGC_SEED"] = "42"']
    for name, value in values.items():
        rhs = repr(str(value))
        if name == 'BATCH':
            rhs = '"32" if _hpo_smoke else ' + rhs
        injection.append('_hpo_os.environ[' + repr(prefix + '_' + name) + '] = ' + rhs)
    injection += ['_hpo_os.environ[' + repr(prefix + '_EPOCHS') + '] = _hpo_epochs',
                  '_hpo_os.environ[' + repr(prefix + '_BATCHES') + '] = _hpo_batches']
    tree = ast.parse(original)
    position = 0
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            position = node.end_lineno
        elif isinstance(node, ast.ImportFrom) and node.module == '__future__':
            position = node.end_lineno
        else:
            break
    lines = original.splitlines(keepends=True)
    lines.insert(position, '\n# Frozen TPE parameter binding; model/training code below is unchanged.\n' + '\n'.join(injection) + '\n\n')
    result = ''.join(lines)
    if model == 'hstu':
        # This upstream template ships its predictor as a separate file.
        # Adapt its root to trial/working/predict.py, preserving the scoring code.
        predictor = (REPO / 'gagc/templates/hstu_perdataset/predict.py').read_text()
        old = '_HERE = os.path.dirname(os.path.abspath(__file__))'
        if predictor.count(old) != 1:
            raise RuntimeError('Pinned HSTU predictor layout changed')
        predictor = predictor.replace(old, '_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))')
        result += '\n\nif __name__ == "__main__":\n    from pathlib import Path as _HPOPath\n'
        result += '    (_HPOPath(__file__).parent / "working/predict.py").write_text(' + repr(predictor) + ')\n'
    ast.parse(result)
    return result


class TPEController(SingleSelectionMixin, Controller):
    def __init__(self, config):
        super().__init__(config, 'tpe')
        if read(self.ledger.path)['reservations']:
            raise RuntimeError('Unsettled GPU workers require review; budget is never reset')
        for path in self.root.glob('trials/*/receipt.json'):
            record = read(path)
            if record['status'] == 'success' and (self.best is None or record['score'] > self.best['score']):
                self.best = record

    def llm(self, *args, **kwargs):
        raise RuntimeError('Traditional TPE must never use an LLM')

    def environment(self, gpu):
        env = super().environment(gpu)
        for name in list(env):
            if any(name.startswith(prefix + '_') for _, prefix in TEMPLATES.values()):
                env.pop(name)
        for _, prefix in TEMPLATES.values():
            env[prefix + '_EPOCHS'] = '200'
        env.update(SASREC_BATCHES='200', GAGC_SEED='42')
        return env


def save_sampler(controller, sampler):
    path = controller.root / 'sampler.pkl'
    temporary = path.with_suffix('.pkl.tmp')
    temporary.write_bytes(pickle.dumps(sampler))
    temporary.replace(path)
    save(controller.root / 'sampler-state.json', {'sha256': sha(path)})


def study_for(controller):
    optuna = optuna_module()
    path = controller.root / 'sampler.pkl'
    if path.exists():
        if sha(path) != read(controller.root / 'sampler-state.json')['sha256']:
            raise RuntimeError('Sampler checkpoint changed')
        sampler = pickle.loads(path.read_bytes())
    else:
        sampler = optuna.samplers.TPESampler(seed=controller.c['sampler_seed'], **SAMPLER)
    study = optuna.create_study(study_name='amazon-tpe', direction='maximize', sampler=sampler,
        pruner=optuna.pruners.NopPruner(), storage='sqlite:///' + str(controller.root / 'study.sqlite3'),
        load_if_exists=True)
    if not study.trials:
        initial = dict(BASELINE)
        if len(controller.c.get('hpo_models', TEMPLATES)) == 1:
            initial.pop('model')
        study.enqueue_trial(initial)
    return study


def tell(study, number, receipt):
    optuna = optuna_module()
    study.set_user_attr('last_receipt', receipt['id'])
    if receipt['status'] == 'success':
        study.tell(number, receipt['score'])
    else:
        study.tell(number, state=optuna.trial.TrialState.FAIL)


def room(controller):
    state = read(controller.ledger.path)
    available = state['total_gpu_seconds'] - state['spent_gpu_seconds'] - sum(state['reservations'].values())
    slots = max(1, len(controller.c['gpus']) - len(state['reservations']))
    return available / slots - 2 >= controller.c['minimum_trial_seconds']


def run(config):
    c = TPEController(config)
    study = study_for(c)
    optuna = optuna_module()
    models = config['hpo_models']
    # Reconcile a process interruption after a receipt was committed, before tell().
    recover = []
    for frozen in study.get_trials(states=(optuna.trial.TrialState.RUNNING,)):
        key = frozen.user_attrs.get('candidate_id')
        if key is None:
            raise RuntimeError('Incomplete study proposal requires inspection')
        folder = c.root / 'trials' / key
        if (folder / 'receipt.json').exists():
            tell(study, frozen.number, read(folder / 'receipt.json'))
        elif folder.exists():
            raise RuntimeError('Incomplete GPU receipt requires accounting review')
        else:
            recover.append((optuna.trial.Trial(study, frozen._trial_id), frozen.params))
    save_sampler(c, study.sampler)

    def proposal():
        if recover:
            trial, params = recover.pop(0)
        else:
            trial = study.ask()
            params = suggest(trial, models)
        key = 'baseline' if trial.number == 0 else f'tpe-{trial.number:05d}'
        trial.set_user_attr('candidate_id', key)
        trial.set_user_attr('sampling_phase', 'baseline' if key == 'baseline' else
            'startup' if len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))) < SAMPLER['n_startup_trials'] else 'tpe')
        code = c.baseline() if key == 'baseline' else source(params)
        trial.set_user_attr('source_sha256', __import__('hashlib').sha256(code.encode()).hexdigest())
        save(c.root / 'proposals' / (key + '.json'), {'trial_number': trial.number, 'params': params,
            'candidate_id': key, 'source_sha256': trial.user_attrs['source_sha256']})
        save_sampler(c, study.sampler)
        return trial, key, code

    if not (c.root / 'trials/baseline/receipt.json').exists():
        trial, key, code = proposal()
        if key != 'baseline':
            raise RuntimeError('Every run must start with the common cold-start baseline')
        receipt = c.trial(key, code, config['gpus'][0], requested=config['trial_timeout_seconds'])
        tell(study, trial.number, receipt)
        save_sampler(c, study.sampler)
        if receipt['status'] != 'success':
            raise RuntimeError('Common baseline failed')

    pending = {}
    free = list(config['gpus'])
    with ThreadPoolExecutor(max_workers=len(free)) as pool:
        while pending or free and room(c):
            while free and room(c):
                gpu = free.pop(0)
                trial, key, code = proposal()
                future = pool.submit(c.trial, key, code, gpu)
                pending[future] = (gpu, trial.number, key)
                # Allow the execution thread to commit its reservation before ask().
                # Actual admission remains atomic in the existing shared ledger.
                import time
                while not future.done():
                    state = read(c.ledger.path)
                    if key in state['reservations'] or key in state['settled']:
                        break
                    time.sleep(.01)
            if not pending:
                break
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in sorted(completed, key=lambda f: pending[f][1]):
                gpu, number, key = pending.pop(future)
                free.append(gpu)
                try:
                    receipt = future.result()
                except BudgetExhausted:
                    study.tell(number, state=optuna.trial.TrialState.FAIL)
                else:
                    tell(study, number, receipt)
                save_sampler(c, study.sampler)
                save(c.root / 'progress.json', {'completed_study_trials': len(study.get_trials(states=(
                    optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL))),
                    'best_validation_hr10': c.best['score'] if c.best else None,
                    'budget': read(c.ledger.path), 'inflight': [v[2] for v in pending.values()]})
    result = c.finish()
    result.update(optimizer='Optuna TPESampler', optuna_version=optuna.__version__,
                  sampler_settings=SAMPLER, sampler_seed=config['sampler_seed'],
                  hpo_models=models, uses_llm=False, scope='predefined model and hyperparameter search')
    result['selected_parameters'] = read(c.root / 'proposals' / (result['selected']['id'] + '.json'))['params']
    result['sampling_phase_counts'] = {phase: sum(t.user_attrs.get('sampling_phase') == phase for t in study.trials)
                                      for phase in ['baseline', 'startup', 'tpe']}
    save(c.root / 'summary.json', result)
    return result
