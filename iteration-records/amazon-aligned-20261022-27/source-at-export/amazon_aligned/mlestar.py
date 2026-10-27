"""MLE-STAR retrieval, merging, ablation/refinement, and ensemble task adapter."""
import ast
import json
from pathlib import Path

from .core import BudgetExhausted, Controller, TEXT, schema, save

CANDIDATES = schema({'candidates': {'type': 'array', 'items': schema({
    'id': TEXT, 'hypothesis': TEXT, 'source': TEXT})}})


def score(result):
    return result['score'] if result['status'] == 'success' else -1.


def source(result):
    return (Path(result['output']) / 'train.py').read_text()


def validate_block(before, after, block):
    def parts(text):
        return {node.name: ast.dump(node) for node in ast.parse(text).body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
    left, right = parts(before), parts(after)
    if block != 'hyperparameters' and block not in left:
        raise ValueError('Selected block is absent from parent: ' + block)
    for key in set(left) | set(right):
        if key != block and left.get(key) != right.get(key):
            raise ValueError('MLE-STAR refinement escaped selected block: ' + key)
    if block == 'hyperparameters':
        names = {'HIDDEN_UNITS', 'NUM_BLOCKS', 'NUM_HEADS', 'DROPOUT_RATE', 'LR', 'WEIGHT_DECAY',
                 'BATCH_SIZE', 'NUM_BATCHES', 'NUM_EPOCHS', 'MAXLEN', 'PATIENCE'}
        def frozen(text):
            result = []
            for node in ast.parse(text).body:
                if isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id in names for t in node.targets):
                    continue
                result.append(ast.dump(node))
            return result
        if frozen(before) != frozen(after):
            raise ValueError('Hyperparameter refinement changed non-hyperparameter statements')
    if block != 'hyperparameters':
        def remaining(text):
            return [ast.dump(n) for n in ast.parse(text).body
                    if not isinstance(n, (ast.ClassDef, ast.FunctionDef))]
        if remaining(before) != remaining(after):
            raise ValueError('MLE-STAR refinement changed constants outside selected block')


def propose(controller, key, instruction, items):
    response = controller.llm(instruction + '\nCONTEXT:\n' + json.dumps(items), CANDIDATES, key=key)
    candidates = response['candidates']
    if {c['id'] for c in candidates} != {i['id'] for i in items} or len(candidates) != len(items):
        raise ValueError('MLE-STAR candidate identities changed')
    lookup = {i['id']: i for i in items}
    valid, invalid = [], []
    for candidate in candidates:
        try:
            ast.parse(candidate['source'])
            item = lookup[candidate['id']]
            if item.get('block'):
                validate_block(item['source'], candidate['source'], item['block'])
        except (ValueError, SyntaxError) as exc:
            invalid.append({'id': candidate['id'], 'status': 'invalid_source', 'score': None,
                            'error': str(exc), 'allocated_gpu_seconds': 0})
        else:
            valid.append(candidate)
    save(controller.root / (key + '.proposals.json'), response)
    return (controller.wave(valid) if valid else []) + invalid


def ensemble_source(members, weights, transform):
    paths = [str(Path(m['output']) / 'working/predict.py') for m in members]
    return '''import importlib.util
import numpy as np
_models = []
for i, path in enumerate(PATHS):
    spec = importlib.util.spec_from_file_location('ensemble_member_' + str(i), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _models.append(module.predict)
def predict(user_id, history, candidates, *, dataset_name=None):
    values = np.asarray([fn(user_id, history, candidates, dataset_name=dataset_name) for fn in _models], dtype=float)
    if TRANSFORM == 'standardize':
        values = (values - values.mean(axis=1, keepdims=True)) / (values.std(axis=1, keepdims=True) + 1e-8)
    elif TRANSFORM == 'rank':
        values = values.argsort(axis=1).argsort(axis=1).astype(float)
    return np.average(values, axis=0, weights=WEIGHTS).tolist()
'''.replace('PATHS', repr(paths)).replace('TRANSFORM', repr(transform)).replace('WEIGHTS', repr(weights))


def run(config):
    controller = Controller(config, 'mlestar')
    baseline = controller.trial('baseline', controller.baseline(), config['gpus'][0],
                                requested=config['trial_timeout_seconds'])
    if baseline['status'] != 'success':
        raise RuntimeError('MLE-STAR common baseline failed')
    retrieval = controller.llm(
        'MLE-STAR initialization: use actual web search to retrieve four different promising '
        'recommendation model/training improvements and primary-source reference implementations. '
        'Start from the supplied SASRec cold start; any architecture change must retain the common task. '
        'Return concrete directions with cited primary-source URLs.\nBaseline:\n' + controller.baseline(),
        schema({'directions': {'type': 'array', 'items': schema({'description': TEXT,
                         'urls': {'type': 'array', 'items': TEXT}, 'implementation': TEXT})}}),
        search=True, key='retrieval')
    if len(retrieval['directions']) < 4 or any(not x['urls'] for x in retrieval['directions']):
        raise RuntimeError('MLE-STAR retrieval must provide four cited reference directions')
    best, histories = [baseline] * 4, [[] for _ in range(4)]
    try:
        for k in range(2):
            items = [{'id': f'initial-{k}-t{t}', 'source': controller.baseline(),
                      'direction': retrieval['directions'][t],
                      'previous': [controller.brief(r) for r in histories[t]]} for t in range(4)]
            results = propose(controller, f'initial-{k}',
                              'Generate a distinct search-informed initial solution for each trajectory. '
                              'Return complete executable train.py including its predictor writer.', items)
            for result in results:
                t = int(result['id'].rsplit('t', 1)[1])
                histories[t].append(result)
                if score(result) > score(best[t]):
                    best[t] = result
        items = []
        for t in range(4):
            alternatives = [r for r in histories[t] if score(r) >= 0 and r['id'] != best[t]['id']]
            if alternatives:
                other = max(alternatives, key=score)
                items.append({'id': f'merge-t{t}', 'source': source(best[t]),
                              'alternative_source': source(other),
                              'results': [controller.brief(best[t]), controller.brief(other)]})
        if items:
            for result in propose(controller, 'merge', 'Merge complementary useful components of the '
                                  'two initial solutions into one training program per trajectory.', items):
                t = int(result['id'].rsplit('t', 1)[1])
                if score(result) > score(best[t]):
                    best[t] = result
        for outer in range(1000):
            if controller.ledger.available() < 1800:
                break
            parents = list(best)
            ablations = []
            for ab in range(2):
                items = [{'id': f'ab-{outer}-{ab}-t{t}', 'source': source(parents[t]),
                          'previous_ablations': [controller.brief(r) for r in ablations]} for t in range(4)]
                ablations += propose(controller, f'ablation-{outer}-{ab}',
                                     'Remove or disable exactly one existing model/training component. '
                                     'Do not add a new improvement. Retraining measures its contribution.', items)
            selection = controller.llm('Use the real retraining ablations to select exactly one '
                'important top-level class or function block per trajectory for targeted refinement. '
                'You may instead select hyperparameters. Return its exact Python symbol name.\n' +
                json.dumps({'parents': [controller.brief(r) for r in parents],
                            'ablations': [controller.brief(r) for r in ablations],
                            'source': [source(r) for r in parents]}),
                schema({'selections': {'type': 'array', 'items': schema({
                    'trajectory': {'type': 'integer'}, 'block': TEXT, 'reason': TEXT})}}),
                key=f'select-{outer}')
            choices = selection['selections']
            if len(choices) != 4 or {r['trajectory'] for r in choices} != set(range(4)):
                raise ValueError('Invalid MLE-STAR block selection')
            blocks = {r['trajectory']: r['block'] for r in choices}
            feedback = []
            for inner in range(2):
                items = [{'id': f'refine-{outer}-{inner}-t{t}', 'source': source(parents[t]),
                          'block': blocks[t], 'ablation_evidence': [controller.brief(r) for r in ablations],
                          'previous_refinements': feedback} for t in range(4)]
                results = propose(controller, f'refine-{outer}-{inner}',
                                  'Refine ONLY the selected top-level block, using ablation evidence '
                                  'and earlier inner-loop feedback. Keep every other block unchanged.', items)
                feedback += [controller.brief(r) for r in results]
                for result in results:
                    t = int(result['id'].rsplit('t', 1)[1])
                    if score(result) > score(best[t]):
                        best[t] = result
            save(controller.root / f'outer-{outer}.json', {'blocks': blocks, 'feedback': feedback,
                                                          'best': [controller.brief(r) for r in best]})
    except BudgetExhausted as exc:
        controller.event('search_budget_stop', reason=str(exc))
    members = list({r['id']: r for r in best if score(r) >= 0}.values())
    # Ablations measure component importance; they are not automatically promoted.
    controller.best = max(members, key=score)
    if len(members) > 1 and controller.ledger.available() > 30:
        plans = controller.llm('MLE-STAR ensemble stage: propose up to four nonnegative weighted '
            'score ensembles of the supplied independently trained trajectories. Use raw, standardize, '
            'or rank transforms. Selection still uses only validation HR@10.\n' +
            json.dumps([controller.brief(r) for r in members]),
            schema({'plans': {'type': 'array', 'items': schema({'weights': {'type': 'array',
                    'items': {'type': 'number'}}, 'transform': {'type': 'string',
                    'enum': ['raw', 'standardize', 'rank']}})}}), key='ensemble')
        for i, plan in enumerate(plans['plans'][:4]):
            weights = plan['weights']
            if len(weights) != len(members) or any(w < 0 for w in weights) or sum(weights) <= 0:
                raise ValueError('Invalid MLE-STAR ensemble weights')
            try:
                controller.validation_job(f'ensemble-{i}', ensemble_source(members, weights, plan['transform']),
                                          config['gpus'][0])
            except BudgetExhausted:
                break
    return controller.finish()
