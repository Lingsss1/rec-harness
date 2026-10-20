"""MLE-STAR control flow with Codex CLI and four parallel NextItNet trials.

Reference: google/adk-recipes 8c0862c103e36e1bb503a87bc1b134ddf6319d62,
python/agents/machine-learning-engineering (Apache-2.0). This is a task-specific
reimplementation, not the unmodified ADK runtime or a paper benchmark result.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
HYPERS = {'MAXLEN': 128, 'HIDDEN_UNITS': 64, 'NUM_BLOCKS': 8, 'KERNEL_SIZE': 3,
          'DROPOUT_RATE': 0.2, 'LR': 0.001, 'WEIGHT_DECAY': 0., 'BATCH_SIZE': 128}
CLASSES = {'NextItNetBlock', 'NextItNetModel'}
SCOPES = ['NextItNetBlock', 'NextItNetModel', 'hyperparameters', 'all']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def baseline_source(template):
    tree = ast.parse(Path(template).read_text())
    parts = ['import math\nimport torch\nimport torch.nn as nn\n']
    parts += [f'{k} = {v!r}' for k, v in HYPERS.items()]
    parts += [ast.unparse(n) for n in tree.body if isinstance(n, ast.ClassDef) and n.name in CLASSES]
    return '\n\n'.join(parts) + '\n'


def source_parts(source):
    tree = ast.parse(source)
    parts = {}
    allowed_imports = {'import math', 'import torch', 'import torch.nn as nn'}
    imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            statement = ast.unparse(node)
            if statement not in allowed_imports:
                raise ValueError('model imports are frozen')
            imports.append(statement)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name not in HYPERS or name in parts:
                raise ValueError('unexpected or duplicate model constant')
            value = ast.literal_eval(node.value)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('numeric model constants required')
            bounds = {'MAXLEN': (4, 512), 'HIDDEN_UNITS': (8, 256), 'NUM_BLOCKS': (1, 16),
                      'KERNEL_SIZE': (1, 9), 'DROPOUT_RATE': (0, .8), 'LR': (1e-5, .02),
                      'WEIGHT_DECAY': (0, .1), 'BATCH_SIZE': (16, 512)}
            if not bounds[name][0] <= value <= bounds[name][1]:
                raise ValueError(f'{name} outside resource bounds')
            if name in ('MAXLEN', 'HIDDEN_UNITS', 'NUM_BLOCKS', 'KERNEL_SIZE', 'BATCH_SIZE') and type(value) is not int:
                raise ValueError('integer shape constant required')
            parts[name] = ast.dump(node)
        elif isinstance(node, ast.ClassDef) and node.name in CLASSES:
            if node.name in parts or node.decorator_list or len(node.bases) != 1 or ast.unparse(node.bases[0]) != 'nn.Module':
                raise ValueError('unexpected model class definition')
            if any(not isinstance(n, ast.FunctionDef) for n in node.body):
                raise ValueError('class bodies must contain only methods')
            parts[node.name] = ast.dump(node)
        else:
            raise ValueError('only frozen imports, numeric constants and model classes allowed')
    if set(parts) != CLASSES | set(HYPERS) or set(imports) != allowed_imports or len(imports) != 3:
        raise ValueError('incomplete model source')
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and node not in tree.body:
            raise ValueError('nested imports not permitted')
        if isinstance(node, (ast.ImportFrom, ast.Global, ast.Nonlocal, ast.AsyncFunctionDef)):
            raise ValueError('unsupported model operation')
        if isinstance(node, ast.With) and any(ast.unparse(x.context_expr) != 'torch.no_grad()' for x in node.items):
            raise ValueError('only torch.no_grad context is allowed')
        if isinstance(node, ast.Attribute) and (node.attr.startswith('__') or node.attr in {
                'load', 'save', 'load_state_dict', 'read', 'write', 'open', 'system', 'popen',
                'from_file', 'frombuffer', 'set_default_device', 'set_num_threads'}):
            if node.attr != '__init__':
                raise ValueError('I/O or interpreter access in model')
        if isinstance(node, ast.Name) and node.id in {'os', 'sys', 'eval', 'exec', 'open', 'compile',
                '__import__', 'globals', 'locals', 'getattr', 'setattr', 'print', 'input', 'breakpoint'}:
            raise ValueError('I/O or dynamic interpreter access in model')
        if isinstance(node, ast.FunctionDef) and node.decorator_list:
            raise ValueError('decorators not permitted')
    return parts


def validate_source(source, parent, scope):
    before, after = source_parts(parent), source_parts(source)
    changed = {k for k in before if before[k] != after[k]}
    if not changed:
        raise ValueError('candidate has no structural change')
    allowed = set(HYPERS) if scope == 'hyperparameters' else set(before) if scope == 'all' else {scope}
    if not changed <= allowed:
        raise ValueError(f'edit escaped selected block {scope}: {sorted(changed)}')
    return sorted(changed)


def object_schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def array_schema(item):
    return {'type': 'array', 'items': item}


TEXT = {'type': 'string'}
CANDIDATES = object_schema({'candidates': array_schema(object_schema({
    'id': TEXT, 'scope': {'type': 'string', 'enum': SCOPES}, 'hypothesis': TEXT, 'source': TEXT}))})
SELECTION = object_schema({'selections': array_schema(object_schema({
    'trajectory': {'type': 'integer'}, 'scope': {'type': 'string', 'enum': SCOPES[:-1]}, 'reason': TEXT}))})
ENSEMBLE = object_schema({'plans': array_schema(object_schema({'hypothesis': TEXT,
    'transform': {'type': 'string', 'enum': ['raw', 'standardize', 'rank', 'softmax']},
    'weights': array_schema({'type': 'number'})}))})


class BudgetExhausted(Exception):
    pass


class Controller:
    def __init__(self, config):
        self.c = config
        self.root = Path(config['run_dir'])
        self.root.mkdir(parents=True, exist_ok=True)
        self.code = self.root / 'models'
        self.code.mkdir(exist_ok=True)
        self.codex_cwd = self.root / 'codex'
        self.codex_cwd.mkdir(exist_ok=True)

    def event(self, kind, **data):
        record = {'time': time.time(), 'kind': kind, **data}
        with (self.root / 'events.jsonl').open('a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def llm(self, key, instruction, context, schema, search=False):
        directory = self.root / 'calls' / key
        directory.mkdir(parents=True, exist_ok=True)
        prompt = ('You implement MLE-STAR for a frozen sequential-recommendation task. '
            'Task: optimize NextItNet on Movies_and_TV; training seed=2024. '
            'Keep the causal dilated-convolution NextItNet backbone. Do not replace it with another model. '
            'Only training interactions are inputs to learning; validation HR@10 selects models. '
            'The host owns data, BPR loss, Adam, epochs, sampling, clipping and evaluation. '
            'Model source may contain ONLY the three existing imports, eight numeric constants, '
            'and the two existing nn.Module classes (methods only); no file/network access or extra imports. '
            'Preserve constructor and forward/encode_sequence/score_candidates interfaces and causality. '
            'Do not run training, edit files or inspect the filesystem. '
            'Return only JSON conforming to the supplied schema.\n' + instruction + '\nCONTEXT:\n' +
            json.dumps(context, ensure_ascii=False))
        request = {'prompt': prompt, 'schema': schema, 'search': search}
        if (directory / 'request.json').exists() and read(directory / 'request.json') != request:
            raise RuntimeError('cached Codex request mismatch: ' + key)
        save(directory / 'request.json', request)
        if (directory / 'response.json').exists():
            return read(directory / 'response.json')
        if (directory / 'started.json').exists():
            raise RuntimeError('unfinished Codex call requires explicit review before retry: ' + key)
        save(directory / 'schema.json', schema)
        command = ['codex', 'exec', '--ephemeral', '--skip-git-repo-check', '--sandbox', 'read-only',
                   '-c', 'approval_policy="never"', '-c', 'web_search="live"' if search else 'web_search="disabled"',
                   '--json', '--output-schema', str(directory / 'schema.json'),
                   '-o', str(directory / 'last-message.txt'), '-C', str(self.codex_cwd), '-']
        save(directory / 'started.json', {'time': time.time(), 'command': command})
        self.event('codex_started', key=key)
        with (directory / 'events.jsonl').open('w') as out, (directory / 'stderr.log').open('w') as err:
            result = subprocess.run(command, input=prompt, text=True, stdout=out, stderr=err,
                                    timeout=1200, cwd=self.codex_cwd)
        if result.returncode != 0 or not (directory / 'last-message.txt').exists():
            raise RuntimeError('Codex transport failed: ' + str(directory))
        response = read(directory / 'last-message.txt')
        save(directory / 'response.json', response)
        self.event('codex_completed', key=key)
        return response

    def spent(self):
        charged = 0
        for directory in (self.root / 'trials').glob('*') if (self.root / 'trials').exists() else []:
            if (directory / 'receipt.json').exists():
                charged += read(directory / 'receipt.json')['allocated_gpu_seconds']
            elif (directory / 'request.json').exists():
                request = read(directory / 'request.json')
                if (directory / 'pid.json').exists():
                    pid = read(directory / 'pid.json')['pid']
                    if Path(f'/proc/{pid}').exists():
                        raise RuntimeError(f'unsettled worker pid {pid}; inspect before recovery')
                # After an interrupted controller, conservatively charge the full reservation.
                charged += request['timeout']
        return charged

    def job(self, key, model, gpu, timeout, mode='train', checkpoint=None):
        d = self.root / 'trials' / key
        d.mkdir(parents=True, exist_ok=True)
        if (d / 'receipt.json').exists():
            receipt = read(d / 'receipt.json')
            if receipt.get('source_sha256') != sha(model):
                raise RuntimeError('cached model source changed: ' + key)
            if receipt['status'] == 'success' and mode == 'train' and sha(d / 'checkpoint.pt') != receipt['checkpoint_sha256']:
                raise RuntimeError('cached checkpoint changed: ' + key)
            return receipt
        if (d / 'request.json').exists():
            # Interrupted jobs are retained as failures; never silently retrain.
            result = {'id': key, 'status': 'interrupted', 'source': str(model), 'source_sha256': sha(model),
                      'allocated_gpu_seconds': read(d / 'request.json')['timeout']}
            save(d / 'receipt.json', result)
            return result
        command = [self.c['python'], str(HERE / 'worker.py'), '--source', str(model),
            '--data', self.c['data_dir'], '--output', str(d), '--seed', str(self.c['seed']),
            '--epochs', str(self.c['epochs']), '--batches', str(self.c['batches']),
            '--eval-users', str(self.c['eval_users']), '--mode', mode]
        if checkpoint:
            command += ['--checkpoint', str(checkpoint)]
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '4',
               'MKL_NUM_THREADS': '4', 'PYTHONHASHSEED': str(self.c['seed']), 'GAGC_DATASETS': 'Movies_and_TV'}
        save(d / 'request.json', {'command': command, 'gpu': gpu, 'timeout': timeout,
                                 'source_sha256': sha(model), 'mode': mode})
        start = time.monotonic()
        self.event('trial_started', id=key, gpu=gpu, timeout=timeout)
        with (d / 'worker.log').open('w') as log:
            proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(d / 'pid.json', {'pid': proc.pid})
            try:
                code = proc.wait(timeout=timeout)
                status = 'success' if code == 0 and (d / 'result.json').exists() else 'crash'
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                status = 'timeout'
        receipt = {'id': key, 'status': status, 'source': str(model), 'source_sha256': sha(model),
            'allocated_gpu_seconds': time.monotonic() - start, 'gpu': gpu, 'output': str(d)}
        if status == 'success':
            receipt.update(read(d / 'result.json'))
        else:
            receipt['error'] = (d / 'worker.log').read_text()[-5000:]
        save(d / 'receipt.json', receipt)
        self.event('trial_completed', id=key, status=status, seconds=receipt['allocated_gpu_seconds'],
                   metrics=receipt.get('metrics'))
        return receipt

    def jobs(self, specs, final=False):
        results = []
        for offset in range(0, len(specs), len(self.c['gpus'])):
            wave = specs[offset:offset + len(self.c['gpus'])]
            pending = [s for s in wave if not (self.root / 'trials' / s['key'] / 'request.json').exists()]
            remaining = self.c['budget_gpu_seconds'] - self.spent() - (0 if final else self.c['final_reserve'])
            timeout = min(self.c['trial_timeout'], int(remaining / max(len(pending), 1)) - 2)
            if pending and timeout < (30 if final else self.c['minimum_trial_seconds']):
                raise BudgetExhausted('insufficient budget for another full trial wave')
            with ThreadPoolExecutor(max_workers=len(self.c['gpus'])) as pool:
                futures = [pool.submit(self.job, gpu=gpu, timeout=max(1, timeout), **spec)
                           for spec, gpu in zip(wave, self.c['gpus'])]
                results.extend(f.result() for f in futures)
        return results

    def brief(self, trial):
        return {k: trial[k] for k in ('id', 'status', 'metrics', 'hypothesis', 'scope', 'error') if k in trial}

    def propose(self, key, instruction, items, context=None, debug_remaining=2):
        response = self.llm(key, instruction + '\nGenerate exactly one candidate for every input id. '
            'Return the FULL model source, not a patch. For each candidate obey the specified scope. '
            'Do not change constants outside hyperparameters/all scope or methods outside the selected class.',
            {'items': items, 'evidence': context}, CANDIDATES)
        candidates = response['candidates']
        expected = {x['id']: x for x in items}
        if len(candidates) != len(expected) or {x['id'] for x in candidates} != set(expected):
            raise ValueError('candidate IDs do not match request')
        specs, info, rejected = [], {}, []
        for candidate in candidates:
            original = expected[candidate['id']]
            path = self.code / (candidate['id'] + '.py')
            try:
                if candidate['scope'] != original['scope']:
                    raise ValueError('candidate changed frozen scope')
                changes = validate_source(candidate['source'], original['source'], candidate['scope'])
                path.write_text(candidate['source'])
                save(path.with_suffix('.json'), {**candidate, 'changed_parts': changes})
                specs.append({'key': candidate['id'], 'model': path})
            except (ValueError, SyntaxError, TypeError) as exc:
                # Invalid model code is a recorded failed attempt, without a GPU launch.
                save(self.root / 'invalid' / (candidate['id'] + '.json'), {**candidate, 'error': str(exc)})
                self.event('candidate_rejected', id=candidate['id'], error=str(exc))
                rejected.append({'id': candidate['id'], 'status': 'invalid', 'error': str(exc)})
            info[candidate['id']] = candidate
        results = self.jobs(specs)
        for result in results:
            result.update({k: info[result['id']][k] for k in ('hypothesis', 'scope')})
        failed = rejected + [r for r in results if r['status'] == 'crash']
        if failed and debug_remaining > 0:
            repaired_items = []
            for failure in failed:
                original = expected[failure['id']]
                repaired_items.append({**original, 'id': f'debug{debug_remaining}-' + failure['id'],
                    'failed_source': info[failure['id']]['source'], 'error': failure['error']})
            results.extend(self.propose(key + '-debug',
                'Debug these failed attempts using the concrete error evidence. Preserve each original '
                'scope and proposal intent; compare changes against the original source, not failed_source.',
                repaired_items, debug_remaining=debug_remaining - 1))
        return results


def hr(trial):
    return trial.get('metrics', {}).get('HR@10', -1) if trial.get('status') == 'success' else -1


def blend(arrays, plan):
    import numpy as np
    weights = np.asarray(plan['weights'], dtype=float)
    if weights.shape != (len(arrays),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError('ensemble requires finite nonnegative member weights')
    weights /= weights.sum()
    transformed = []
    for scores in arrays:
        if plan['transform'] == 'standardize':
            scores = (scores - scores.mean(1, keepdims=True)) / scores.std(1, keepdims=True).clip(1e-8)
        elif plan['transform'] == 'rank':
            scores = np.asarray([np.searchsorted(np.sort(row), row, side='left') for row in scores], dtype=float)
        elif plan['transform'] == 'softmax':
            scores = np.exp(scores - scores.max(1, keepdims=True))
            scores /= scores.sum(1, keepdims=True)
        elif plan['transform'] != 'raw':
            raise ValueError('unknown transform')
        transformed.append(scores)
    return sum(w * x for w, x in zip(weights, transformed))


def retrieve(controller, source):
    retrieval = controller.llm('retrieval',
        'Use web search to retrieve task-specific NextItNet improvements and corresponding reference code. '
        'Find four distinct candidate directions compatible with the frozen NextItNet backbone. '
        'Audit readout: input sequences are RIGHT aligned but baseline score_candidates indexes '
        'count_nonzero-1, which can read padding. Include correcting this bug among candidate '
        'directions and distinguish bug-fix gains from architectural gains. '
        'Include primary-source URLs and concrete implementation details. '
        'Do not claim you searched unless actual web tool results are available.',
        {'baseline': source},
        object_schema({'directions': array_schema(object_schema({'description': TEXT,
            'urls': array_schema(TEXT), 'implementation': TEXT})), 'search_performed': {'type': 'boolean'}}), search=True)
    if not retrieval['search_performed']:
        raise RuntimeError('retrieval did not perform web search; cannot label run MLE-STAR')
    if not retrieval['directions'] or any(not x['urls'] for x in retrieval['directions']):
        raise RuntimeError('retrieval returned no cited candidate directions')
    return retrieval


def search(controller):
    c = controller.c
    baseline = controller.code / 'baseline.py'
    source = baseline_source(Path(c['repo']) / 'gagc/templates/nextitnet_perdataset/train.py')
    if baseline.exists() and baseline.read_text() != source:
        raise RuntimeError('baseline source changed')
    baseline.write_text(source)
    source_parts(source)
    retrieval = retrieve(controller, source)
    initial = controller.jobs([{'key': 'baseline', 'model': baseline}])[0]
    if hr(initial) < 0:
        raise RuntimeError('baseline failed; inspect ' + initial.get('output', ''))
    best = [initial.copy() for _ in c['gpus']]
    histories = [[] for _ in best]
    targets_history = [[] for _ in best]
    all_trials = [initial]
    try:
        # Official initialization: retrieve/evaluate candidate models, then
        # rank and try to merge the lower candidate into the better solution.
        for k in range(c['initial_candidates']):
            items = [{'id': f'init-{k}-t{t}', 'scope': 'all', 'source': source,
                      'direction': retrieval['directions'][t % len(retrieval['directions'])],
                      'history': [controller.brief(x) for x in histories[t]]} for t in range(len(best))]
            results = controller.propose(f'init-{k}', 'Generate a search-informed initial NextItNet solution. '
                'All initial solutions must correct the known RIGHT-aligned readout bug (last position '
                'rather than count_nonzero-1). Apply the retrieved direction on top of that correction. '
                'For direction zero, the first candidate should be the pure readout fix; subsequent '
                'candidates can explore a different causal readout. Distinguish bug-fix and architectural '
                'changes in the hypothesis. Use a distinct alternative to prior candidates for this trajectory.', items)
            for result in results:
                t = int(result['id'].rsplit('t', 1)[1])
                histories[t].append(result)
                if hr(result) > hr(best[t]):
                    best[t] = result
            all_trials.extend(results)
        merge_items = []
        for t in range(len(best)):
            other = [r for r in histories[t] if hr(r) >= 0 and r['id'] != best[t]['id']]
            if other:
                alternative = max(other, key=hr)
                merge_items.append({'id': f'merge-t{t}', 'scope': 'all',
                    'source': Path(best[t]['source']).read_text(),
                    'alternative_source': Path(alternative['source']).read_text(),
                    'results': [controller.brief(best[t]), controller.brief(alternative)]})
        if merge_items:
            merged = controller.propose('merge-initial', 'Merge complementary useful components of the two '
                'candidate solutions into one NextItNet model, as MLE-STAR initialization does.', merge_items)
            all_trials.extend(merged)
            for result in merged:
                t = int(result['id'].rsplit('t', 1)[1])
                if hr(result) > hr(best[t]):
                    best[t] = result
        for outer in range(c['outer_rounds']):
            parents = [x.copy() for x in best]
            ablations = []
            for ab in range(c['ablations_per_round']):
                items = [{'id': f'ab-{outer}-{ab}-t{t}', 'scope': 'all',
                    'source': Path(parents[t]['source']).read_text(),
                    'previous_ablations': [controller.brief(x) for x in ablations if x['id'].endswith(f't{t}')],
                    'previous_target_blocks': targets_history[t]} for t in range(len(best))]
                results = controller.propose(f'ablation-{outer}-{ab}',
                    'Perform a simple genuine ablation: remove/disable exactly one existing model component '
                    'and change nothing unrelated. Select a different component from previous ablations. '
                    'The host retrains from scratch; these are component contributions, not new improvements.', items)
                ablations.extend(results)
            # Ablations are measured but are not promoted automatically.
            selections = controller.llm(f'select-block-{outer}',
                'Summarize the real ablation effects and choose exactly one target block for each trajectory. '
                'Prefer the component whose removal most reduces validation HR@10. Use previous block history '
                'to explore other components in later outer iterations. The selected scope must be one class '
                'or hyperparameters; the inner loop may edit ONLY this selected block.',
                {'parents': [controller.brief(x) for x in parents],
                 'ablations': [controller.brief(x) for x in ablations], 'previous_blocks': targets_history}, SELECTION)
            selected = selections['selections']
            if len(selected) != len(best) or {x['trajectory'] for x in selected} != set(range(len(best))):
                raise ValueError('invalid outer-loop target selection')
            scopes = {x['trajectory']: x['scope'] for x in selected}
            inner_history = [[] for _ in best]
            for inner in range(c['inner_rounds']):
                items = [{'id': f'refine-{outer}-{inner}-t{t}', 'scope': scopes[t],
                    'source': Path(parents[t]['source']).read_text(),
                    'ablation_results': [controller.brief(x) for x in ablations if x['id'].endswith(f't{t}')],
                    'previous_plans_and_scores': inner_history[t]} for t in range(len(best))]
                results = controller.propose(f'refine-{outer}-{inner}',
                    'Propose and implement a targeted refinement plan for the selected block. '
                    'Use ablation evidence and feedback from previous inner-loop experiments. '
                    'Each alternative is based on the unchanged outer-loop parent. Do not edit another block.', items)
                all_trials.extend(results)
                for result in results:
                    t = int(result['id'].rsplit('t', 1)[1])
                    inner_history[t].append(controller.brief(result))
                    if hr(result) > hr(best[t]):
                        best[t] = result
            for t in range(len(best)):
                targets_history[t].append(scopes[t])
            save(controller.root / f'outer-{outer}.json', {'best': best, 'selections': selected,
                'ablations': ablations, 'inner_history': inner_history})
    except BudgetExhausted as exc:
        controller.event('search_budget_stop', reason=str(exc))
    return initial, best, all_trials


def finalize(controller, baseline, best):
    import numpy as np
    from mle_star_codex.worker import metrics
    members = list({x['id']: x for x in best + [baseline]}.values())
    hashes = {x['rows_sha256'] for x in members}
    if len(hashes) != 1:
        raise RuntimeError('ensemble validation rows differ')
    arrays = [np.load(Path(x['output']) / 'scores.npy') for x in members]
    chosen = {'hypothesis': 'best single model', 'transform': 'raw',
              'weights': [float(i == max(range(len(members)), key=lambda j: hr(members[j]))) for i in range(len(members))]}
    score = metrics(blend(arrays, chosen))
    attempts = []
    if len(members) > 1:
        for r in range(controller.c['ensemble_rounds']):
            response = controller.llm(f'ensemble-{r}',
                'Propose up to four ensemble strategies based on member validation scores and previous feedback. '
                'Weights follow the supplied member order; require nonnegative weights with positive sum. '
                'Use raw, per-query standardize, per-query softmax, or per-query rank scores. '
                'Use previous attempts to refine the strategy. Test labels and test metrics are unavailable.',
                {'members': [controller.brief(x) for x in members], 'previous_attempts': attempts}, ENSEMBLE)
            if len(response['plans']) > 4:
                raise ValueError('too many ensemble plans')
            for plan in response['plans']:
                result = metrics(blend(arrays, plan))
                attempts.append({'plan': plan, 'metrics': result})
                if result['HR@10'] > score['HR@10']:
                    chosen, score = plan, result
    save(controller.root / 'selection.json', {'members': members, 'plan': chosen, 'validation': score,
                                            'ensemble_attempts': attempts})
    # Freeze selection before the first held-out test read. Include baseline for
    # reporting; no test observations are returned to the search controller/LLM.
    indices = [i for i, w in enumerate(chosen['weights']) if w > 0]
    baseline_index = next(i for i, m in enumerate(members) if m['id'] == baseline['id'])
    needed = sorted(set(indices + [baseline_index]))
    specs = [{'key': f'final-{i}', 'model': Path(members[i]['source']), 'mode': 'test',
              'checkpoint': Path(members[i]['output']) / 'checkpoint.pt'} for i in needed]
    tests = controller.jobs(specs, final=True)
    if any(t['status'] != 'success' for t in tests):
        raise RuntimeError('final evaluation failed; frozen selection is retained')
    if len({t['rows_sha256'] for t in tests}) != 1:
        raise RuntimeError('test candidate rows differ')
    test_map = {i: np.load(Path(t['output']) / 'scores.npy') for i, t in zip(needed, tests)}
    selected_plan = {**chosen, 'weights': [chosen['weights'][i] for i in indices]}
    final_metrics = metrics(blend([test_map[i] for i in indices], selected_plan))
    report = {'status': 'complete', 'method': 'MLE-STAR Codex CLI adaptation',
        'seed': controller.c['seed'], 'gpus': controller.c['gpus'],
        'baseline_validation': baseline['metrics'], 'selected_validation': score,
        'baseline_test': metrics(test_map[baseline_index]), 'selected_test': final_metrics,
        'allocated_gpu_seconds': controller.spent(), 'selected_plan': chosen,
        'members': [m['id'] for m in members], 'limitations': [
            'single training seed; no claim of statistical significance',
            'NextItNet-constrained adaptation, not the original MLE-bench protocol',
            'four parallel solutions and bounded host-owned score ensembles']}
    report['readout_audit'] = {m['id']: read(Path(m['output']) / 'correctness-after-training.json') for m in members}
    save(controller.root / 'summary.json', report)
    controller.event('complete', summary=report)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    c = read(a.config)
    if c['seed'] != 2024 or len(c['gpus']) != 4 or len(set(c['gpus'])) != 4:
        raise ValueError('this experiment requires seed=2024 and four distinct GPU IDs')
    if c['budget_gpu_seconds'] <= c['final_reserve'] or c['final_reserve'] <= 0:
        raise ValueError('invalid total/final inference budget')
    controller = Controller(c)
    with (controller.root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = {'config': c, 'code_hashes': {p.name: sha(p) for p in HERE.glob('*.py')},
            'data_hashes': {p.name: sha(p) for p in (Path(c['data_dir']) / 'trainval').glob('*') if p.suffix in ('.txt', '.json')},
            'heldout_test_sha256': sha(Path(c['data_dir']) / 'test/Movies_and_TV_test.txt'),
            'references': {'rec_harness': '74a927fefaf7d120c52b6e3a0809958bd33984f0',
                           'official_mle_star': '8c0862c103e36e1bb503a87bc1b134ddf6319d62'},
            'template_sha256': sha(Path(c['repo']) / 'gagc/templates/nextitnet_perdataset/train.py')}
        freeze_path = controller.root / 'frozen.json'
        if freeze_path.exists() and read(freeze_path) != frozen:
            raise RuntimeError('frozen experiment code/config/data changed')
        save(freeze_path, frozen)
        if (controller.root / 'summary.json').exists():
            print(json.dumps(read(controller.root / 'summary.json'), indent=2))
            return
        if (controller.root / 'search-result.json').exists():
            result = read(controller.root / 'search-result.json')
        else:
            baseline, best, trials = search(controller)
            result = {'baseline': baseline, 'best': best, 'trials': trials}
            save(controller.root / 'search-result.json', result)
        finalize(controller, result['baseline'], result['best'])


if __name__ == '__main__':
    main()
