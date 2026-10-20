"""Matched-call autoresearch control using the unchanged MLE-STAR GPU worker.

Policy reference: karpathy/autoresearch, program.md, MIT license. The original
repository trains nanochat; this adapter substitutes the existing NextItNet task
and uses identical frozen training/scoring to the preceding comparison arm.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from mle_star_codex.run import (
    CANDIDATES, Controller, BudgetExhausted, baseline_source, hr, read, save, sha,
)
from mle_star_codex import run as shared

HERE = Path(__file__).resolve().parent


class AutoresearchController(Controller):
    def llm(self, key, instruction, context, schema, search=False):
        if search:
            raise ValueError('this autoresearch comparison has no web retrieval stage')
        directory = self.root / 'calls' / key
        directory.mkdir(parents=True, exist_ok=True)
        prompt = ('Follow the task-specific autoresearch program below. It takes precedence over '
            'the upstream nanochat example. Do not run training, inspect files, edit files, or use tools. '
            'Return only JSON conforming to the supplied schema.\nTASK PROGRAM:\n' +
            (HERE / 'program.md').read_text() + '\nUPSTREAM POLICY REFERENCE:\n' +
            Path(self.c['upstream_program']).read_text() + '\nCURRENT STEP:\n' + instruction +
            '\nCONTEXT:\n' + json.dumps(context, ensure_ascii=False))
        request = {'prompt': prompt, 'schema': schema, 'search': False}
        if (directory / 'request.json').exists() and read(directory / 'request.json') != request:
            raise RuntimeError('cached autoresearch request mismatch: ' + key)
        save(directory / 'request.json', request)
        if (directory / 'response.json').exists():
            return read(directory / 'response.json')
        if (directory / 'started.json').exists():
            raise RuntimeError('unfinished Codex call requires explicit review before retry: ' + key)
        save(directory / 'schema.json', schema)
        command = ['codex', 'exec', '--ephemeral', '--skip-git-repo-check', '--sandbox', 'read-only',
                   '-c', 'approval_policy="never"', '-c', 'web_search="disabled"',
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


def keep_or_discard(parent, candidate):
    if candidate.get('status') != 'success':
        return parent, 'crash'
    if hr(candidate) > hr(parent):
        return candidate, 'keep'
    return parent, 'discard'


def choose_single(incumbents):
    # Stable tie breaking by trajectory order; no test result enters selection.
    return max(incumbents, key=hr)


def write_results(controller, baseline, rounds):
    header = 'source_sha256\tvalidation_hr10\tstatus\ttrajectory\titeration\tparent\tdescription\n'
    rows = [header, f'{baseline["source_sha256"]}\t{hr(baseline):.6f}\tkeep\tshared\t0\t-\tbaseline\n']
    for round_record in rounds:
        for d in round_record['decisions']:
            trial = d['trial']
            description = trial.get('hypothesis', trial.get('error', 'failed attempt')).replace('\t', ' ').replace('\n', ' ')
            rows.append(f'{trial.get("source_sha256", "-")}\t{hr(trial):.6f}\t{d["decision"]}\t{d["trajectory"]}\t'
                        f'{round_record["iteration"]}\t{d["parent_id"]}\t{description}\n')
    path = controller.root / 'results.tsv'
    temp = path.with_suffix('.tmp')
    temp.write_text(''.join(rows))
    temp.replace(path)


def run_search(controller):
    c = controller.c
    source = baseline_source(Path(c['repo']) / 'gagc/templates/nextitnet_perdataset/train.py')
    baseline_path = controller.code / 'baseline.py'
    if baseline_path.exists() and baseline_path.read_text() != source:
        raise RuntimeError('raw baseline source changed')
    baseline_path.write_text(source)
    baseline = controller.jobs([{'key': 'baseline', 'model': baseline_path}])[0]
    if hr(baseline) < 0:
        raise RuntimeError('autoresearch baseline failed; inspect its worker log')
    incumbents = [baseline.copy() for _ in c['gpus']]
    histories = [[] for _ in incumbents]
    rounds = []
    write_results(controller, baseline, rounds)
    for iteration in range(1, c['iterations_per_trajectory'] + 1):
        key = f'iteration-{iteration:02d}'
        items = []
        for t, parent in enumerate(incumbents):
            item = {'id': f'{key}-t{t}', 'scope': 'all',
                'source': Path(parent['source']).read_text(), 'parent_id': parent['id'],
                'parent_validation': parent['metrics'], 'trajectory': t, 'iteration': iteration,
                'previous_attempts': histories[t]}
            if histories[t] and histories[t][-1]['status'] != 'success':
                failed = controller.code / (histories[t][-1]['id'] + '.py')
                if failed.exists():
                    item['last_failed_source'] = failed.read_text()
            items.append(item)
        try:
            # The inherited generator/AST validator is neutral with this llm()
            # override. No MLE-STAR prompt or retrieval is used. GPU retries are
            # disabled: a crashed training consumes one of the 44 attempt slots.
            trials = controller.propose(key,
                'Perform the next autoresearch edit-run-evaluate step for each trajectory. '
                'Propose one coherent full-file experiment per item based on its incumbent and TSV-like '
                'history. Scope is all: you may edit any permitted model block or bounded constant. '
                'Improve validation HR@10 while retaining simplicity. There is no ablation requirement. '
                'If an earlier attempt crashed, repair it or move to another idea within this same slot.',
                items, debug_remaining=0)
        except BudgetExhausted as exc:
            controller.event('search_budget_stop', reason=str(exc))
            break
        by_id = {trial['id']: trial for trial in trials}
        decisions = []
        for t, parent in enumerate(incumbents):
            trial_id = f'{key}-t{t}'
            trial = by_id.get(trial_id)
            if trial is None:
                invalid = read(controller.root / 'invalid' / (trial_id + '.json'))
                trial = {'id': trial_id, 'status': 'invalid', 'error': invalid['error'],
                         'hypothesis': invalid['hypothesis']}
            updated, decision = keep_or_discard(parent, trial)
            decisions.append({'trajectory': t, 'parent_id': parent['id'], 'decision': decision,
                              'incumbent_id': updated['id'], 'trial': trial})
            histories[t].append({**controller.brief(trial), 'decision': decision, 'parent_id': parent['id']})
            incumbents[t] = updated
        record = {'iteration': iteration, 'decisions': decisions, 'incumbents': incumbents.copy(),
                  'settled_gpu_seconds': controller.spent()}
        save(controller.root / 'rounds' / (key + '.json'), record)
        rounds.append(record)
        write_results(controller, baseline, rounds)
        controller.event('iteration_completed', iteration=iteration,
            candidates=[{'id': d['trial']['id'], 'decision': d['decision'], 'hr10': hr(d['trial'])} for d in decisions],
            incumbents=[{'id': x['id'], 'hr10': hr(x)} for x in incumbents])
    selected = choose_single(incumbents)
    return {'baseline': baseline, 'incumbents': incumbents, 'selected': selected,
            'completed_iterations_per_trajectory': len(rounds),
            'candidate_attempt_slots': sum(len(r['decisions']) for r in rounds)}


def reference_training_calls(reference):
    receipts = []
    for path in (reference / 'trials').glob('*/receipt.json'):
        if read(path.parent / 'request.json')['mode'] == 'train' and path.parent.name != 'baseline':
            receipts.append(path)
    return len(receipts)


def reference_single(reference):
    selection = read(reference / 'selection.json')
    member = max(selection['members'], key=hr)
    matches = []
    for path in (reference / 'trials').glob('final-*/receipt.json'):
        receipt = read(path)
        request = read(path.parent / 'request.json')
        if receipt['source_sha256'] == member['source_sha256'] and request['mode'] == 'test':
            # Also bind the checkpoint, avoiding a match to another initialization.
            argv = request['command']
            checkpoint = argv[argv.index('--checkpoint') + 1]
            if sha(checkpoint) == member['checkpoint_sha256']:
                matches.append(receipt)
    if len(matches) != 1:
        raise RuntimeError('reference best single-model test receipt is missing or ambiguous')
    return member, matches[0]


def finalize(controller, search_result):
    chosen, baseline = search_result['selected'], search_result['baseline']
    selection = {'method': 'autoresearch greedy single-model selection', 'selected': chosen,
        'baseline': baseline, 'incumbents': search_result['incumbents'],
        'candidate_attempt_slots': search_result['candidate_attempt_slots'],
        'completed_iterations_per_trajectory': search_result['completed_iterations_per_trajectory']}
    selection_path = controller.root / 'selection.json'
    if selection_path.exists() and read(selection_path) != selection:
        raise RuntimeError('frozen final selection changed')
    save(selection_path, selection)
    specs = [{'key': 'final-selected', 'model': Path(chosen['source']), 'mode': 'test',
              'checkpoint': Path(chosen['output']) / 'checkpoint.pt'}]
    if chosen['id'] != baseline['id']:
        specs.append({'key': 'final-baseline', 'model': Path(baseline['source']), 'mode': 'test',
                      'checkpoint': Path(baseline['output']) / 'checkpoint.pt'})
    results = controller.jobs(specs, final=True)
    if any(x['status'] != 'success' for x in results):
        raise RuntimeError('final single-model evaluation failed; selection remains frozen')
    selected_test, baseline_test = results[0], results[-1]
    if selected_test['rows_sha256'] != baseline_test['rows_sha256']:
        raise RuntimeError('selected and baseline test rows differ')
    reference = Path(controller.c['reference_run'])
    # Only now, after all search and final selection, inspect the old outcomes.
    ref_single, ref_single_test = reference_single(reference)
    ref_summary = read(reference / 'summary.json')
    if selected_test['rows_sha256'] != ref_single_test['rows_sha256']:
        raise RuntimeError('autoresearch and MLE-STAR test candidate rows differ')
    if chosen['rows_sha256'] != ref_single['rows_sha256']:
        raise RuntimeError('autoresearch and MLE-STAR validation candidate rows differ')
    all_receipts = [read(p) for p in (controller.root / 'trials').glob('*/receipt.json')]
    train_count = sum(read(Path(x['output']) / 'request.json')['mode'] == 'train'
                      for x in all_receipts if x.get('output') and x['id'] != 'baseline')
    report = {'status': 'complete', 'method': 'Karpathy autoresearch Codex CLI / NextItNet adaptation',
        'seed': controller.c['seed'], 'gpus': controller.c['gpus'],
        'candidate_attempt_slots': search_result['candidate_attempt_slots'],
        'candidate_training_calls': train_count,
        'reference_candidate_training_calls': reference_training_calls(reference),
        'matched_training_call_count': train_count == reference_training_calls(reference),
        'completed_iterations_per_trajectory': search_result['completed_iterations_per_trajectory'],
        'selected_id': chosen['id'], 'baseline_validation': baseline['metrics'],
        'selected_validation': chosen['metrics'], 'baseline_test': baseline_test['metrics'],
        'selected_test': selected_test['metrics'], 'allocated_gpu_seconds': controller.spent(),
        'mle_star_single': {'id': ref_single['id'], 'validation': ref_single['metrics'],
                            'test': ref_single_test['metrics']},
        'mle_star_ensemble_supplementary': {'validation': ref_summary['selected_validation'],
                                          'test': ref_summary['selected_test']},
        'readout_audit': read(Path(chosen['output']) / 'correctness-after-training.json'),
        'limitations': ['one training seed; not a statistical significance result',
            'matched training-call count, not matched actual GPU time or LLM tokens',
            'task-specific adaptation with frozen common trainer rather than original nanochat five-minute training',
            'greedy parent chains with four batched proposals per Codex call; no ensemble']}
    save(controller.root / 'summary.json', report)
    save(controller.root / 'comparison.json', {'primary': {
        'autoresearch_single': {'validation': chosen['metrics'], 'test': selected_test['metrics']},
        'mle_star_single': report['mle_star_single']},
        'supplementary_mle_star_ensemble': report['mle_star_ensemble_supplementary'],
        'matched_training_call_count': report['matched_training_call_count']})
    controller.event('complete', summary=report)


def freeze(controller):
    c = controller.c
    reference = Path(c['reference_run'])
    prior = read(reference / 'frozen.json')
    keys = ('seed', 'gpus', 'budget_gpu_seconds', 'final_reserve', 'trial_timeout',
            'minimum_trial_seconds', 'epochs', 'batches', 'eval_users')
    if any(c[k] != prior['config'][k] for k in keys):
        raise ValueError('training/resource settings differ from the MLE-STAR reference')
    count = reference_training_calls(reference)
    if count != len(c['gpus']) * c['iterations_per_trajectory']:
        raise ValueError('candidate training-call allowance is not matched')
    if sha(Path(shared.__file__)) != prior['code_hashes']['run.py'] or sha(HERE.parent / 'mle_star_codex/worker.py') != prior['code_hashes']['worker.py']:
        raise ValueError('shared training/controller implementation changed since reference run')
    data_hashes = {name: sha(Path(c['data_dir']) / 'trainval' / name) for name in prior['data_hashes']}
    if data_hashes != prior['data_hashes'] or sha(Path(c['data_dir']) / 'test/Movies_and_TV_test.txt') != prior['heldout_test_sha256']:
        raise ValueError('comparison data differ from reference')
    template = Path(c['repo']) / 'gagc/templates/nextitnet_perdataset/train.py'
    if sha(template) != prior['template_sha256']:
        raise ValueError('raw baseline template differs from reference')
    upstream = Path(c['upstream_program']).parent
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    frozen = {'config': c, 'reference_frozen_sha256': sha(reference / 'frozen.json'),
        'data_hashes': data_hashes, 'heldout_test_sha256': prior['heldout_test_sha256'],
        'template_sha256': sha(template), 'shared_controller_sha256': sha(shared.__file__),
        'shared_worker_sha256': sha(HERE.parent / 'mle_star_codex/worker.py'),
        'code_hashes': {p.name: sha(p) for p in HERE.glob('*.py')},
        'task_program_sha256': sha(HERE / 'program.md'), 'upstream_commit': revision,
        'upstream_program_sha256': sha(c['upstream_program']), 'candidate_training_call_allowance': count}
    path = controller.root / 'frozen.json'
    if path.exists() and read(path) != frozen:
        raise RuntimeError('frozen comparison code/config/data changed')
    save(path, frozen)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    controller = AutoresearchController(read(a.config))
    with (controller.root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze(controller)
        if (controller.root / 'summary.json').exists():
            print(json.dumps(read(controller.root / 'summary.json'), indent=2))
            return
        path = controller.root / 'search-result.json'
        if path.exists():
            result = read(path)
        else:
            result = run_search(controller)
            save(path, result)
        finalize(controller, result)


if __name__ == '__main__':
    main()
