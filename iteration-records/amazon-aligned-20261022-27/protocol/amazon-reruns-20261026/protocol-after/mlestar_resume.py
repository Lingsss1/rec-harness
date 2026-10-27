"""Audited continuation of the interrupted MLE-STAR run, retaining its ledger."""
import argparse
import copy
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .bootstrap import bootstrap
bootstrap()
from . import mlestar
from .core import Controller, TASK, TEXT, read, save, schema
from .launch import manifest
from .prepare import ROOT, WORKSPACE, sha


def identities_match(response, items):
    candidates = response.get('candidates', [])
    return (len(candidates) == len(items)
            and {c['id'] for c in candidates} == {i['id'] for i in items})


class ResumeController(Controller):
    def __init__(self, config, method):
        super().__init__(config, method)
        if read(self.ledger.path)['reservations']:
            raise RuntimeError('Unsettled workers must be reviewed before continuation')

    def replay(self, receipt):
        if receipt['status'] == 'success' and (self.best is None or receipt['score'] > self.best['score']):
            self.best = receipt
        self.event('trial_replayed', id=receipt['id'], original_gpu_seconds=receipt['allocated_gpu_seconds'])
        return receipt

    def trial(self, key, source, gpu, requested=None, files=None, smoke=False):
        directory = self.root / 'trials' / key
        if directory.exists():
            receipt_path = directory / 'receipt.json'
            if not receipt_path.exists():
                raise RuntimeError('Incomplete existing trial requires review: ' + key)
            if (directory / 'train.py').read_text() != source:
                raise RuntimeError('Cached trial source changed: ' + key)
            if files or smoke:
                raise RuntimeError('Unexpected extra files/smoke in MLE-STAR replay')
            return self.replay(read(receipt_path))
        return super().trial(key, source, gpu, requested, files, smoke)

    def wave(self, proposals):
        cached, pending = [], []
        for proposal in proposals:
            directory = self.root / 'trials' / proposal['id']
            if directory.exists():
                cached.append(self.trial(proposal['id'], proposal['source'], self.c['gpus'][0]))
            else:
                pending.append(proposal)
        fresh = super().wave(pending) if pending else []
        receipts = {r['id']: r for r in cached + fresh}
        return [receipts[p['id']] for p in proposals]

    def cached_call(self, key, instruction, output_schema, search):
        directory = self.root / 'calls' / key
        if not directory.exists():
            return None
        request = read(directory / 'request.json')
        expected = TASK + '\nMETHOD: ' + self.method + '\n' + instruction
        if output_schema is None:
            expected += '\nReturn the requested completion verbatim inside the JSON text field.'
        if request['prompt'] != expected or request['search'] != search or request['schema'] != output_schema:
            raise RuntimeError('Cached LLM request changed: ' + key)
        if not (directory / 'response.json').exists():
            raise RuntimeError('Incomplete LLM response requires review: ' + key)
        self.event('llm_replayed', call=key)
        return read(directory / 'response.json')

    def llm(self, instruction, output_schema=None, search=False, key=None):
        if key is None:
            raise RuntimeError('MLE-STAR replay requires stable call names')
        response = self.cached_call(key, instruction, output_schema, search)
        if output_schema != mlestar.CANDIDATES:
            return response if response is not None else super().llm(instruction, output_schema, search, key)
        prefix, context = instruction.rsplit('\nCONTEXT:\n', 1)
        items = json.loads(context)
        if response is not None and identities_match(response, items):
            return response
        # A fixed object with a one-value ID enum cannot silently omit or rename
        # trajectories. Each trajectory retains its original context/evidence.
        candidates = []
        for item in items:
            single_schema = schema({'id': {'type': 'string', 'enum': [item['id']]},
                                    'hypothesis': TEXT, 'source': TEXT})
            single_prompt = (prefix + '\nGenerate the complete program for ONLY this trajectory. '
                             'Preserve its exact id. Other trajectories are generated separately. '
                             'Return one object matching the provided schema.\nCONTEXT:\n'
                             + json.dumps(item))
            single_key = 'repair-' + key + '-' + item['id']
            candidate = self.cached_call(single_key, single_prompt, single_schema, search)
            if candidate is None:
                candidate = super().llm(single_prompt, single_schema, search, single_key)
            candidates.append(candidate)
        result = {'candidates': candidates}
        if not identities_match(result, items):
            raise RuntimeError('Fixed-identity generation violated its schema')
        save(self.root / 'recovery' / (key + '.response.json'), result)
        return result


def execute(config):
    root = Path(config['root'])
    method_root = root / 'mlestar'
    recovery = method_root / 'recovery'
    lock = (method_root / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (method_root / 'summary.json').exists():
        raise RuntimeError('MLE-STAR is already complete')
    registration = read(recovery / 'registration.json')
    if manifest(config) != registration['files']:
        raise RuntimeError('Registered recovery inputs changed')
    budget = read(method_root / 'budget.json')
    if sha(method_root / 'budget.json') != registration['budget_sha256'] or budget['reservations']:
        raise RuntimeError('Budget changed before continuation')
    os.sched_setaffinity(0, set(config['cpu_cores']))
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True)
    devices = {int(row.split(',')[0]): [int(x.strip()) for x in row.split(',')[1:]]
               for row in inventory.strip().splitlines()}
    if any(devices[g][0] >= 512 or devices[g][1] >= 5 for g in config['gpus']):
        raise RuntimeError('The four registered GPUs are occupied')
    old_queue = read(root / 'queue-status.json')
    save(recovery / 'original-queue-status.json', old_queue)
    save(root / 'queue-status.json', {'status': 'running', 'method': 'mlestar', 'recovery': True,
         'completed': [r for r in old_queue['completed'] if r['method'] != 'mlestar'],
         'gpu_ids': config['gpus'], 'updated_at': time.time()})
    save(recovery / 'status.json', {'status': 'running', 'prior_spent_gpu_seconds': budget['spent_gpu_seconds'],
                                   'budget_total_gpu_seconds': budget['total_gpu_seconds'], 'started_at': time.time()})
    mlestar.Controller = ResumeController
    try:
        result = mlestar.run(config)
    except BaseException:
        save(recovery / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
        save(recovery / 'status.json', {'status': 'failed', 'time': time.time()})
        save(root / 'queue-status.json', {'status': 'finished_with_failures', 'recovery': True,
             'completed': old_queue['completed'], 'updated_at': time.time()})
        raise
    save(recovery / 'status.json', {'status': 'complete', 'test_score': result['test']['score'],
                                   'finished_at': time.time()})
    save(root / 'queue-status.json', {'status': 'complete', 'recovery': True,
         'completed': [({'method': 'mlestar', 'status': 'complete', 'exit_code': 0,
                         'prior_interruption_retained': True} if r['method'] == 'mlestar' else r)
                       for r in old_queue['completed']], 'updated_at': time.time()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    config = read(ROOT / 'config.json')
    logging.basicConfig(level=logging.INFO)
    if args.worker:
        return execute(config)
    recovery = ROOT / 'mlestar/recovery'
    if (recovery / 'registration.json').exists():
        raise RuntimeError('Existing recovery registration; inspect before retrying')
    frozen = read(ROOT / 'frozen.json')
    changed = [name for name, value in frozen['files'].items()
               if not (WORKSPACE / name).exists() or sha(WORKSPACE / name) != value]
    if changed:
        raise RuntimeError('Original registered files changed: ' + repr(changed))
    if (ROOT / 'mlestar/summary.json').exists():
        raise RuntimeError('MLE-STAR is already complete')
    save(recovery / 'registration.json', {'files': manifest(config),
         'original_frozen_sha256': sha(ROOT / 'frozen.json'),
         'budget_sha256': sha(ROOT / 'mlestar/budget.json'), 'registered_at': time.time(),
         'repair': 'Replay cached calls and receipts; generate each new candidate with fixed ID enum; original ledger retained'})
    command = [sys.executable, '-u', '-m', 'amazon_aligned.mlestar_resume', '--worker']
    with (recovery / 'controller.log').open('a') as log:
        proc = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL, stdout=log,
             stderr=subprocess.STDOUT, start_new_session=True,
             env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
    save(recovery / 'launcher.json', {'pid': proc.pid, 'command': command, 'started_at': time.time()})
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError('Recovery exited immediately; inspect controller.log')
    print(json.dumps({'pid': proc.pid, 'root': str(recovery)}))


if __name__ == '__main__':
    main()
