"""Use the isolated jump-v2 fork with the existing aligned Amazon executor.

No experiment starts unless --run is explicitly supplied. --check-config only
validates provenance and fork imports. Output roots must be new directories.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import hashlib
import logging
import os
from pathlib import Path
import sys
import subprocess
import time
import traceback

FORK = Path(__file__).resolve().parent
WORKSPACE = FORK.parents[1]
sys.path.insert(0, str(WORKSPACE))
from amazon_aligned.bootstrap import bootstrap
bootstrap()
# Change bindings only inside this new process, before the common adapter import.
from amazon_aligned import prepare
prepare.REPO = FORK
sys.path.insert(0, str(FORK))
from amazon_aligned.core import BudgetExhausted, read, save
from amazon_aligned.recharness import RecHarnessController, native, digest


class JumpController(RecHarnessController):
    def group(self, specs, diagnostics=None):
        original_limit = self.c['trial_timeout_seconds']
        caps = [float(s['max_trial_gpu_seconds']) for s in specs if s.get('max_trial_gpu_seconds') is not None]
        if caps:
            self.c['trial_timeout_seconds'] = min(original_limit, min(caps))
        try:
            return super().group(specs, diagnostics)
        finally:
            self.c['trial_timeout_seconds'] = original_limit

    def run(self):
        self.setup()
        diagnostics = {}
        for _ in range(1000):
            previous = self.store.state()
            specs = json.loads(native.propose_action_group(previous.to_json(), json.dumps(diagnostics)))
            if not specs:
                self.event('search_budget_stop', reason='no budget-safe proposal')
                break
            try:
                results = self.group(specs, diagnostics)
            except BudgetExhausted as exc:
                self.event('search_budget_stop', reason=str(exc))
                break
            valid = [r for r in results if native._is_valid_trial(r)]
            diagnostics = max(valid, key=lambda r: r['val_score']) if valid else results[0]
            if not previous.baseline_done and not valid:
                raise RuntimeError('Common baseline failed')
            if self.ledger.available() < self.c['minimum_trial_seconds'] + 3:
                break
        finalized = json.loads(native.finalize_jump_search('matched_budget_stop'))
        if not finalized['ok']:
            raise RuntimeError(finalized['error'])
        selected = digest((self.workspace / 'best.py').read_text())
        if selected not in self.promotion_receipts:
            raise RuntimeError('Final incumbent has no matching trained artifact')
        global_best = self.best
        self.best = self.promotion_receipts[selected]
        history = self.store.state().score_history
        if history and self.best['score'] < max(history) - 1e-12:
            raise RuntimeError('Final model is worse than the retained global validation incumbent')
        save(self.root / 'native/final-selection.json', {'policy': 'safe incumbent after final jump resolution',
            'selected': self.brief(self.best), 'global_best': self.brief(global_best),
            'jump_finalization': finalized, 'state': json.loads(self.store.state().to_json())})
        return self.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--jump', choices=['on', 'off'], default='on')
    args = parser.parse_args()
    config = deepcopy(read(args.config))
    if Path(native.__file__).resolve() != FORK / 'gagc/tools.py':
        raise RuntimeError('Original native package was imported instead of jump-v2')
    os.environ['GAGC_JUMP_ENABLED'] = 'true' if args.jump == 'on' else 'false'
    if not args.run:
        print(json.dumps({'checked': True, 'native_source': native.__file__, 'jump': args.jump,
                          'budget_gpu_seconds': config['search_budget_gpu_seconds'], 'started': False}))
        return
    if args.output is None:
        parser.error('--run requires --output pointing to a new directory')
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError('Output must be a new independent experiment directory')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True)
    devices = {int(row.split(',')[0]): [int(v.strip()) for v in row.split(',')[1:]]
               for row in inventory.strip().splitlines()}
    if any(devices[g][0] >= 512 or devices[g][1] >= 5 for g in config['gpus']):
        raise RuntimeError('The configured four GPUs must be idle before this experiment starts')
    output.mkdir(parents=True)
    config['root'] = str(output)
    config['methods'] = ['recharness']
    config['jump_fork'] = {'source': str(FORK), 'enabled': args.jump == 'on', 'version': 'jump-v2'}
    config['jump_fork']['policy'] = read(FORK / 'JUMP_POLICY.json')
    save(output / 'config.json', config)
    paths = [p for p in FORK.rglob('*.py') if '__pycache__' not in p.parts]
    paths += list((WORKSPACE / 'amazon_aligned').glob('*.py'))
    paths += [FORK / 'JUMP_POLICY.json', output / 'config.json', Path(config['data_dir']) / 'prepared.json']
    save(output / 'frozen.json', {'registered_at': time.time(), 'files': {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}})
    lock = (output / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, set(config['cpu_cores']))
    logging.basicConfig(level=logging.INFO)
    save(output / 'status.json', {'status': 'running', 'started_at': time.time()})
    try:
        result = JumpController(config).run()
    except BaseException:
        save(output / 'failure.json', {'error': traceback.format_exc(), 'time': time.time()})
        save(output / 'status.json', {'status': 'failed'})
        raise
    save(output / 'status.json', {'status': 'complete', 'test_hr10': result['test']['score']})


if __name__ == '__main__':
    main()
