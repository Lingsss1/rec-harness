"""Pinned-data preparation and six matched 24-hour TAAC research runs.

Run with this repository's .venv Python. Preparation/calibration is outside the
research budget; training, agents, retrieval, failures and final evaluation are
inside each run's 86400-second limit. No research runs consume final scores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback

from recevobench.taac import loop
from recevobench.taac.common import file_hash, read, write

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
BASE = REPO / 'examples/recommendation_autoresearch'
DEFAULT_ROOT = REPO / 'output/taac-auto-vs-enhanced-20261104'
SEEDS = [42, 43, 44]
BUDGET = 86400


def now():
    return datetime.now(timezone.utc).isoformat()


def cli(root, name, args, timeout):
    with (root / (name + '.log')).open('a') as log:
        process = subprocess.run([sys.executable, '-u', '-m', 'recevobench.taac', *args],
                                 cwd=REPO, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, timeout=timeout)
    if process.returncode:
        raise RuntimeError(f'{name} failed: inspect {root / (name + ".log")}')


def comparison(root):
    rows = []
    for mode in ['auto', 'enhanced']:
        for seed in SEEDS:
            study = root / mode / f'seed-{seed}'
            if not (study / 'state.json').exists():
                continue
            state = read(study / 'state.json')
            incumbent = next((t for t in state['trials'] if t['id'] == state['incumbent']), None)
            usage = []
            for path in (study / 'agents').glob('*/model-usage.json'):
                usage.extend(read(path)['calls'])
            row = {
                'mode': mode, 'seed': seed,
                'status': read(study / 'worker.json') if (study / 'worker.json').exists() else {'phase': 'initialized'},
                'trials': len(state['trials']),
                'valid_trials': sum(t['score'] is not None for t in state['trials']),
                'failures': sum(t['score'] is None for t in state['trials']),
                'feedback_baseline': state['trials'][0]['score'] if state['trials'] else None,
                'feedback_best': incumbent['score'] if incumbent else None,
                'selected': state.get('selected'),
                'spent_seconds': state['spent_seconds'], 'budget_seconds': BUDGET,
                'llm_calls': len(usage),
                'tokens': {k: sum(u.get(k, 0) for u in usage) for k in ['input_tokens', 'cached_input_tokens', 'cache_write_input_tokens', 'output_tokens', 'reasoning_output_tokens']},
                'usd_cost': None,
            }
            if (study / 'final.json').exists():
                final = read(study / 'final.json')
                first = state['trials'][0]['id']
                row['holdout'] = final
                row['final_baseline'] = final['results'][first]['final']['score']
                row['final_selected'] = final['results'][final['selected']]['final']['score']
                row['final_gain'] = row['final_selected'] - row['final_baseline']
            rows.append(row)
    paired = []
    for seed in SEEDS:
        a = next((r for r in rows if r['seed'] == seed and r['mode'] == 'auto'), None)
        e = next((r for r in rows if r['seed'] == seed and r['mode'] == 'enhanced'), None)
        if a and e and 'final_selected' in a and 'final_selected' in e:
            paired.append({'seed':seed, 'enhanced_minus_auto':e['final_selected']-a['final_selected']})
    summaries = {}
    for mode in ['auto', 'enhanced']:
        complete = [r for r in rows if r['mode'] == mode and 'final_selected' in r]
        summaries[mode] = {'completed_seeds':len(complete)}
        if complete:
            for key in ['final_baseline', 'final_selected', 'final_gain']:
                values = [r[key] for r in complete]
                summaries[mode][key] = {'mean':statistics.mean(values), 'sample_stddev':statistics.stdev(values) if len(values)>1 else None}
    result = {'updated_at':now(), 'model':'gpt-6.1-sol', 'effort':'high',
              'budget_seconds_per_seed_and_method':BUDGET,
              'scope':'TencentGR-1M local frozen holdout; not original competition leaderboard',
              'runs':rows, 'summary':summaries, 'paired_final_differences':paired,
              'mean_paired_final_difference':statistics.mean(p['enhanced_minus_auto'] for p in paired) if paired else None,
              'limitations':'Three paired research trajectories. No claim of statistical significance. External knowledge remains unverified. USD cost unknown without billing receipts.'}
    write(root / 'comparison.json', result)
    columns = ['mode','seed','trials','failures','feedback_baseline','feedback_best','final_baseline','final_selected','final_gain','spent_seconds','llm_calls']
    (root / 'comparison.tsv').write_text('\t'.join(columns)+'\n'+'\n'.join('\t'.join(str(r.get(k,'')) for k in columns) for r in rows)+'\n')
    return result


def worker(study):
    study = Path(study).resolve()
    # A separate persistent worker owns every run; no two controllers can write it.
    with (study / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read(study / 'config.json')
        budget = read(study.parents[1] / 'budget.json')
        deadline = budget['start_unix'] + BUDGET
        final_reserve = budget['final_reserve_seconds']
        def status(phase, **extra):
            write(study / 'worker.json', {'phase':phase, 'pid':os.getpid(), 'at':now(),
                  'deadline_utc':datetime.fromtimestamp(deadline, timezone.utc).isoformat(), **extra})
        status('baseline')
        if not read(study / 'state.json')['trials']:
            result = loop.baseline(study)
            if result['score'] is None:
                status('failed', error='Baseline failed; inspect trials/t0000')
                return 1
        errors = 0
        while True:
            state = read(study / 'state.json')
            if state['closed']:
                status('completed' if (study / 'final.json').exists() else 'final_failed')
                return 0 if (study / 'final.json').exists() else 1
            research_remaining = min(deadline-time.time(), BUDGET-state['spent_seconds']) - final_reserve
            # The frozen runner reserves a whole trial. Never knowingly begin a
            # round that could consume the final-evaluation reserve.
            phases = 3 if config['enhanced'] else 1
            worst_round = config['limits']['trial_seconds'] + phases*config['limits']['agent_seconds'] + (config['limits'].get('retrieval_seconds', 0) if config['enhanced'] else 0) + 60
            if (study / 'STOP').exists() or research_remaining < worst_round:
                break
            status('research', completed_trials=len(state['trials']), research_remaining=research_remaining)
            try:
                loop.run(study, rounds=1)
                errors = 0
            except Exception as error:
                errors += 1
                with (study / 'controller-errors.jsonl').open('a') as stream:
                    stream.write(json.dumps({'at':now(),'error':str(error),'traceback':traceback.format_exc(),'consecutive':errors})+'\n')
                if read(study / 'state.json')['pending']:
                    loop.recover(study)
                if errors >= 3:
                    status('research_stopped', error=str(error))
                    break
        status('finalizing')
        try:
            loop.finish(study)
            loop.stage_memory(study)
            state = read(study / 'state.json')
            status('completed', trials=len(state['trials']), selected=state['selected'], spent_seconds=state['spent_seconds'], elapsed_seconds=time.time()-budget['start_unix'])
            return 0
        except Exception as error:
            status('final_failed', error=str(error), traceback=traceback.format_exc())
            return 1


def suite(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'suite.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(phase, **extra):
            write(root / 'suite.json', {'phase':phase, 'pid':os.getpid(), 'at':now(), **extra})
        try:
            status('waiting_for_verified_downloads')
            while not (root / 'assets/downloads-verified.json').exists():
                if (root / 'STOP').exists():
                    status('stopped_before_research')
                    return 0
                process = read(root / 'download-process.json')['pid']
                if Path(f'/proc/{process}/stat').exists():
                    if Path(f'/proc/{process}/stat').read_text().split()[2] == 'Z':
                        raise RuntimeError('Download worker exited before verification; inspect download.log')
                else:
                    raise RuntimeError('Download worker missing before verification; inspect download.log')
                time.sleep(10)
            assets = root / 'assets'
            prepare_config = {
                'raw_root':str(assets/'raw'),
                'release':'TAAC2025/TencentGR-1M@4be8735e2a7e9fc11fe7eb2a5e1e06e27c57e453',
                'indexer_sha256':file_hash(assets/'raw/indexer.pkl'), 'fixture':False,
                'split_salt':'taac-local-v1', 'fractions':[0.85,0.05,0.05,0.05],
                'cut_fraction':0.8, 'history_limit':100, 'mm_emb_id':'81',
            }
            write(root/'prepare.json', prepare_config)
            prepared = root / 'prepared'
            status('waiting_for_prepared_data')
            with (root/'prepare.lock').open('a') as prepare_lock:
                fcntl.flock(prepare_lock, fcntl.LOCK_EX)
                if not (prepared/'manifest.json').exists():
                    status('preparing_official_data')
                    cli(root, 'prepare', ['prepare','--config',str(root/'prepare.json'),'--output',str(prepared)], 10800)
            status('verifying_prepared_data')
            cli(root,'verify',['verify','--data',str(prepared)],3600)
            meta = read(prepared/'public/metadata.json')
            if meta['fixture'] or meta['n_candidates'] != 660000:
                raise ValueError('Real data and full candidate pool required')
            write(root/'environment.json', {'python':sys.version, 'executable':sys.executable,
                  'packages':{p:importlib.metadata.version(p) for p in ['torch','numpy','pyarrow','transformers','tokenizers','safetensors','jsonschema','recevobench']},
                  'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                  'controller_identity':loop.controller_identity(),
                  'launcher_hashes':{p.name:file_hash(p) for p in SCRIPTS.glob('*.py')},
                  'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total','--format=csv'],text=True)})
            (root/'source.patch').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=REPO))
            snapshot = root/'source-at-launch'
            shutil.copytree(SCRIPTS,snapshot/'scripts/taac24',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copytree(REPO/'src/recevobench/taac',snapshot/'src/recevobench/taac',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
            (root/'requirements-frozen.txt').write_text(subprocess.check_output(['/root/miniconda3/bin/uv','pip','freeze','--python',sys.executable],text=True))
            status('checking_runtime_and_agent')
            tested_identity = {'controller':loop.controller_identity(),'python':sys.executable}
            tested = root/'runtime-tests-passed.json'
            if not tested.exists() or read(tested) != tested_identity:
                with (root/'runtime-tests.log').open('w') as log:
                    check = subprocess.run([sys.executable,'-m','pytest','tests/taac','-q'],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=900)
                if check.returncode:
                    raise RuntimeError('Runtime regression tests failed; inspect runtime-tests.log')
                write(tested,tested_identity)
            integration = root/'agent-integration-preflight'
            if not (integration/'passed.json').exists():
                with (integration/'run.log').open('a') as log:
                    check = subprocess.run([sys.executable,'-u',str(integration/'run.py')],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                if check.returncode:
                    raise RuntimeError('Real coding-agent integration failed; inspect agent-integration-preflight/run.log')
            write(root/'preflight-passed.json', {'at':now(),'runtime_tests':True,'real_coding_agent':True,'fixture_preflight_only':True})
            calibration = root/'calibration'
            calibration.mkdir(exist_ok=True)
            write(calibration/'protocol.json', {'epochs':3,'batch_size':2048,'max_steps':5,'cpu_threads':4})
            status('calibrating_real_gpu_training')
            if not (calibration/'train/training.json').exists():
                cli(root,'calibrate-train',['train','--public',str(prepared/'public'), '--candidate',str(BASE/'baseline/candidate.py'), '--output',str(calibration/'train'), '--protocol',str(calibration/'protocol.json'), '--device','cuda:3','--seed','42'], 7200)
            # Only public feedback queries enter this performance probe.
            if not (calibration/'feedback/inference.json').exists():
                with (prepared/'public/feedback.jsonl').open() as source:
                    (calibration/'queries.jsonl').write_text(''.join(line for _,line in zip(range(128), source)))
                status('calibrating_full_catalog_inference')
                cli(root,'calibrate-predict',['predict','--public',str(prepared/'public'),'--candidate',str(BASE/'baseline/candidate.py'),'--checkpoint',str(calibration/'train/checkpoint.pt'),'--queries',str(calibration/'queries.jsonl'),'--output',str(calibration/'feedback'),'--device','cuda:3'],3600)
            train = read(calibration/'train/training.json')
            if train['device'] != 'cuda:3' or not train['peak_vram_bytes']:
                raise ValueError('GPU calibration must report memory on its assigned nonzero device')
            inference = read(calibration/'feedback/inference.json')
            n_train = meta['counts']['train_examples']
            projected_train = train['training_seconds']/train['steps'] * math.ceil(n_train/2048)*3
            n_queries = sum(1 for _ in (calibration/'queries.jsonl').open())
            projected_inference = inference['seconds']/n_queries * max(meta['counts'][p+'_queries'] for p in ['feedback','gate','final'])
            # Calibrated upper bounds permit complete paired runs and reserve
            # four full inference passes (seed+selected, gate+final) per study.
            trial_seconds = math.ceil(max(1800, (projected_train+projected_inference)*1.5+300))
            final_reserve = math.ceil(max(1800, projected_inference*6+600))
            if trial_seconds + 3*1200 + final_reserve >= BUDGET/2:
                raise RuntimeError('Calibration leaves insufficient room for baseline plus evolution in 24h; inspect calibration before changing the frozen protocol')
            write(root/'calibration.json', {'fixture':False,'not_a_benchmark_result':True,'projected_train_seconds':projected_train,'projected_inference_seconds':projected_inference,'trial_seconds':trial_seconds,'final_reserve_seconds':final_reserve})
            status('initializing_matched_studies')
            configs = root/'configs'
            configs.mkdir(exist_ok=True)
            for mode, devices in [('auto',[0,1,2]), ('enhanced',[4,5,6])]:
                for seed, gpu in zip(SEEDS,devices):
                    config = read(BASE/'study.example.json')
                    config.update(data=str(prepared),candidate=str(BASE/'baseline/candidate.py'),program=str(BASE/'program.md'),knowledge=str(BASE/'knowledge.jsonl'),device=f'cuda:{gpu}',seeds=[seed],enhanced=mode=='enhanced',agent={'argv':[sys.executable,str(SCRIPTS/'codex_agent.py'),'{request}']})
                    config['limits'].update(total_seconds=BUDGET,trial_seconds=trial_seconds,agent_seconds=1200,max_trials=1000000,retrieval_seconds=180)
                    config['retrieval']['encoder']['model_path'] = str(assets/'encoder')
                    config['train'] = {'epochs':3,'batch_size':2048,'max_steps':None,'cpu_threads':4}
                    path = configs/f'{mode}-{seed}.json'
                    write(path,config)
                    study = root/mode/f'seed-{seed}'
                    if not study.exists():
                        loop.initialize(path,study)
            start = time.time()
            write(root/'budget.json', {'start_unix':start,'start_utc':now(),'deadline_utc':datetime.fromtimestamp(start+BUDGET,timezone.utc).isoformat(),'total_seconds':BUDGET,'final_reserve_seconds':final_reserve,'preparation_excluded':True,'early_stop_rule':'Do not begin a round unless its maximum reservation plus the final-evaluation reserve fits. This may finish before 24h.'})
            processes = []
            for mode in ['auto','enhanced']:
                for seed in SEEDS:
                    study = root/mode/f'seed-{seed}'
                    with (study/'controller.log').open('a') as log:
                        p = subprocess.Popen([sys.executable,'-u',str(__file__),'worker','--study',str(study)],cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    processes.append((mode,seed,p))
            status('running', workers=[{'mode':m,'seed':s,'pid':p.pid} for m,s,p in processes],start_utc=read(root/'budget.json')['start_utc'],deadline_utc=read(root/'budget.json')['deadline_utc'])
            while any(p.poll() is None for _,_,p in processes):
                comparison(root)
                if (root/'STOP').exists():
                    for mode,seed,_ in processes:
                        (root/mode/f'seed-{seed}'/'STOP').touch()
                time.sleep(30)
            comparison(root)
            codes = {f'{m}-{s}':p.returncode for m,s,p in processes}
            status('completed' if all(c==0 for c in codes.values()) else 'completed_with_failures', exit_codes=codes)
            return 0 if all(c==0 for c in codes.values()) else 1
        except BaseException as error:
            status('failed',error=str(error),traceback=traceback.format_exc())
            raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['suite','worker','status'])
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--study',type=Path)
    args=parser.parse_args()
    if args.command=='suite':
        return suite(args.root)
    if args.command=='worker':
        return worker(args.study)
    print(json.dumps({'suite':read(args.root/'suite.json') if (args.root/'suite.json').exists() else None,'budget':read(args.root/'budget.json') if (args.root/'budget.json').exists() else None,'comparison':comparison(args.root)},ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
