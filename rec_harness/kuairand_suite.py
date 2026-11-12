"""Unattended baseline + graph-D / no-graph-D supervisor for GPUs 0–7."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .autonomous import configuration
from .core import ContractError, canonical, digest, read_json, write_json
from .kuairand_baseline import lock


def repair_transient(root, *, reason):
    """Only retry observed transport errors, preserving every failed call and usage."""
    from .adapters.kuairand_research import KuaiRandResearch
    from .core import Artifacts
    c = KuaiRandResearch(root)
    try:
        with c.lock():
            state = c.state()
            if state.get('status') != 'blocked' or 'Codex transport failed' not in state.get('blocking_reason', ''): return False
            failed = []
            for request in (c.root/'stages').rglob('request.json'):
                directory = request.parent
                if (directory/'accepted.json').exists() or not (directory/'usage.json').exists(): continue
                text = ''.join(p.read_text(errors='replace') for p in directory.glob('*.txt'))
                stdout = directory/'stdout.jsonl'
                if stdout.exists(): text += stdout.read_text(errors='replace')
                if not any(term in text.lower() for term in ('429', '502', '503', '504', 'connection', 'temporarily', 'stream disconnected', 'timeout', 'timed out', 'server_error')):
                    return False
                failed.append(directory)
            if not failed: return False
            archives = []
            for directory in failed:
                refs = c.artifacts.snapshot(directory, [str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()])
                target = c.root/'failed-calls'/(digest(str(directory))[:16]+'-'+str(time.time_ns()))
                target.parent.mkdir(exist_ok=True); directory.rename(target)
                archives.append({'path': str(target.relative_to(c.root)), 'refs': refs})
            c.store.event('kuairand_transport_repair', {'reason': reason, 'retained_calls': archives})
            state.update(status='waiting_baselines', repair_reason=reason); c.save(state)
            return True
    finally: c.store.close()


def retire_preparation(root):
    """Replace only this session's known download job; never touch training."""
    baseline = root/'baseline'; state = read_json(baseline/'controller.json')
    if state['stage'] != 'prepare' or state['status'] != 'running': return
    attempts = sorted((baseline/'jobs/preparation').glob('attempt-*.json'))
    latest = read_json(attempts[-1]); pid = latest['pid']
    cmdline = Path('/proc')/str(pid)/'cmdline'
    if not cmdline.exists(): return
    args = cmdline.read_bytes().split(b'\0')
    if b'longseq_bench.pipeline' not in args or b'--prepare-only' not in args:
        raise ContractError('refuse to replace an unknown process')
    os.kill(pid, signal.SIGTERM)
    for _ in range(30):
        if read_json(baseline/'controller.json')['status'] != 'running': break
        time.sleep(1)
    else: raise ContractError('old preparation did not stop')
    partial = root/'data/downloads/KuaiRand-27K.tar.gz.part'
    if partial.exists():
        retired = root/'data/downloads/interrupted'/str(time.time_ns()); retired.mkdir(parents=True)
        partial.rename(retired/partial.name)
        write_json(retired/'reason.json', {'reason': 'possible overlap of two earlier download writers; preserve bytes and restart verified range download',
            'process_receipt': latest, 'retired_at': time.time()})


def suite(root, *, replace_preparation=False):
    from .adapters.kuairand_research import KuaiRandResearch
    from .kuairand_baseline import run as baseline_run
    from .kuairand_download import download
    from .graph_llm_audit import audit
    workspace = Path(__file__).resolve().parents[1]
    root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
    state = {'status': 'running', 'stage': 'acceptance', 'started_at': time.time(),
        'gpus': {'baseline_then_graph': [0, 1, 2, 3], 'no_graph': [4, 5, 6, 7]},
        'authorization': 'Full D both arms, unlimited rounds/cost, autonomous recoverable repair, frozen final with no feedback'}
    with lock(root/'.suite.lock'):
        write_json(root/'suite.json', state)
        try:
            if replace_preparation: retire_preparation(root)
            if (root/'STOP').exists(): return
            actual = read_json(root/'graph-audit/actual-prefix-v2/report.json')
            if actual.get('passed') is not True: raise ContractError('existing graph failed actual LLM readback')
            fixture_dir = root/'graph-audit/counter-revision-live'
            if not (fixture_dir/'report.json').exists():
                audit(root/'acceptance/graph-readback-fixture.json', fixture_dir)
            if not read_json(fixture_dir/'report.json')['passed']: raise ContractError('live counter/revision readback failed')
            from .kuairand_acceptance import run as worker_acceptance
            worker_acceptance(root, read_json(workspace/'examples/kuairand/baseline.json'))
            configs = [configuration(workspace/'examples/kuairand'/file) for file in ('graph.json', 'no-graph.json')]
            common = [{k: v for k, v in c.items() if k not in {'memory', 'gpus'}} for c in configs]
            if common[0] != common[1]: raise ContractError('research arms differ outside graph visibility and GPUs')
            for name, config in zip(('graph', 'no-graph'), configs):
                path = root/name/'campaign'
                if not (path/'kuairand-auto.json').exists(): KuaiRandResearch.init(path, config)
            left, right = [read_json(root/name/'campaign/kuairand-auto.json') for name in ('graph', 'no-graph')]
            if left['protocol_hash'] != right['protocol_hash'] or left['runtime'] != right['runtime']:
                raise ContractError('frozen arms have different protocols/environments')
            write_json(root/'comparison-contract.json', {'method': 'D-adaptive-v1', 'graph_protocol_hash': left['protocol_hash'],
                'no_graph_protocol_hash': right['protocol_hash'], 'runtime': left['runtime'],
                'difference': ['graph visibility', 'GPU indices'], 'max_rounds': None,
                'fixed_per_plan_rounds': None, 'final_feedback': False, 'same_seed_is_not_replication': True})
            state['stage'] = 'official_data_download'; write_json(root/'suite.json', state)
            try: download(root/'data/downloads')
            except ContractError as error:
                if 'does not support exact byte ranges' not in str(error): raise
                state['download_fallback'] = 'official serial downloader, same MD5'
            state['stage'] = 'original_four_model_baselines'; write_json(root/'suite.json', state)
            if not (root/'baseline/controller.json').exists() or read_json(root/'baseline/controller.json')['status'] != 'completed':
                baseline_run(read_json(workspace/'examples/kuairand/baseline.json'))
            if (root/'STOP').exists(): return
            state['stage'] = 'continuous_two_arm_research'; write_json(root/'suite.json', state)
            # Separate processes keep arm failures, contexts and final outputs independent.
            children = []
            restart_count = {'graph': 0, 'no-graph': 0}
            def start(name):
                path = root/name/'campaign'
                with (root/name/'controller.log').open('a') as log:
                    return subprocess.Popen([sys.executable, '-m', 'rec_harness', 'autoresearch', str(path)],
                        cwd=workspace, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            for name in ('graph', 'no-graph'):
                children.append((name, start(name)))
            state['children'] = {name: p.pid for name, p in children}; write_json(root/'suite.json', state)
            while children:
                if (root/'STOP').exists():
                    for name, _ in children: (root/name/'campaign/STOP').touch()
                for name, process in list(children):
                    if process.poll() is None: continue
                    children.remove((name, process))
                    if not (root/'STOP').exists() and process.returncode != 0 and restart_count[name] < 3:
                        time.sleep(20 * (restart_count[name]+1))
                        if repair_transient(root/name/'campaign', reason='autonomous retry of observed transient Codex service error'):
                            restart_count[name] += 1; children.append((name, start(name)))
                            state.setdefault('transport_restarts', {})[name] = restart_count[name]
                            continue
                    state.setdefault('finished_children', {})[name] = process.returncode
                    write_json(root/'suite.json', state)
                if children: time.sleep(10)
            state['status'] = 'stopped' if (root/'STOP').exists() else 'blocked'
        except BaseException as error:
            state.update(status='blocked', reason=str(error)); raise
        finally:
            state['updated_at'] = time.time(); write_json(root/'suite.json', state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['launch', 'run', 'status', 'stop', 'restart-preparation'])
    p.add_argument('--root', default='artifacts/kuairand-research')
    p.add_argument('--replace-preparation', action='store_true')
    p.add_argument('--reason')
    a = p.parse_args(); root = Path(a.root).resolve()
    if a.action == 'restart-preparation':
        if not a.reason: raise ContractError('preparation repair requires a recorded reason')
        state = read_json(root/'suite.json')
        if state['stage'] != 'official_data_download': raise ContractError('automatic registration repair is only allowed before formal baseline/research training')
        from .process_identity import identity
        record = read_json(root/'supervisor-launch.json')['worker']
        current = identity(record['pid'])
        if current['start_ticks'] != record['start_ticks'] or current['boot_id'] != record['boot_id']:
            raise ContractError('supervisor identity changed; refuse to stop another process')
        os.kill(record['pid'], signal.SIGTERM)
        for _ in range(30):
            try: alive = identity(record['pid'])['state'] not in {'Z', 'X'}
            except FileNotFoundError: alive = False
            if not alive: break
            time.sleep(1)
        else: raise ContractError('previous supervisor did not exit')
        retained = []
        revision = root/'pretraining-registrations'/str(time.time_ns())
        for name in ('graph', 'no-graph'):
            campaign = root/name/'campaign'
            if not campaign.exists(): continue
            old = read_json(campaign/'state.json')
            if old['records'] or old['stage'] != 0: raise ContractError('cannot re-register a campaign with research decisions or training')
            revision.mkdir(parents=True, exist_ok=True)
            campaign.rename(revision/name)
            retained.append(str((revision/name).relative_to(root)))
        write_json(revision/'repair.json', {'reason': a.reason, 'retained_empty_registrations': retained,
            'training_started': False, 'range_chunks': 'retained and resumed with official MD5 validation'})
        a.action = 'launch'; a.replace_preparation = False
    if a.action == 'launch':
        root.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, '-m', 'rec_harness.kuairand_suite', 'run', '--root', str(root)]
        if a.replace_preparation: argv.append('--replace-preparation')
        with (root/'supervisor.log').open('a') as log:
            process = subprocess.Popen(argv, cwd=Path(__file__).resolve().parents[1], stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        from .process_identity import identity
        write_json(root/'supervisor-launch.json', {'argv': argv, 'worker': identity(process.pid)})
        print(canonical({'pid': process.pid, 'root': str(root)}))
    elif a.action == 'run': suite(root, replace_preparation=a.replace_preparation)
    elif a.action == 'stop':
        (root/'STOP').touch()
        for name in ('graph', 'no-graph'):
            path = root/name/'campaign'
            if path.exists(): (path/'STOP').touch()
        if (root/'baseline').exists(): (root/'baseline/STOP').touch()
        print(canonical({'stop_new_tasks': True, 'active_training': 'finish_and_checkpoint'}))
    else:
        result = read_json(root/'suite.json') if (root/'suite.json').exists() else {'status': 'not_launched'}
        for name in ('graph', 'no-graph'):
            path = root/name/'campaign/state.json'
            if path.exists():
                state = read_json(path)
                result[name] = {k: state[k] for k in ('status', 'stage', 'frontier')}
                result[name]['completed_models'] = len(state['records'])
        print(canonical(result))


if __name__ == '__main__': main()
