"""Operational and host-only final-evaluation controls; never feed final results to roles."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .core import ContractError, canonical, digest, file_hash, read_json, write_json
from .kuairand_baseline import command, environment, lock


def freeze_final(controller, state):
    """Freeze every Pareto candidate once. Enqueue, do not read scores back."""
    queue = controller.root/'host-only-final'
    queue.mkdir(exist_ok=True)
    for model_id in state['frontier']:
        target = queue/model_id
        if (target/'seal.json').exists(): continue
        folder = controller.model_folder(model_id)
        checkpoint = folder/'training/last.pt'
        result = read_json(folder/'training/result.json')
        target.mkdir(exist_ok=True)
        seal = {'model_id': model_id, 'stage': state['stage'], 'selected_by': 'validation_pareto',
            'candidate_hash': file_hash(folder/'model.py'), 'checkpoint_hash': file_hash(checkpoint),
            'treatment_hash': file_hash(folder/'training.json'), 'validation_result_hash': digest(result),
            'protocol_hash': controller.protocol, 'frozen_at': time.time(), 'no_feedback': True}
        write_json(target/'seal.json', seal)
        # Record selection identity only. Final outputs are never imported into graph/context.
        controller.store.event('kuairand_final_frozen', {'model_id': model_id, 'seal_hash': digest(seal),
            'qualification': 'final results excluded from research evidence'})


def final_run(root, model_id, gpus):
    from .adapters.kuairand_research import KuaiRandResearch
    c = KuaiRandResearch(root)
    try:
        target = c.root/'host-only-final'/model_id
        seal = read_json(target/'seal.json'); folder = c.model_folder(model_id)
        for path, key in [('model.py', 'candidate_hash'), ('training.json', 'treatment_hash'), ('training/last.pt', 'checkpoint_hash')]:
            if file_hash(folder/path) != seal[key]: raise ContractError('sealed model identity changed')
        if (target/'result.json').exists(): return {'status': 'completed', 'model_id': model_id}
        if (target/'reserved.json').exists(): raise ContractError('final state uncertain; do not automatically repeat')
        if any(g not in c.config['gpus'] for g in gpus) or len(gpus) != 1:
            raise ContractError('final scoring must use one GPU owned by this arm')
        write_json(target/'reserved.json', {'seal_hash': digest(seal), 'started_at': time.time()})
        command([c.python, '-m', 'rec_harness.kuairand_control', 'final-worker', str(c.root), '--model', model_id],
            target/'job', cwd=c.workspace, env=environment(c.benchmark, c.workspace, gpus), gpus=gpus)
        # Only completion metadata, never metrics, crosses this boundary.
        return {'status': 'completed', 'model_id': model_id}
    finally: c.store.close()


def final_worker(root, model_id):
    import torch
    from torch.utils.data import DataLoader
    from .adapters.kuairand_research import KuaiRandResearch
    c = KuaiRandResearch(root)
    try:
        sys.path.insert(0, str(c.benchmark))
        from .kuairand_data import ResearchDataset
        from .kuairand_worker import load_candidate, wrapper, score_model
        target = c.root/'host-only-final'/model_id; seal = read_json(target/'seal.json')
        folder = c.model_folder(model_id)
        if file_hash(folder/'training/last.pt') != seal['checkpoint_hash']: raise ContractError('final checkpoint changed')
        cfg = c.config['protocol']; state = c.state()
        model_kind = next(r['model'] for r in state['records'] if r['id'] == model_id)
        model = wrapper(load_candidate(folder/'model.py', {**cfg['model'], 'name': model_kind}),
                        cfg, read_json(folder/'training.json')).cuda()
        saved = torch.load(folder/'training/last.pt', map_location='cpu', weights_only=False)
        if saved['next_epoch'] != 101: raise ContractError('final candidate is not fully trained')
        model.load_state_dict(saved['model'])
        cache = read_json(c.root/'cache-prepared.json')['cache']
        loader = DataLoader(ResearchDataset(cache, 'final'), batch_size=cfg['evaluation']['batch_size'], shuffle=False)
        started = time.perf_counter()
        metrics = score_model(model, loader, 'cuda', cfg['model']['num_items'], cfg['evaluation']['candidate_chunk'])
        torch.cuda.synchronize()
        write_json(target/'result.json', {'kind': 'host_only_frozen_final', 'seal_hash': digest(seal),
            'metrics': metrics, 'scoring_seconds': time.perf_counter()-started, 'no_feedback': True})
    finally: c.store.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['status', 'stop', 'resume', 'repair', 'prepare-cache', 'final-run', 'final-worker'])
    p.add_argument('directory'); p.add_argument('--model'); p.add_argument('--gpu', type=int)
    p.add_argument('--reason')
    a = p.parse_args()
    if a.action == 'final-worker': final_worker(a.directory, a.model); return
    if a.action == 'final-run': print(canonical(final_run(a.directory, a.model, [a.gpu]))); return
    from .adapters.kuairand_research import KuaiRandResearch
    c = KuaiRandResearch(a.directory)
    try:
        if a.action == 'status': result = c.status()
        elif a.action == 'stop':
            (c.root/'STOP').touch(); result = {'stop_new_tasks': True, 'running_tasks': 'finish_and_checkpoint'}
        elif a.action == 'prepare-cache':
            cache = c.ensure_cache(); write_json(c.root/'cache-prepared.json', {'cache': str(cache)}); result = {'prepared': True}
        elif a.action == 'repair':
            if not a.reason: raise ContractError('repair requires a recorded reason')
            with lock(c.root/'.controller.lock'):
                state = c.state()
                if state['status'] != 'blocked': raise ContractError('only blocked campaigns need repair')
                state.update(status='waiting_baselines', repaired_at=time.time(), repair_reason=a.reason)
                c.save(state); c.store.event('kuairand_repair', {'reason': a.reason})
            result = c.status()
        else:
            (c.root/'STOP').unlink(missing_ok=True); result = c.run()
        print(canonical(result))
    finally: c.store.close()


if __name__ == '__main__': main()
