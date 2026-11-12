"""Small CUDA/DDP research-worker resume check, excluded from real results."""
import csv
from pathlib import Path
import sys

from .core import ContractError, digest, file_hash, read_json, write_json
from .kuairand_baseline import command, environment


def run(root, config):
    from .adapters.kuairand_research import BASE_SOURCE, TREATMENT
    root = Path(root)/'acceptance/research-worker-resume'
    root.mkdir(parents=True, exist_ok=True)
    if (root/'passed.json').exists(): return read_json(root/'passed.json')
    benchmark = Path(config['benchmark']); workspace = Path(__file__).resolve().parents[1]
    python = str(Path(config['python']).absolute())
    cfg = read_json(benchmark/'configs/benchmark.json')
    cfg['model'].update(num_items=31, max_history=16, dim=8, heads=2, dqk=4, dv=8,
        input_dropout=0., linear_dropout=0., query_chunk=4, sla_local=4, sla_global=2,
        truncate_keep=8, dare_top_k=3, cc_group_size=4, cc_channel_size=2)
    cfg['evaluation'].update(batch_size=16, candidate_chunk=16)
    write_json(root/'config.json', cfg); write_json(root/'training.json', TREATMENT)
    (root/'model.py').write_text(BASE_SOURCE)
    source = root/'synthetic.csv'
    with source.open('w') as f:
        writer = csv.writer(f); writer.writerow(['user_id', 'sequence_item_ids', 'sequence_timestamps'])
        for uid in range(128):
            writer.writerow([uid, ','.join(str((uid+i)%31) for i in range(19)), ','.join(str(1000*i) for i in range(19))])
    cache = root/'cache'
    env = environment(benchmark.resolve(), workspace, [4, 5])
    if not cache.exists():
        command([python, '-c',
            'from rec_harness.kuairand_data import make_research_cache; import sys; make_research_cache(sys.argv[1],sys.argv[2],16)',
            str(source), str(cache)], root/'cache-job', cwd=workspace, env=env)
    args = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2', '-m',
        'rec_harness.kuairand_worker', 'train', '--synthetic', '--benchmark', str(benchmark), '--model', 'hstu',
        '--candidate', str(root/'model.py'), '--config', str(root/'config.json'), '--treatment', str(root/'training.json'),
        '--cache', str(cache), '--output', str(root/'training')]
    output = root/'training'
    if not (output/'last.pt').exists():
        command([*args, '--stop-after-epoch', '2'], root/'jobs/interruption', cwd=workspace, env=env, gpus=[4, 5])
    command(args, root/'jobs/resume', cwd=workspace, env=env, gpus=[4, 5])
    before = file_hash(output/'result.json')
    result = read_json(output/'result.json')
    if result['kind'] != 'synthetic_research_validation' or result['epochs'] != 101:
        raise ContractError('research-worker synthetic recovery result missing')
    command(args, root/'jobs/completed-resume', cwd=workspace, env=env, gpus=[4, 5])
    if file_hash(output/'result.json') != before: raise ContractError('completed resume changed result')
    receipt = {'passed': True, 'synthetic': True, 'formal_result': False,
        'checks': ['CUDA_DDP_training', 'injected_epoch_boundary_stop', 'resumed_to_101', 'completed_resume_does_not_rescore'],
        'result_hash': before}
    write_json(root/'passed.json', receipt)
    return receipt
