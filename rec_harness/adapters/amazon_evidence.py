"""Import real Amazon worker outcomes into immutable graph facts and CAS archives."""
from __future__ import annotations

import argparse
from pathlib import Path

from rec_harness.core import Artifacts, ContractError, digest, file_hash, read_json, write_json
from rec_harness.runtime import edge, node
from rec_harness.store import Store
from .amazon_identity import record_version


def job_partition(config, result):
    """Project the frozen seed split, never infer it from a receipt label."""
    keys = ('development_seeds', 'confirmation_seeds')
    if not any(key in config for key in keys):
        return 'unknown'
    groups = [config.get(key) for key in keys]
    if any(not isinstance(group, list) or any(type(seed) is not int for seed in group)
           for group in groups):
        raise ContractError('evidence requires complete frozen seed partitions')
    if any(len(set(group)) != len(group) for group in groups) or set(groups[0]) & set(groups[1]):
        raise ContractError('evidence seed partitions overlap or contain duplicates')
    seed = result.get('seed')
    if type(seed) is not int or seed not in groups[0] + groups[1]:
        raise ContractError('evidence seed outside frozen partitions')
    partition = 'development' if seed in groups[0] else 'confirmation'
    if 'partition' in result and result['partition'] != partition:
        raise ContractError('evidence receipt partition differs from frozen split')
    return partition


def restore_job_archive(refs,artifacts,destination):
    """Restore archived files, expanding locked runtime refs; does not launch a job."""
    destination=Path(destination)
    if destination.exists():raise ContractError('job archive restore requires a new destination')
    expanded=dict(refs);modes={}
    lock='isolated/runtime-lock.json'
    if lock in refs:
        from rec_harness.runtime_archive import validate_runtime
        archive=validate_runtime(read_json(artifacts.verify(refs[lock])))
        for name in ['isolated/request.json','isolated/completed.json']:
            if name not in refs or read_json(artifacts.verify(refs[name])).get('runtime_archive_hash')!=archive['hash']:
                raise ContractError('archived job runtime binding changed')
        for name,ref in archive['files'].items():
            target='isolated/jail/'+name
            if target in expanded and expanded[target]!=ref:
                raise ContractError('flat and locked runtime references disagree')
            expanded[target]=ref;modes[target]=archive['modes'][name]
    verified=set()
    for name,ref in expanded.items():
        path=Path(name)
        if not path.parts or path.is_absolute() or '..' in path.parts:
            raise ContractError('invalid archived job path')
        identity=(ref['sha256'],ref['size'])
        if identity not in verified:artifacts.verify(ref);verified.add(identity)
    artifacts.restore(expanded,destination)
    for name,mode in modes.items():(destination/name).chmod(mode)
    return {'files':len(expanded),'runtime_files':len(modes),'scope':'Archived files restored; device nodes and execution authorization are not recreated.'}


def archive_job(directory,artifacts,verified_runtime_objects=None):
    """Keep runtime bytes through their lock, without expanding them into graph attrs."""
    directory=Path(directory);verified=set() if verified_runtime_objects is None else verified_runtime_objects
    runtime={};lock=directory/'isolated/runtime-lock.json'
    if lock.exists():
        from rec_harness.runtime_archive import validate_runtime
        archive=validate_runtime(read_json(lock))
        request=read_json(directory/'isolated/request.json')
        receipt=read_json(directory/'isolated/completed.json')
        if request.get('runtime_archive_hash')!=archive['hash'] or receipt.get('runtime_archive_hash')!=archive['hash']:
            raise ContractError('job runtime archive binding changed')
        from rec_harness.process_identity import assert_stopped
        assert_stopped(directory/'isolated')
        for name,ref in archive['files'].items():
            key=(ref['sha256'],ref['size'])
            if key not in verified:artifacts.verify(ref);verified.add(key)
            path=directory/'isolated/jail'/name
            if path.is_symlink() or not path.is_file() or file_hash(path)!=ref['sha256']:
                raise ContractError('job runtime bytes differ from archive')
            runtime['isolated/jail/'+name]=ref
    files=[str(p.relative_to(directory)) for p in directory.rglob('*')
        if p.is_file() and not p.is_symlink() and str(p.relative_to(directory)) not in runtime]
    refs=artifacts.snapshot(directory,files)
    # The retained lock contains the complete path -> CAS mapping. Inputs,
    # outputs, logs and even unexpected files inside jail remain archived above.
    return refs


def ingest(directory):
    directory = Path(directory).resolve()
    from .amazon_controls import load_config
    config = load_config(directory)
    store = Store(directory / 'memory')
    artifacts = Artifacts(directory / 'artifacts/objects')
    imported = [];verified_runtime_objects=set()
    try:
        known = {n['id'] for n in store.snapshot()['nodes']}
        paths = sorted([*(directory / 'jobs').glob('*/completed.json'),
                        *(directory / 'attempt-history').glob('*/*/completed.json')],
                       key=lambda p: read_json(p)['process']['completed_at'])
        for path in paths:
            result = read_json(path)
            job_id = result.get('task_id', result['job_id'])
            if job_id in known:
                continue
            partition = job_partition(config, result)
            manifest = read_json(path.parent / 'manifest.json') if (path.parent / 'manifest.json').exists() else None
            task, protocol = 'amazon-' + result['dataset'], config['protocol_hash']
            refs = archive_job(path.parent,artifacts,verified_runtime_objects)
            source_ref = artifacts.put(config['model_versions'][record_version(result)]['path'])
            model_id, observation_id = job_id + '-model', job_id + '-observation'
            attrs = {'valid': result['status'] == 'success', 'status': result['status'], 'metrics': result['metrics'],
                     'partition': partition,
                     # Retrying a category/seed is not new independent evidence.
                     'independent_run_id': result['job_id'], 'seed': result['seed'], 'dataset': result['dataset'],
                     'effect_gate_passed': False, 'artifact_refs': refs, 'source_ref': source_ref,
                     'allocated_gpu_seconds': result['allocated_gpu_seconds'], 'peak_memory_bytes': result.get('peak_memory_bytes'),
                     'data_hashes': manifest['data_hashes'] if manifest else None,
                     'evaluation_candidates_hash': manifest['candidates_hash'] if manifest else None,
                     'evidence_dependency_ids': [job_id]}
            nodes = [node(job_id, 'Experiment', attrs, task, protocol),
                     node(model_id, 'ModelVersion', {'code_hash': config['model_versions'][record_version(result)]['sha256'],
                          'source_ref': source_ref, 'design_id': record_version(result), 'initialization': 'fresh',
                          'partition': partition,
                          'checkpoint_ref': refs.get('checkpoint.pt'), 'evidence_dependency_ids': [job_id]}, task, protocol),
                     node(observation_id, 'Observation', {'metrics': result['metrics'], 'partition': partition,
                          'evidence_dependency_ids': [job_id]}, task, protocol)]
            edges = [edge(job_id + '-evaluates', job_id, model_id, 'EVALUATES', [job_id], task, store.version + 1),
                     edge(job_id + '-produces', job_id, observation_id, 'PRODUCES', [job_id], task, store.version + 1)]
            if result['status'] != 'success':
                failure_id = job_id + '-failure'
                nodes.append(node(failure_id, 'Failure', {'failure_kind': result['status'], 'partition': partition,
                    'execution_evidence_only': True}, task, protocol))
                edges.append(edge(job_id + '-fails', job_id, failure_id, 'HAS_FAILURE', [job_id], task, store.version + 1))
            store.patch({'base_version': store.version, 'nodes': nodes, 'edges': edges})
            imported.append(job_id)
        report = {'imported': imported, 'graph_version': store.version, 'integrity': store.verify()}
        write_json(directory / 'evidence-import.json', report)
        return report
    finally:
        store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('directory')
    print(ingest(parser.parse_args().directory))
