"""Live, reproducible readback of the actual pre-design graph, not utility proof."""
import argparse
import os
from pathlib import Path

from .agents import object_schema, run_process, strings, validate_schema
from .core import ContractError, canonical, digest, file_hash, read_json, write_json


def expected(graph):
    ids = {n['id'] for n in graph['nodes']}
    if len(ids) != len(graph['nodes']):
        raise ContractError('duplicate graph identity')
    for e in graph['edges']:
        if not {e['source'], e['target'], *e['attrs'].get('evidence_refs', [])} <= ids:
            raise ContractError('graph relation has invisible reference')
    for n in graph['nodes']:
        if not set(n['attrs'].get('evidence_refs', [])) <= ids:
            raise ContractError('graph interpretation has invisible evidence')
        if n['attrs'].get('partition') in {'confirmation', 'test', 'final'}:
            raise ContractError('nondevelopment evidence in LLM graph')
    return {
        'snapshot_hash': digest(graph),
        'local_observation_ids': sorted(n['id'] for n in graph['nodes']
            if n['type'] in {'Experiment', 'Observation', 'Failure'} and n['attrs'].get('origin') == 'local_evidence'),
        'provisional_explanation_ids': sorted(n['id'] for n in graph['nodes']
            if n['type'] in {'Claim', 'GapHypothesis'} and n['attrs'].get('origin') == 'agent_hypothesis'),
        'unverified_source_ids': sorted(n['id'] for n in graph['nodes'] if n['type'] == 'Source'
            and n['attrs'].get('verification_status') != 'content_retrieved'),
        'counterevidence_edge_ids': sorted(e['id'] for e in graph['edges'] if e['relation'] == 'CONTRADICTS'),
        'revision_edge_ids': sorted(e['id'] for e in graph['edges'] if e['relation'] == 'SUPERSEDES'),
        'single_seed_mechanism_status': 'provisional',
    }


def audit(source, directory, *, executable='codex', model='gpt-6.1-sol', timeout=600):
    directory = Path(directory).resolve()
    if directory.exists():
        raise ContractError('audit output must be new; preserve prior transport and usage')
    raw = read_json(source)
    graph = raw.get('evidence_graph', raw)
    truth = expected(graph)
    directory.mkdir(parents=True)
    schema = object_schema({k: {'type': 'string'} if isinstance(v, str) else strings()
                            for k, v in truth.items()})
    schema['properties']['single_seed_mechanism_status']['enum'] = ['unknown', 'provisional', 'supported', 'confirmed_in_scope']
    context = {'evidence_graph': graph, 'snapshot_hash': digest(graph),
        'qualification': 'Readback of archived development evidence; not a new recommendation experiment.'}
    write_json(directory/'context.json', context)
    write_json(directory/'schema.json', schema)
    prompt = ('Read context.json only. Return exact node/edge IDs from the actual graph, sorted ascending. '
        'local_observation_ids: all Experiment/Observation/Failure with origin=local_evidence. '
        'provisional_explanation_ids: all Claim/GapHypothesis with origin=agent_hypothesis. '
        'unverified_source_ids: all Source nodes whose verification_status is not content_retrieved. '
        'counterevidence_edge_ids: all CONTRADICTS edges; revision_edge_ids: all SUPERSEDES edges. '
        'Copy snapshot_hash from the context. For single_seed_mechanism_status return exactly one enum token: '
        'the strongest permitted status of a mechanism interpretation based on one development seed '
        '(unknown, provisional, supported, confirmed_in_scope). Do not put an explanatory sentence in this field. '
        'Do not search, train, read other directories, or spawn agents. Never equate failed training with falsification. '
        'Return only the schema object.')
    argv = [executable, 'exec', '--json', '--ephemeral', '--skip-git-repo-check',
        '--sandbox', 'read-only', '--cd', str(directory), '--model', model,
        '-c', 'model_reasoning_effort="high"', '-c', 'web_search="disabled"',
        '-c', 'approval_policy="never"',
        '--output-schema', str(directory/'schema.json'),
        '--output-last-message', str(directory/'response.json'), '-']
    write_json(directory/'request.json', {'argv': argv, 'prompt': prompt, 'context_hash': digest(context),
        'source_sha256': file_hash(source), 'source': str(Path(source).resolve())})
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}
    result = run_process(argv, directory, directory, stdin=prompt, timeout=timeout, env=env)
    usages = []
    import json
    for line in (directory/'stdout.jsonl').read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get('type') == 'turn.completed': usages.append(event.get('usage'))
    write_json(directory/'usage.json', {'calls': 1, 'usage': usages, 'wall_seconds': result['wall_seconds']})
    if result['status'] != 'success' or not usages:
        raise ContractError('live graph readback transport failed; inspect preserved receipts')
    response = read_json(directory/'response.json')
    validate_schema(response, schema)
    checks = {k: response[k] == v for k, v in truth.items()}
    report = {'passed': all(checks.values()), 'checks': checks, 'graph_hash': digest(graph),
        'actual_response_hash': digest(response), 'qualification':
        'Exact graph readback and provenance classification only; does not prove reasoning quality or search utility.'}
    write_json(directory/'report.json', report)
    if not report['passed']: raise ContractError('graph readback mismatch')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source'); p.add_argument('output')
    a = p.parse_args()
    print(canonical(audit(a.source, a.output)))


if __name__ == '__main__': main()
