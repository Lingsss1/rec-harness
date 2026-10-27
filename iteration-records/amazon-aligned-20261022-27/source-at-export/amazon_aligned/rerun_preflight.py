"""Regression replay, real GPU workers and native initialization for the reruns."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from .bootstrap import bootstrap
bootstrap()
from .core import Controller, read, save, schema, TEXT
from .prepare import WORKSPACE, sha
from .protocol_fixes import diagnosis_presentation, restore_diagnosis_prompt
from .reruns import DESTINATION, registered_files


def configs():
    plan = read(DESTINATION / 'plan.json')
    return {q['jobs'][0]['method']: read(q['jobs'][0]['config']) for q in plan['queues']}


def regression():
    command = [sys.executable, '-m', 'unittest', 'amazon_aligned.test_protocol_fixes',
               'amazon_aligned.test_accounting', 'amazon_aligned.test_recharness', '-v']
    with (DESTINATION / 'preflight/tests.log').open('w') as log:
        subprocess.run(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT, check=True)
    previous = WORKSPACE / 'artifacts/amazon-repeats-20261025/self_evolverec-run-02/self_evolverec'
    raw_path = previous / 'trials/candidate-00006-gpu3/diagnosis.json'
    raw = read(raw_path)
    merged = {'metrics': {}, 'metric_definitions': {}}
    for dataset, value in raw.items():
        for key in merged:
            merged[key].update({dataset + ':' + k: v for k, v in value[key].items()})
    # Restore the real evolved probe that omitted the presentation function.
    absent = []
    for path in previous.glob('trials/*/diagnosis_tools.py'):
        if not any(isinstance(n, ast.FunctionDef) and n.name == 'diagnosis_interpreter_prompt'
                   for n in ast.parse(path.read_text()).body):
            absent.append(path)
    if not absent:
        raise RuntimeError('Expected original missing-function regression fixture')
    repaired, changed = restore_diagnosis_prompt(absent[0].read_text())
    # Load only the pinned function, avoiding candidate module side effects.
    tree = ast.parse(repaired)
    function = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'diagnosis_interpreter_prompt'][-1]
    namespace = {'json': json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<presentation-regression>', 'exec'), namespace)
    prompt, payload, info = diagnosis_presentation(SimpleNamespace(**namespace), merged)
    assert changed and len(prompt) < 650001
    assert set(payload['metrics']) == set(merged['metrics'])
    save(DESTINATION / 'preflight/regression.json', {
        'passed': True, 'unit_tests': 10, 'original_failed_call_chars': 2670775,
        'diagnostic_fixture': str(raw_path), 'fixture_sha256': sha(raw_path),
        'missing_function_fixture': str(absent[0]), 'missing_fixture_sha256': sha(absent[0]),
        'restored_probe_sha256': __import__('hashlib').sha256(repaired.encode()).hexdigest(),
        'prompt': info})


def gpu():
    all_configs = configs()
    diagnosis = (WORKSPACE / 'artifacts/Self-EvolveRec-code/examples/sasrec_cds/initial_code/diagnosis_tools.py').read_text()
    # Deliberately remove the original function to exercise the actual field fix.
    tree = ast.parse(diagnosis)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'diagnosis_interpreter_prompt')
    lines = diagnosis.splitlines(keepends=True)
    del lines[node.lineno - 1:node.end_lineno]
    repaired, changed = restore_diagnosis_prompt(''.join(lines))
    assert changed
    controllers = {}
    for method, config in all_configs.items():
        controllers[method] = Controller({**config, 'root': str(DESTINATION / 'preflight/gpu')}, method)
    def one(pair):
        method, gpu_id = pair
        controller = controllers[method]
        return controller.trial('gpu-' + str(gpu_id), controller.baseline(), gpu_id, requested=600,
            files={'diagnosis_tools.py': repaired} if method == 'self_evolverec' else None, smoke=True)
    pairs = [(method, g) for method, c in all_configs.items() for g in c['gpus']]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, pairs))
    save(DESTINATION / 'preflight/gpu.json', {
        'passed': all(r['status'] == 'success' for r in results), 'results': results,
        'gpu_seconds_outside_formal_search': sum(r['allocated_gpu_seconds'] for r in results)})
    if any(r['status'] != 'success' for r in results):
        raise RuntimeError('GPU/probe/single-model gate preflight failed; inspect worker logs')


def native_mle():
    from .mlevolve import NativeController, build
    from .rerun_controller import SingleSelectionMixin
    class ProbeController(SingleSelectionMixin, NativeController):
        pass
    config = {**configs()['mlevolve'], 'root': str(DESTINATION / 'preflight')}
    controller = ProbeController(config, 'native-mlevolve')
    controller.best = {'id': 'initialization-only', 'status': 'not_trained', 'score': None}
    agent = build(controller)
    assert agent.acfg.use_fusion is False and agent.acfg.use_aggregation is False
    assert agent.max_fusion_drafts == 0 and len(agent.journal.nodes) == 1
    assert agent.global_memory is not None
    value = controller.query_native('Transport and single-model contract acceptance: return status=ready '
        'and describe the allowed predictor and prohibited model combinations in one sentence.',
        schema({'status': {'type': 'string', 'enum': ['ready']}, 'constraint': TEXT}))
    save(DESTINATION / 'preflight/native-mlevolve.json', {
        'passed': value['status'] == 'ready', 'use_fusion': agent.acfg.use_fusion,
        'use_aggregation': agent.acfg.use_aggregation, 'max_fusion_drafts': agent.max_fusion_drafts,
        'memory': type(agent.global_memory).__name__, 'response': value,
        'model': config['llm_model'], 'effort': config['llm_effort']})


def native_self():
    from .self_evolverec import UPSTREAM, create_engine, install_transport
    from .rerun_controller import FreshSelfController
    sys.path.insert(0, str(UPSTREAM))
    config = {**configs()['self_evolverec'], 'root': str(DESTINATION / 'preflight')}
    controller = FreshSelfController(config, 'native-self')
    install_transport(controller)
    from self_evolverec import Self_EvolveRec
    Self_EvolveRec._setup_logging = lambda self: None
    engine = create_engine(controller, 0, (UPSTREAM / 'examples/sasrec_cds/initial_code/diagnosis_tools.py').read_text())
    assert engine.config.coder.reasoning_effort == 'medium'
    # Exercise the actual replacement Runner and schema with a small real call.
    import asyncio
    from agents import Agent, Runner
    result = asyncio.run(Runner.run(Agent(name='Diagnostic contract acceptance',
        instructions='Return the word ready. You retain diagnosis_interpreter_prompt and bounded diagnostic examples.',
        model=config['llm_model']), input='Confirm the transport is ready.'))
    assert 'ready' in result.final_output.lower()
    save(DESTINATION / 'preflight/native-self.json', {
        'passed': True, 'native_engine': type(engine).__name__, 'transport': result.final_output,
        'model': config['llm_model'], 'effort': config['llm_effort'],
        'diagnosis_contract_fix': True, 'bounded_payloads': True})


def constraint_gate():
    from .prepare import ROOT
    config = configs()['mlevolve']
    previous = read(ROOT / 'mlevolve/summary.json')
    target = DESTINATION / 'preflight/ensemble-rejected.json'
    command = [config['worker_python'], '-m', 'amazon_aligned.single_model',
               '--audit', previous['selected']['output'], '--data', config['data_dir'], '--output', str(target)]
    controller = Controller({**config, 'root': str(DESTINATION / 'preflight')}, 'constraint-gate')
    with target.with_suffix('.log').open('w') as log:
        subprocess.run(command, cwd=WORKSPACE, env=controller.environment(''),
                       stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
    result = read(target)
    assert result['eligible'] is False
    # Real baseline predictors already passed the same audit inside all four
    # MLEvolve smoke workers, before external validation could be accepted.
    audits = [read(p) for p in (DESTINATION / 'preflight/gpu/mlevolve/smoke').glob('*/single-model-audit.json')]
    assert len(audits) == 4 and all(a['eligible'] is True for a in audits)
    save(DESTINATION / 'preflight/constraint-gate.json', {
        'passed': True, 'previous_ensemble_rejected': result['reason'],
        'single_model_baselines_accepted': len(audits)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', required=True, choices=['regression', 'gpu', 'native-mlevolve', 'native-self', 'constraint-gate', 'accept'])
    args = parser.parse_args()
    (DESTINATION / 'preflight').mkdir(parents=True, exist_ok=True)
    if args.phase == 'regression':
        regression()
    elif args.phase == 'gpu':
        gpu()
    elif args.phase == 'native-mlevolve':
        native_mle()
    elif args.phase == 'native-self':
        native_self()
    elif args.phase == 'constraint-gate':
        constraint_gate()
    else:
        checks = {name: read(DESTINATION / 'preflight' / (name + '.json'))['passed']
                  for name in ['regression', 'gpu', 'native-mlevolve', 'native-self', 'constraint-gate']}
        if not all(checks.values()):
            raise RuntimeError('Rerun preflight did not pass')
        plan = read(DESTINATION / 'plan.json')
        save(DESTINATION / 'preflight/acceptance.json', {
            'passed': True, 'checks': checks, 'files': registered_files(plan),
            'plan_sha256': sha(DESTINATION / 'plan.json'), 'registered_at': time.time()})
    print('RERUN_PREFLIGHT_' + args.phase.upper() + '_PASSED', flush=True)


if __name__ == '__main__':
    main()
