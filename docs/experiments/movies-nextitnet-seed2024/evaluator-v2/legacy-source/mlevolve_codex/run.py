"""Run upstream MLEvolve search/agents against the frozen RecHarness worker.

Only the LLM transport and task-specific execution/metric boundaries are replaced.
The upstream search, draft/debug/improve/evolution/fusion agents, diff planner,
code review, and hybrid experience memory remain callable upstream code.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import copy
import fcntl
import json
import logging
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

from mle_star_codex.run import Controller, baseline_source, read, save, sha, source_parts

WORKSPACE = Path(__file__).resolve().parents[1]
UPSTREAM = WORKSPACE / 'artifacts/MLEvolve'
ROOT = WORKSPACE / 'artifacts/mlevolve-movies-2024'
COMMIT = 'f4a37b96c948bfed6564e62f3229e0b38e908d33'
TEXT = {'type': 'string'}
BOUNDARY = '''This run is a frozen NextItNet / Movies_and_TV adaptation, seed 2024.
The task-specific contract overrides generic Kaggle pipeline instructions below.
Optimize validation HR@10 (higher is better) using ONLY a causal dilated-convolution
NextItNet. Source consists of exactly import math, import torch, import torch.nn as nn;
eight numeric constants MAXLEN,HIDDEN_UNITS,NUM_BLOCKS,KERNEL_SIZE,DROPOUT_RATE,LR,
WEIGHT_DECAY,BATCH_SIZE; and classes NextItNetBlock and NextItNetModel, methods only.
Preserve constructor, forward, encode_sequence and score_candidates interfaces.
The host owns preprocessing, splits, BPR loss, Adam, clipping, seed, epochs, negative
sampling, all training and all evaluation. Model code must not import extra modules,
access files/network, print, run training, implement evaluation, or generate CSV files.
For this task 'submission' means the host checkpoint and candidate scores. Generic
instructions to implement pipelines, install dependencies, print scores or save CSVs
are fulfilled by the frozen host; do not add such operations to model code.
Planner modules data_preparation/training/evaluation are frozen host components;
select only model-related components or hyperparameters. Fusion means merging
architectural insights into ONE NextItNet, not prediction ensembling. No test metric
is available during search. There are exactly 44 candidate slots, including debug
and fusion attempts. Do not inspect the filesystem, use tools, or edit files. Return
the requested response format, including markdown code or SEARCH/REPLACE when asked.
'''


def transport_schema(schema):
    """Close upstream function schemas, including its dynamic module-plan map."""
    result = copy.deepcopy(schema)
    if result.get('type') == 'object':
        if 'properties' not in result:
            value = result.get('additionalProperties')
            if not isinstance(value, dict):
                raise ValueError('Unspecified upstream object schema')
            result['properties'] = {key: copy.deepcopy(value) for key in (
                'data_processing_and_feature_engineering', 'model_design', 'training_evaluation')}
            result.setdefault('required', [])
        result['additionalProperties'] = False
        result['properties'] = {k: transport_schema(v) for k, v in result['properties'].items()}
    if result.get('type') == 'array':
        result['items'] = transport_schema(result['items'])
    for key in ('anyOf', 'oneOf', 'allOf'):
        if key in result:
            result[key] = [transport_schema(v) for v in result[key]]
    return result


class NativeController(Controller):
    def __init__(self, config):
        super().__init__(config)
        self.call_lock = threading.Lock()
        self.event_lock = threading.Lock()
        self.next_call = 0

    def event(self, kind, **data):
        with self.event_lock:
            super().event(kind, **data)

    def query_native(self, prompt, schema=None):
        from staged_research.codex import service_schema, remove_optional_nulls
        import jsonschema
        with self.call_lock:
            self.next_call += 1
            key = '%05d' % self.next_call
        d = self.root / 'calls' / key
        d.mkdir(parents=True, exist_ok=False)
        original = schema or {'type': 'object', 'properties': {'text': TEXT}, 'required': ['text']}
        bounded = transport_schema(original)
        normalized = service_schema(bounded)
        save(d / 'schema.json', normalized)
        instruction = BOUNDARY + '\nUPSTREAM REQUEST:\n' + prompt
        if schema is None:
            instruction += '\nReturn the requested completion verbatim inside the JSON text field.'
        save(d / 'request.json', {'prompt': instruction, 'original_schema': original})
        command = ['codex', 'exec', '--ephemeral', '--skip-git-repo-check', '--ignore-user-config',
            '--ignore-rules', '--model', self.c['model'], '--sandbox', 'read-only',
            '-c', 'approval_policy="never"', '-c', 'web_search="disabled"',
            '-c', 'model_reasoning_effort="' + self.c['effort'] + '"',
            '--json', '--output-schema', str(d / 'schema.json'), '-o', str(d / 'answer.json'),
            '-C', str(self.codex_cwd), '-']
        save(d / 'started.json', {'command': command, 'time': time.time()})
        self.event('codex_started', key=key)
        with (d / 'events.jsonl').open('w') as out, (d / 'stderr.log').open('w') as err:
            result = subprocess.run(command, input=instruction, text=True, stdout=out,
                stderr=err, timeout=1200, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        if result.returncode or not (d / 'answer.json').exists():
            raise RuntimeError('Codex transport failed; retain incomplete call: ' + str(d))
        value = remove_optional_nulls(read(d / 'answer.json'), bounded)
        jsonschema.validate(value, original)
        save(d / 'response.json', value)
        self.event('codex_completed', key=key)
        return value if schema is not None else value['text']


def install_transport(controller):
    # Patch before importing agents, whose modules import these function objects.
    import llm
    def query(system_message=None, user_message=None, func_spec=None, **kwargs):
        prompt = '\n'.join(llm.compile_prompt_to_md(x) for x in (system_message, user_message) if x)
        return controller.query_native(prompt, func_spec.json_schema if func_spec else None)
    def generate(prompt, json_schema=None, **kwargs):
        result = controller.query_native(llm.compile_prompt_to_md(prompt), json_schema)
        return json.dumps(result) if json_schema is not None else result
    llm.query, llm.generate = query, generate


def parse_result(agent, node, exec_result):
    """Score is host-owned; LLM supplies interpretation and memory summary only."""
    from utils.metric import MetricValue, WorstMetricValue
    receipt = json.loads(exec_result.term_out[0])
    node.absorb_exec_result(exec_result)
    ok = receipt['status'] == 'success'
    node.is_buggy, node.is_valid = not ok, ok
    node.metric = MetricValue(receipt['metrics']['HR@10'], maximize=True) if ok else WorstMetricValue()
    response = agent.controller.query_native(json.dumps({'role': 'execution interpretation',
        'plan': node.plan, 'model': node.code, 'host_result': receipt,
        'instruction': 'Explain observed success/failure and summarize architecture for experience memory. Do not invent metrics.'}),
        {'type': 'object', 'properties': {'analysis': TEXT, 'code_summary': TEXT},
         'required': ['analysis', 'code_summary']})
    node.analysis, node.code_summary = response['analysis'], response['code_summary']
    if ok:
        with agent.memory_lock:
            if not agent.global_memory.save_node(node, node.parent):
                raise RuntimeError('Experience memory save failed')
    return node


def validate_node(agent, node):
    if not node.is_buggy:
        agent.branch_successful_nodes.setdefault(node.branch_id, []).append(node)


def persist_best(agent, node):
    from engine.solution_manager import update_top_candidates
    if node.is_buggy or node.metric.value is None:
        return
    with agent.save_node_lock:
        update_top_candidates(agent, node)
        if agent.best_node is None or node.metric > agent.best_node.metric:
            agent.best_node, agent.best_metric = node, node.metric.value


def execute(controller, gpu, code, node_id, _reset=True):
    source = controller.code / (node_id + '.py')
    source.write_text(code)
    try:
        source_parts(code)
    except Exception as exc:
        receipt = {'id': node_id, 'status': 'invalid_source', 'error': str(exc), 'allocated_gpu_seconds': 0}
        save(controller.root / 'invalid' / (node_id + '.json'), receipt)
    else:
        receipt = controller.job(node_id, source, gpu, controller.c['trial_timeout'])
    from engine.executor import ExecutionResult
    return ExecutionResult(term_out=[json.dumps(receipt)], exec_time=receipt['allocated_gpu_seconds'],
        exc_type=None if receipt['status'] == 'success' else receipt['status'])


def snapshot(agent, controller, reserved, completed, inflight):
    # Host progress contains validation evidence only; test selection happens afterwards.
    nodes = agent.journal.nodes
    save(controller.root / 'progress.json', {'reserved_candidate_slots': reserved,
        'completed_candidates': completed, 'candidate_budget': 44,
        'inflight': sorted(inflight), 'best_validation': agent.best_metric,
        'best_id': agent.best_node.id if agent.best_node else None,
        'memory_records': len(agent.global_memory.records),
        'nodes': [{'id': n.id, 'parent': n.parent.id if n.parent else None,
            'stage': n.stage, 'branch': n.branch_id, 'metric': n.metric.value if n.metric else None,
            'buggy': n.is_buggy, 'plan': n.plan} for n in nodes if n.stage != 'root']})


def build_agent(controller):
    from omegaconf import OmegaConf
    install_transport(controller)
    from engine.agent_search import AgentSearch
    from engine.search_node import Journal
    from agents import result_parse_agent
    from engine import execution, solution_manager
    import engine.agent_search as search_module
    result_parse_agent.determine_metric_direction = lambda agent: setattr(agent, 'metric_maximize', True)
    result_parse_agent.run = parse_result
    execution.validate_executed_node = validate_node
    solution_manager.update_best_solution = persist_best
    # The task contract is already fixed; do not let generic preprocessing rewrite it.
    search_module.clean_task_desc = lambda desc, cfg: desc
    cfg = OmegaConf.load(UPSTREAM / 'config/config.yaml')
    cfg.workspace_dir = str(controller.root / 'native')
    cfg.agent.steps, cfg.agent.seed = 44, 2024
    cfg.agent.initial_drafts = 4
    cfg.agent.search.parallel_search_num = 4
    cfg.agent.search.num_gpus = 4
    cfg.agent.use_stepwise_generation = False  # Full pipeline generation is outside frozen model-only scope.
    cfg.agent.memory_embedding_device = 'cpu'
    cfg.agent.memory_embedding_model_path = str(controller.root / 'embedding-model')
    cfg.coldstart.use_coldstart, cfg.coldstart.description = False, ''
    cfg.agent.code.model = cfg.agent.feedback.model = controller.c['model']
    cfg.agent.code.api_key = cfg.agent.feedback.api_key = ''
    cfg.agent.code.base_url = cfg.agent.feedback.base_url = ''
    save(controller.root / 'native-config.json', OmegaConf.to_container(cfg))
    # Config dataclasses normally convert paths in load_cfg; preserve that here.
    class ConfigView:
        pass
    view = ConfigView()
    for key in cfg:
        setattr(view, key, cfg[key])
    view.workspace_dir = controller.root / 'native'
    view.workspace_dir.mkdir(exist_ok=True)
    base = baseline_source(Path(controller.c['repo']) / 'gagc/templates/nextitnet_perdataset/train.py')
    agent = AgentSearch(BOUNDARY + '\nOriginal baseline model:\n' + base, view, Journal())
    if agent.global_memory is None:
        raise RuntimeError('MLEvolve hybrid memory must initialize; never silently run without memory')
    agent.controller, agent.memory_lock = controller, threading.RLock()
    # Retrieval and index mutation share the upstream memory object across workers.
    retrieve = agent.global_memory.retrieve_similar_records
    def guarded_retrieve(*args, **kwargs):
        with agent.memory_lock:
            return retrieve(*args, **kwargs)
    agent.global_memory.retrieve_similar_records = guarded_retrieve
    agent.data_preview = 'Frozen host data: 12,290 users, 17,566 items; train 131,548 interactions. Validation 200 users, 99 sampled negatives each; seed42 sampling; training seed2024. No dataset access is needed in model code.'
    agent.search_start_time = time.time()
    return agent


def search(agent, controller):
    completed, reserved = 0, 0
    pending = {}
    # Upstream deferred draft generation preserves diversity while training four drafts in parallel.
    drafts = []
    for _ in range(4):
        reserved += 1
        draft = agent.step(node=None, exec_callback=None, execute_immediately=False)
        if draft.stage == 'root' or not getattr(draft, 'pending_execution', False):
            raise RuntimeError('Initial draft generation did not produce a deferred node')
        drafts.append(draft)
        snapshot(agent, controller, reserved, completed, [])
    with ThreadPoolExecutor(max_workers=4) as pool:
        for gpu, draft in zip(controller.c['gpus'], drafts):
            callback = lambda code, nid, reset=True, g=gpu: execute(controller, g, code, nid, reset)
            pending[pool.submit(agent.execute_deferred_node, draft, callback)] = gpu
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                gpu = pending.pop(future)
                node = future.result()
                completed += 1
                snapshot(agent, controller, reserved, completed, pending.values())
                controller.event('candidate_accounted', completed=completed, budget=44,
                    id=node.id, stage=node.stage, best_validation=agent.best_metric)
                if reserved < 44:
                    reserved += 1
                    progress = completed / 44
                    agent.search_start_time = time.time() - progress * agent.acfg.time_limit
                    callback = lambda code, nid, reset=True, g=gpu: execute(controller, g, code, nid, reset)
                    pending[pool.submit(agent.step, node, callback)] = gpu
    if completed != 44 or len(agent.journal.nodes) != 45:
        raise RuntimeError('Budget accounting mismatch')
    return completed


def run(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / 'progress.json').exists() or (root / 'calls').exists():
        raise RuntimeError('Existing run: inspect retained outcomes before any restart; no silent repetition')
    sys.path.insert(0, str(UPSTREAM))
    sys.path.insert(0, str(WORKSPACE / 'staged-autoresearch'))
    random.seed(2024)
    import torch
    torch.set_num_threads(4)
    c = copy.deepcopy(read(WORKSPACE / 'artifacts/mle-star-movies-2024/config.json'))
    c.update(run_dir=str(root), method='MLEvolve upstream engine / frozen NextItNet adaptation',
        candidate_budget=44, model='gpt-6.1-sol', effort='high', upstream_commit=COMMIT,
        time_policy='upstream 12h progress replaced by completed_candidate_slots/44 for selection and fusion windows')
    save(root / 'config.json', c)
    actual = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError('Unexpected upstream commit')
    pins = [*Path(__file__).parent.glob('*.py'), WORKSPACE / 'mle_star_codex/run.py', WORKSPACE / 'mle_star_codex/worker.py']
    pins += list(UPSTREAM.glob('engine/**/*.py')) + list(UPSTREAM.glob('agents/**/*.py')) + list(UPSTREAM.glob('llm/**/*.py'))
    pins += list(UPSTREAM.glob('utils/**/*.py')) + list(UPSTREAM.glob('config/**/*.py'))
    pins += [UPSTREAM / 'config/config.yaml', root / 'embedding-model.json', root / 'embedding-acceptance.json', root / 'environment.txt']
    save(root / 'frozen.json', {'files': {str(p): sha(p) for p in pins}, 'config_sha256': sha(root / 'config.json')})
    controller = NativeController(c)
    baseline = controller.code / 'baseline.py'
    baseline.write_text(baseline_source(Path(c['repo']) / 'gagc/templates/nextitnet_perdataset/train.py'))
    baseline_result = controller.job('baseline', baseline, 0, c['trial_timeout'])
    if baseline_result['status'] != 'success':
        raise RuntimeError('Fresh baseline failed')
    if baseline_result['metrics']['HR@10'] != 0.395:
        raise RuntimeError('Fresh baseline differs from the matched reference')
    agent = build_agent(controller)
    completed = search(agent, controller)
    selected = agent.best_node
    if selected is None:
        raise RuntimeError('No valid candidates')
    save(root / 'selection.json', {'id': selected.id, 'plan': selected.plan,
        'validation_hr10': selected.metric.value, 'source': str(controller.code / (selected.id + '.py'))})
    tests = controller.jobs([
        {'key': 'final-baseline', 'model': baseline, 'mode': 'predict',
         'checkpoint': root / 'trials/baseline/checkpoint.pt'},
        {'key': 'final-selected', 'model': controller.code / (selected.id + '.py'), 'mode': 'predict',
         'checkpoint': root / 'trials' / selected.id / 'checkpoint.pt'}], final=True)
    if any(t['status'] != 'success' for t in tests):
        raise RuntimeError('Final test failed; retained for inspection')
    save(root / 'summary.json', {'status': 'complete', 'candidate_budget': 44,
        'completed_candidates': completed, 'training_calls': sum((d / 'request.json').exists()
        for d in (root / 'trials').iterdir() if d.name not in ('baseline', 'final-baseline', 'final-selected')),
        'baseline_validation': baseline_result['metrics'], 'selected_validation_hr10': selected.metric.value,
        'baseline_test': tests[0]['metrics'], 'selected_test': tests[1]['metrics'],
        'allocated_gpu_seconds': controller.spent(), 'model': c['model'], 'effort': c['effort'],
        'limitations': ['single training seed', 'training-call matched, not token/GPU-time matched',
            'model-only task adapter; stepwise full-pipeline code generation disabled',
            'time-based policy uses candidate-budget progress', 'prior MLE-STAR effort not retroactively changed',
            'test split has been used in earlier studies', 'no prediction ensemble']})
    controller.event('complete', summary=read(root / 'summary.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
    run(args.root.resolve())


if __name__ == '__main__':
    main()
