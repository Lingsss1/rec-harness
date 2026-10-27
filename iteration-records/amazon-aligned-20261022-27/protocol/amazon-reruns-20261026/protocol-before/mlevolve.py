"""Actual MLEvolve AgentSearch with the common full-program task boundary."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import logging
from pathlib import Path
import random
import sys
import threading
import time
import types

from .core import BudgetExhausted, Controller, TASK, save
from .prepare import DATASETS, WORKSPACE

UPSTREAM = WORKSPACE / 'artifacts/MLEvolve'


class NativeController(Controller):
    def query_native(self, prompt, output_schema=None):
        from mlevolve_codex.run import transport_schema
        return self.llm(prompt, transport_schema(output_schema) if output_schema else None)


def build(controller):
    sys.path.insert(0, str(UPSTREAM))
    # Upstream agents is a namespace directory; the separately installed SDK has
    # a regular package with the same name. Bind the native namespace explicitly.
    native_agents = types.ModuleType('agents')
    native_agents.__path__ = [str(UPSTREAM / 'agents')]
    sys.modules['agents'] = native_agents
    import llm
    def query(system_message=None, user_message=None, func_spec=None, **kwargs):
        prompt = '\n'.join(llm.compile_prompt_to_md(x) for x in (system_message, user_message) if x)
        return controller.query_native(prompt, func_spec.json_schema if func_spec else None)
    def generate(prompt, json_schema=None, **kwargs):
        value = controller.query_native(llm.compile_prompt_to_md(prompt), json_schema)
        return json.dumps(value) if json_schema else value
    llm.query, llm.generate = query, generate
    from omegaconf import OmegaConf
    from engine.agent_search import AgentSearch
    from engine.search_node import Journal
    from agents import result_parse_agent
    from engine import execution, solution_manager
    import engine.agent_search as search_module
    from mlevolve_codex.run import parse_result, validate_node, persist_best
    result_parse_agent.determine_metric_direction = lambda agent: setattr(agent, 'metric_maximize', True)
    result_parse_agent.run = parse_result
    execution.validate_executed_node = validate_node
    solution_manager.update_best_solution = persist_best
    search_module.clean_task_desc = lambda description, cfg: description
    cfg = OmegaConf.load(UPSTREAM / 'config/config.yaml')
    cfg.agent.initial_drafts = cfg.agent.search.parallel_search_num = cfg.agent.search.num_gpus = 4
    cfg.agent.steps, cfg.agent.seed, cfg.agent.time_limit = 1000, 42, 43200
    cfg.agent.use_stepwise_generation = False
    cfg.agent.memory_embedding_device = 'cpu'
    cfg.agent.memory_embedding_model_path = str(WORKSPACE / 'artifacts/mlevolve-movies-2024/embedding-model')
    cfg.agent.code.model = cfg.agent.feedback.model = controller.c['llm_model']
    cfg.agent.code.api_key = cfg.agent.feedback.api_key = ''
    cfg.coldstart.use_coldstart = False
    cfg.workspace_dir = str(controller.root / 'native')
    save(controller.root / 'native-config.json', OmegaConf.to_container(cfg))
    class ConfigView:
        pass
    view = ConfigView()
    for key in cfg:
        setattr(view, key, cfg[key])
    view.workspace_dir = controller.root / 'native'
    view.workspace_dir.mkdir(exist_ok=True)
    agent = AgentSearch(TASK + '\nOriginal cold-start train.py:\n' + controller.baseline(), view, Journal())
    if agent.global_memory is None:
        raise RuntimeError('Native MLEvolve memory initialization failed')
    agent.controller, agent.memory_lock = controller, threading.RLock()
    retrieve = agent.global_memory.retrieve_similar_records
    def guarded(*args, **kwargs):
        with agent.memory_lock:
            return retrieve(*args, **kwargs)
    agent.global_memory.retrieve_similar_records = guarded
    agent.data_preview = json.dumps({'datasets': DATASETS, 'input_format': 'user item per line',
                                    'baseline_validation': controller.brief(controller.best),
                                    'held_out_test': 'unavailable until final selection'})
    agent.search_start_time = time.time()
    return agent


def execute(controller, gpu, code, node_id, reset=True, allowance=None):
    receipt = controller.trial(node_id, code, gpu, requested=allowance)
    if receipt['status'] == 'success':
        receipt['metrics'] = {'HR@10': receipt['score']}
    from engine.executor import ExecutionResult
    return ExecutionResult(term_out=[json.dumps(receipt)], exec_time=receipt['allocated_gpu_seconds'],
                           exc_type=None if receipt['status'] == 'success' else receipt['status'])


def run(config):
    random.seed(42)
    controller = NativeController(config, 'mlevolve')
    baseline = controller.trial('baseline', controller.baseline(), config['gpus'][0],
                                requested=config['trial_timeout_seconds'])
    if baseline['status'] != 'success':
        raise RuntimeError('MLEvolve common baseline failed')
    agent = build(controller)
    drafts = [agent.step(node=None, exec_callback=None, execute_immediately=False) for _ in range(4)]
    pending, completed = {}, 0
    allowance = min(config['trial_timeout_seconds'], controller.ledger.available() / 4 - 3)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for gpu, draft in zip(config['gpus'], drafts):
            callback = lambda code, nid, reset=True, g=gpu: execute(controller, g, code, nid, reset, allowance)
            pending[pool.submit(agent.execute_deferred_node, draft, callback)] = gpu
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                gpu = pending.pop(future)
                try:
                    node = future.result()
                except BudgetExhausted:
                    continue
                completed += 1
                save(controller.root / 'progress.json', {
                    'completed_candidates': completed, 'available_gpu_seconds': controller.ledger.available(),
                    'best': controller.brief(controller.best), 'memory_records': len(agent.global_memory.records),
                    'native_nodes': len(agent.journal.nodes)})
                if controller.ledger.available() < config['minimum_trial_seconds'] + 3 or completed >= 1000:
                    continue
                state = json.loads(controller.ledger.path.read_text())
                used = state['spent_gpu_seconds'] / config['search_budget_gpu_seconds']
                agent.search_start_time = time.time() - used * agent.acfg.time_limit
                callback = lambda code, nid, reset=True, g=gpu: execute(controller, g, code, nid, reset)
                pending[pool.submit(agent.step, node, callback)] = gpu
    return controller.finish()
