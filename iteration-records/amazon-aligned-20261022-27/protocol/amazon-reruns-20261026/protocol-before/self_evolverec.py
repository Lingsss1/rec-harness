"""Four native Self_EvolveRec trajectories sharing one GPU-time ledger."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import logging
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

from .core import BudgetExhausted, Controller, TASK, save, read
from .prepare import WORKSPACE

UPSTREAM = WORKSPACE / 'artifacts/Self-EvolveRec-code'
INTERFACE = '''def self_evolverec_interface():
    raise RuntimeError('Training and grading are executed by the common host boundary.')
'''
DIAGNOSIS_CONTRACT = '''
Diagnostic task adapter: diagnosis_tools.py must retain class DiagnosisProbe(view, device)
and method run_full_diagnosis(train_loader), returning metrics and metric_definitions.
view.model is the actual trained nn.Module. view.item_emb, log2feats(sequence),
predict_score(features, items), user_train, meta_data, review_data implement the
initial SASRec compatibility contract. Adapt probes when changing the architecture.
train_loader yields (None, sequence, positive, negative), tensors of shape B x L.
Only public interaction histories, item titles, descriptions and timestamps are available.
Review ratings/text, item ratings/categories/prices/stores are NOT available in this
shared RecHarness task. Never invent them. Simulator diversity/conformity traits are
explicitly unobserved. Twenty real training users (five/category) receive full-catalog
unseen-item recommendations, as the native simulator requires. Selection uses the
common macro validation HR@10, replacing the native HR@5/NDCG@5 combined score.
'''


def parse_files(code):
    cleaned = re.sub(r'```\w*\n|```', '', code)
    matches = re.findall(r'^# === (.+?) ===\n(.*?)(?=^# === .+? ===\n|\Z)', cleaned, re.M | re.S)
    files = {}
    for name, contents in matches:
        if name in files:
            raise ValueError('Duplicate native code file: ' + name)
        if name not in {'main_code.py', 'models.py', 'diagnosis_tools.py', 'self_evolverec_interface.py'}:
            raise ValueError('Unexpected native file: ' + name)
        files[name] = contents
    if not {'main_code.py', 'diagnosis_tools.py'} <= set(files):
        raise ValueError('Native code must contain main_code.py and diagnosis_tools.py')
    return files


def install_transport(controller):
    import agents
    import agents.tracing
    agents.set_tracing_disabled(True)
    agents.tracing.trace = lambda *args, **kwargs: nullcontext()
    class Result:
        def __init__(self, value, history):
            self.final_output, self.history = value, history
        def final_output_as(self, output_type):
            return self.final_output
        def to_input_list(self):
            value = self.final_output
            text = value if isinstance(value, str) else value.model_dump_json()
            return self.history + [{'role': 'assistant', 'content': text}]
    async def run(agent, input=None, *args, **kwargs):
        history = input if isinstance(input, list) else [{'role': 'user', 'content': str(input)}]
        prompt = DIAGNOSIS_CONTRACT + '\nNATIVE ROLE: ' + agent.name + '\n' + str(agent.instructions)
        prompt += '\nINPUT:\n' + json.dumps(history, ensure_ascii=False)
        output_type = agent.output_type
        output_schema = output_type.model_json_schema() if output_type not in (None, str) else None
        value = await asyncio.to_thread(controller.llm, prompt, output_schema, bool(agent.tools))
        if output_schema is not None:
            value = output_type.model_validate(value)
        return Result(value, history)
    agents.Runner.run = staticmethod(run)


def feedback(controller, receipt):
    """Use native Simulator behavior/interview/aggregation and diagnostic prompts."""
    from simulator import simulator as simulator_module, prompts_list
    proxy = SimpleNamespace(responses=SimpleNamespace(create=lambda **kwargs:
        SimpleNamespace(output_text=controller.llm(DIAGNOSIS_CONTRACT + '\n' + str(kwargs['input'])))))
    simulator_module.OpenAI = prompts_list.OpenAI = lambda **kwargs: proxy
    os.environ['OPENAI_API_KEY'] = 'codex-transport-placeholder-not-a-key'
    trial = Path(receipt['output'])
    per_user = {}
    for bundle in read(trial / 'simulation-input.json'):
        view = SimpleNamespace(**{key: bundle[key] for key in ['user_train', 'meta_data', 'review_data']})
        simulator = simulator_module.Simulator(controller.c['llm_model'], view)
        simulator.item_statistic = bundle['popularity']
        for user, hist in bundle['user_train']['History'].items():
            simulator.personality['activity'][user] = simulator.assign_score(len(hist), bundle['activity_thresholds'])
            simulator.personality['diversity'][user] = 1
            simulator.personality['conformity'][user] = 1
        simulator.defined_personality['diversity'][1] = 'UNOBSERVED: no item-category labels are available; do not invent a diversity trait.'
        simulator.defined_personality['conformity'][1] = 'UNOBSERVED: no ratings are available; do not invent a conformity trait.'
        def one(pair):
            return pair['user'], simulator.conduct_simulation(pair['user'], pair['recommendations'])
        with ThreadPoolExecutor(max_workers=4) as pool:
            for user, report in pool.map(one, bundle['pairs']):
                per_user[bundle['dataset'] + ':' + user] = report
    raw = read(trial / 'diagnosis.json')
    merged = {'metrics': {}, 'metric_definitions': {}}
    for dataset, result in raw.items():
        merged['metrics'].update({dataset + ':' + k: v for k, v in result['metrics'].items()})
        merged['metric_definitions'].update({dataset + ':' + k: v for k, v in result['metric_definitions'].items()})
    diagnosis = __import__('importlib.util', fromlist=['spec_from_file_location'])
    spec = diagnosis.spec_from_file_location('_diagnosis_for_prompt', trial / 'diagnosis_tools.py')
    module = diagnosis.module_from_spec(spec)
    spec.loader.exec_module(module)
    mathematical = prompts_list.math_diagnosis_agent(module.diagnosis_interpreter_prompt(merged), controller.c['llm_model'])
    critical, simulated, diagnosed = prompts_list.get_summarized_suggestion(
        per_user, merged, mathematical, controller.c['llm_model'], simulator_module.aggregator_prompt)
    result = {'simulator_comment': simulated, 'diagnosis_comment': diagnosed,
              'native_critical_flag': critical,
              'critical_flag_policy': 'feedback only; external HR@10 remains the sole selection metric'}
    save(trial / 'directional-feedback.json', {'feedback': result, 'users': per_user})
    return result


async def run_async(config):
    sys.path.insert(0, str(UPSTREAM))
    controller = Controller(config, 'self_evolverec')
    install_transport(controller)
    from self_evolverec import Self_EvolveRec
    import problem as native_problem
    original_diagnosis = (UPSTREAM / 'examples/sasrec_cds/initial_code/diagnosis_tools.py').read_text()
    baseline = await asyncio.to_thread(controller.trial, 'baseline', controller.baseline(), config['gpus'][0],
                    config['trial_timeout_seconds'], {'diagnosis_tools.py': original_diagnosis})
    if baseline['status'] != 'success':
        raise RuntimeError('Self-EvolveRec common baseline/diagnostic acceptance failed')
    baseline_feedback = await asyncio.to_thread(feedback, controller, baseline)
    key_lock = asyncio.Lock()
    counter = 0
    problem_slots = {}

    async def evaluate(self, code, program_id, is_initial=False):
        nonlocal counter
        if is_initial:
            return {'combined_score': baseline['score'], 'simulator_comment': baseline_feedback['simulator_comment'],
                    'diagnosis_comment': baseline_feedback['diagnosis_comment']}, code
        gpu = problem_slots[str(self.workspace)]
        current = code
        for retry in range(self.max_retry_times + 1):
            async with key_lock:
                counter += 1
                key = f'candidate-{counter:05d}-gpu{gpu}'
            try:
                files = parse_files(current)
                extras = {k: v for k, v in files.items() if k in {'models.py', 'diagnosis_tools.py'}}
                receipt = await asyncio.to_thread(controller.trial, key, files['main_code.py'], gpu,
                                                  None, extras)
            except ValueError as exc:
                receipt = {'status': 'invalid_source', 'error': str(exc)}
            if receipt['status'] == 'success':
                reports = await asyncio.to_thread(feedback, controller, receipt)
                return {'combined_score': receipt['score'], 'simulator_comment': reports['simulator_comment'],
                        'diagnosis_comment': reports['diagnosis_comment']}, current
            if retry == self.max_retry_times:
                return {'combined_score': -1., 'simulator_comment': 'Execution failed; no simulated-user observations.',
                        'diagnosis_comment': receipt.get('error', 'Failed execution')}, current
            ok, current, _ = await self._debugging(current, receipt.get('error', 'Execution failed'), retry)
            if not ok:
                return {'combined_score': -1., 'simulator_comment': 'Execution failed; no user observations.',
                        'diagnosis_comment': receipt.get('error', 'Execution failed')}, current
        raise RuntimeError('Unreachable debugging state')
    native_problem.Problem.evaluate = evaluate
    # Keep one host logging setup while all four native controllers share the process.
    Self_EvolveRec._setup_logging = lambda self: None
    engines = []
    for t, gpu in enumerate(config['gpus']):
        engine = create_engine(controller, t, original_diagnosis)
        problem_slots[str(engine.problem.workspace)] = gpu
        engines.append(engine)
    async def trajectory(engine):
        try:
            return await engine.run(iterations=1000)
        except BudgetExhausted as exc:
            controller.event('trajectory_budget_stop', workspace=engine.workspace, reason=str(exc))
            return None
    results = await asyncio.gather(*(trajectory(engine) for engine in engines), return_exceptions=True)
    failures = [str(result) for result in results if isinstance(result, BaseException)]
    if failures:
        save(controller.root / 'trajectory-errors.json', failures)
        raise RuntimeError('Native Self-EvolveRec trajectories failed: ' + '; '.join(failures))
    return await asyncio.to_thread(controller.finish)


def create_engine(controller, trajectory, original_diagnosis):
    from omegaconf import OmegaConf
    from self_evolverec import Self_EvolveRec
    config = controller.c
    native_cfg = OmegaConf.load(UPSTREAM / 'configs/config_baseline.yaml')
    native_cfg.workspace = str(controller.root / 'native' / f'trajectory-{trajectory}')
    native_cfg.problem = 'amazon'
    native_cfg.max_iterations = 1000
    native_cfg.max_debug_retry = 10
    native_cfg.log_dir = str(Path(native_cfg.workspace) / 'logs')
    native_cfg.database.random_seed = 42 + trajectory
    native_cfg.checkpoint = str(Path(native_cfg.workspace) / 'checkpoints')
    for role in ['planner', 'searcher', 'writer']:
        native_cfg.researcher[role] = config['llm_model']
    for role in ['developer', 'debugger']:
        native_cfg.coder[role] = config['llm_model']
    native_cfg.researcher.reasoning_effort = native_cfg.coder.reasoning_effort = config['llm_effort']
    workspace = Path(native_cfg.workspace) / 'amazon'
    initial = workspace / 'initial_code'
    initial.mkdir(parents=True, exist_ok=True)
    (initial / 'main_code.py').write_text(controller.baseline())
    (initial / 'diagnosis_tools.py').write_text(original_diagnosis)
    (initial / 'self_evolverec_interface.py').write_text(INTERFACE)
    save(workspace / 'info.json', {'problem': {'description': TASK, 'interface': 'self_evolverec_interface.py'},
         'initial_idea': {'title': 'SASRec RecHarness cold start', 'content': TASK,
                          'supplement': 'https://arxiv.org/abs/1808.09781'}})
    save(workspace / 'info_sim_diag.json', {'problem_diagnosis': {'description': DIAGNOSIS_CONTRACT},
         'initial_idea_diagnosis': {'title': 'Native Self-EvolveRec DiagnosisProbe',
                                  'content': DIAGNOSIS_CONTRACT, 'supplement': ''}})
    save(workspace / 'native-config.json', OmegaConf.to_container(native_cfg))
    return Self_EvolveRec(native_cfg, TASK + '\n' + DIAGNOSIS_CONTRACT)


def run(config):
    return asyncio.run(run_async(config))
