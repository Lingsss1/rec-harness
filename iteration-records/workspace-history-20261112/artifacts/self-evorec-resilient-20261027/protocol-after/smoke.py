"""Real four-GPU / frozen-grader acceptance and native transport checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys
import time

from .bootstrap import bootstrap
bootstrap()
from .core import Controller, read, save, schema, TEXT
from .prepare import ROOT, WORKSPACE, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('--phase', choices=['gpu', 'native', 'llm', 'draft', 'researcher', 'accept'], required=True)
    args = parser.parse_args()
    config = read(args.config)
    root = Path(config['root']) / 'preflight'
    root.mkdir(exist_ok=True)
    test_config = {**config, 'root': str(root)}
    if args.phase == 'gpu':
        controller = Controller(test_config, 'gpu')
        diagnosis = (WORKSPACE / 'artifacts/Self-EvolveRec-code/examples/sasrec_cds/initial_code/diagnosis_tools.py').read_text()
        def one(gpu):
            return controller.trial('gpu-' + str(gpu), controller.baseline(), gpu, requested=600,
                                    files={'diagnosis_tools.py': diagnosis}, smoke=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(one, config['gpus']))
        if any(r['status'] != 'success' for r in results):
            raise RuntimeError('GPU/diagnostic acceptance failed; inspect preflight/gpu/smoke')
        save(root / 'gpu.json', {'passed': True, 'results': results,
                                'gpu_seconds_outside_search_budget': sum(r['allocated_gpu_seconds'] for r in results)})
    elif args.phase == 'native':
        # Import both native runtimes in separate interpreters to avoid their 'agents' namespace collision.
        import subprocess
        for method in ['mlevolve', 'self_evolverec']:
            command = [sys.executable, '-m', 'amazon_aligned.native_accept', '--method', method]
            subprocess.run(command, check=True, cwd=WORKSPACE)
        save(root / 'native.json', {'passed': True})
    elif args.phase == 'draft':
        import ast
        from .mlevolve import NativeController, build
        controller = NativeController(test_config, 'draft')
        controller.best = {'id': 'not-trained-acceptance-only', 'status': 'not_trained', 'score': None}
        agent = build(controller)
        draft = agent.step(node=None, exec_callback=None, execute_immediately=False)
        ast.parse(draft.code)
        save(root / 'draft.json', {'passed': bool(draft.plan and draft.code),
                                  'native_node': draft.id, 'stage': draft.stage,
                                  'model': config['llm_model'], 'effort': config['llm_effort']})
    elif args.phase == 'researcher':
        import asyncio
        from .self_evolverec import UPSTREAM, install_transport, create_engine
        sys.path.insert(0, str(UPSTREAM))
        controller = Controller(test_config, 'researcher-strict')
        install_transport(controller)
        from self_evolverec import Self_EvolveRec
        Self_EvolveRec._setup_logging = lambda self: None
        engine = create_engine(controller, 0, (UPSTREAM / 'examples/sasrec_cds/initial_code/diagnosis_tools.py').read_text())
        engine.researcher.update_topic('Optimize the four-category Amazon sequential recommender',
                                      'amazon', 'Common fixed RecHarness HR@10 validation')
        idea = asyncio.run(engine.researcher.read_paper('SASRec',
            'Self-attentive sequential recommendation is the provided cold start; explain a concrete improvement.',
            'https://arxiv.org/abs/1808.09781'))
        save(root / 'researcher.json', {'passed': True, 'idea': idea.model_dump(),
                                      'model': config['llm_model'], 'effort': config['llm_effort']})
    elif args.phase == 'llm':
        controller = Controller(test_config, 'llm')
        response = controller.llm('Transport acceptance only: return status=ready and explain in one '
            'short sentence that you will use four datasets, validation HR@10 and no hidden test feedback.',
            schema({'status': {'type': 'string', 'enum': ['ready']}, 'understanding': TEXT}),
            key='transport-' + config['llm_effort'])
        save(root / 'llm.json', {'passed': response['status'] == 'ready', 'response': response,
                                'model': config['llm_model'], 'effort': config['llm_effort']})
    else:
        checks = {name: read(root / (name + '.json'))['passed']
                  for name in ['gpu', 'native', 'llm', 'draft', 'researcher']}
        for name in ['llm', 'draft', 'researcher']:
            record = read(root / (name + '.json'))
            checks[name] = checks[name] and record['model'] == config['llm_model'] and record['effort'] == config['llm_effort']
        save(root / 'acceptance.json', {'passed': all(checks.values()), 'checks': checks, 'time': time.time(),
                                      'config_sha256': sha(args.config)})
        if not all(checks.values()):
            raise RuntimeError('Preflight checks did not all pass')
    print('PREFLIGHT_' + args.phase.upper() + '_PASSED', flush=True)


if __name__ == '__main__':
    main()
