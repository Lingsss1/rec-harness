"""Keep DiscoveryRSI's native hypothesis/critique/selection/replay loop.

The task adapter compiles four hypotheses and evaluates them on the existing
four-GPU trainer each episode. Native scientific-critic scores select the
hypothesis retained for the next episode; validation HR@10 independently selects
the final checkpoint. No previous method's winning model enters agent context.
"""
from dataclasses import asdict
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from mle_star_codex.run import Controller, baseline_source, object_schema, read, save, sha, source_parts

WORKSPACE = Path(__file__).resolve().parents[1]
UPSTREAM = WORKSPACE / 'artifacts/DiscoveryRSI-source'
ROOT = WORKSPACE / 'artifacts/discoveryrsi-movies-2024-v2'
sys.path.insert(0, str(UPSTREAM / 'src'))
from discovery_rsi.core import DiscoveryLoop, ReplayBuffer
from discovery_rsi.harness import ResearchHarness
from discovery_rsi.manifest import ResearchManifest

TEXT = {'type': 'string'}
STRINGS = {'type': 'array', 'items': TEXT}
PROPOSALS = object_schema({'candidates': {'type': 'array', 'minItems': 4, 'maxItems': 4,
    'items': object_schema({'claim': TEXT, 'plan': TEXT, 'alternatives': STRINGS, 'evidence': STRINGS})}})
CRITIQUE = object_schema({'score': {'type': 'number', 'minimum': 0, 'maximum': 10}, 'critique': TEXT})
COMPILED = object_schema({'models': {'type': 'array', 'minItems': 4, 'maxItems': 4,
    'items': object_schema({'id': TEXT, 'source': TEXT, 'implementation_note': TEXT})}})
BOUNDARY = '''Frozen Movies_and_TV / NextItNet comparison, training seed 2024.
Use the public DiscoveryRSI hypothesis-development workflow for this task.
Retain a causal dilated-convolution NextItNet backbone. The host alone owns data,
preprocessing, BPR loss, Adam, gradient clipping=5, training, validation, checkpoint
selection, negative sampling and final testing. Validation HR@10 is higher-is-better.
The fixed trainer uses at most 200 epochs, 200 batches/epoch, validation every 5
epochs and patience 20; evaluation uses seed42, 200 validation users, 99 negatives.
There are 11 episodes with four proposals each: exactly 44 candidate attempts.
Every invalid source or training failure consumes its slot; no out-of-budget repair.
Model source has ONLY these imports: import math; import torch; import torch.nn as nn.
It has exactly eight numeric constants MAXLEN,HIDDEN_UNITS,NUM_BLOCKS,KERNEL_SIZE,
DROPOUT_RATE,LR,WEIGHT_DECAY,BATCH_SIZE, and two nn.Module classes NextItNetBlock
and NextItNetModel (methods only). Preserve constructors and the forward,
encode_sequence and score_candidates interfaces. Constants must respect:
MAXLEN integer[4,512],HIDDEN_UNITS integer[8,256],NUM_BLOCKS integer[1,16],
KERNEL_SIZE integer[1,9],DROPOUT_RATE[0,.8],LR[1e-5,.02],WEIGHT_DECAY[0,.1],
BATCH_SIZE integer[16,512]. No file/network access, extra imports, top-level calls,
printing, training loops or metric implementation in model code. No prediction
ensembles. Evaluator v2 permits separate, normalized, or projected target vectors
provided training and candidate scores obey causality and target independence.
Histories are right aligned. The raw baseline's count_nonzero-1 readout can select
padding or an earlier position: distinguish correctness fixes from architecture gains.
Use only supplied evidence; missing results are unknown. Do not invent measurements.
Do not inspect the filesystem, use tools, read other runs, or edit files. Return the
requested structured JSON. The LLM must not access test data or test metrics.
'''


def choose_best(receipts):
    valid = [r for r in receipts if r['status'] == 'success']
    if not valid:
        raise RuntimeError('No valid model')
    return max(valid, key=lambda r: r['metrics']['HR@10'])  # first on ties


class CodexController(Controller):
    def query(self, key, prompt, schema):
        directory = self.root / 'calls' / key
        directory.mkdir(parents=True, exist_ok=True)
        request = {'prompt': BOUNDARY + '\n' + prompt, 'schema': schema}
        if (directory / 'started.json').exists():
            raise RuntimeError('Call already started; inspect before recovery: ' + key)
        save(directory / 'request.json', request)
        save(directory / 'schema.json', schema)
        command = [self.c['codex'], 'exec', '--ephemeral', '--skip-git-repo-check',
            '--ignore-user-config', '--ignore-rules', '--model', self.c['model'],
            '--sandbox', 'read-only', '-c', 'approval_policy="never"',
            '-c', 'web_search="disabled"', '-c', 'model_reasoning_effort="' + self.c['effort'] + '"',
            '--json', '--output-schema', str(directory / 'schema.json'),
            '-o', str(directory / 'answer.json'), '-C', str(self.codex_cwd), '-']
        save(directory / 'started.json', {'command': command, 'time': time.time()})
        self.event('codex_started', key=key)
        start = time.monotonic()
        with (directory / 'events.jsonl').open('w') as out, (directory / 'stderr.log').open('w') as err:
            result = subprocess.run(command, input=request['prompt'], text=True, stdout=out,
                stderr=err, timeout=1200, env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
        if result.returncode or not (directory / 'answer.json').exists():
            raise RuntimeError('Codex transport failed; retain original call: ' + key)
        events = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines() if line.strip()]
        completed = [e for e in events if e.get('type') == 'turn.completed']
        if not completed or any(e.get('type') == 'turn.failed' for e in events):
            raise RuntimeError('Codex turn did not complete: ' + key)
        response = read(directory / 'answer.json')
        save(directory / 'response.json', response)
        save(directory / 'usage.json', {'usage': completed[-1].get('usage', {}),
            'elapsed_seconds': time.monotonic() - start, 'ephemeral_fresh_call': True})
        self.event('codex_completed', key=key)
        return response


class PromptAdapter:
    """ModelAdapter protocol implementation; native lead/critic prompts stay intact."""
    def __init__(self, controller):
        self.controller = controller
        self.key = None
        self.schema = None

    def complete(self, prompt):
        if self.key is None or self.schema is None:
            raise RuntimeError('Native role was not registered')
        return json.dumps(self.controller.query(self.key, prompt, self.schema), ensure_ascii=False)


class AuditReplay(ReplayBuffer):
    def __init__(self, harness):
        super().__init__(harness.controller.root / 'episodes.jsonl')
        self.harness = harness

    def append(self, episode):
        super().append(episode)
        self.harness.finish_episode(episode)


class MoviesHarness(ResearchHarness):
    def __init__(self, controller, baseline):
        manifest = ResearchManifest.from_dict({
            'name': 'Movies_NextItNet_seed2024',
            'question': 'Develop falsifiable improvements to causal NextItNet validation HR@10 on Movies_and_TV under the supplied frozen training protocol.',
            'domains': ['sequential_recommendation'],
            'roles': ['lead', 'generation', 'critique', 'evolution', 'compiler'],
            'max_candidates': 4, 'evidence_policy': 'host-owned training and validation receipts only',
            'metadata': {'seed': 2024, 'candidate_budget': 44, 'check_version': 2,
                'test_access': False, 'training_workers': 4},
        })
        super().__init__(PromptAdapter(controller), controller.c['episodes'],
            str(controller.root / 'episodes.jsonl'), manifest,
            str(UPSTREAM / 'plugins/discovery-rsi/roles'))
        self.controller = controller
        self.baseline = baseline
        self.receipts = {'baseline': baseline}
        self.proposals = {}
        self.implementations = {}
        self.round_receipts = []
        self.episodes_completed = 0
        self.reserved = 0
        self.loop = None
        save(controller.root / 'research-manifest.json', manifest.to_dict())

    def _context(self):
        history = [] if self.loop is None else self.loop.history
        details = []
        for candidate in history:
            details.append({'hypothesis': asdict(candidate),
                'host_result': self.controller.brief(self.receipts[candidate.id])})
        parent = history[-1].id if history else 'baseline'
        model = self.controller.code / (parent + '.py')
        packet = {'retained_hypothesis_path': details, 'code_parent_id': parent,
            'parent_source': model.read_text(),
            'baseline_validation': self.baseline['metrics'],
            'current_episode_results': [self.controller.brief(r) for r in self.round_receipts],
            'completed_candidate_attempts': len(self.receipts) - 1,
            'best_validation_so_far': choose_best(list(self.receipts.values()))['metrics'],
            'fixed_dataset_summary': {'users': 12290, 'items': 17566, 'training_interactions': 131548}}
        return super()._context() + '\nFROZEN TASK EVIDENCE:\n' + json.dumps(packet, ensure_ascii=False)

    def _candidates(self, number, history):
        self.round_receipts = []
        self.adapter.key, self.adapter.schema = f'propose-{number:02d}', PROPOSALS
        candidates = super()._candidates(number, history)
        if len(candidates) != 4:
            raise RuntimeError('Expected exactly four native hypotheses; no silent padding or retry')
        self.reserved += 4
        for candidate in candidates:
            self.proposals[candidate.id] = asdict(candidate)
        save(self.controller.root / 'rounds' / f'episode-{number:02d}-proposals.json',
             [asdict(c) for c in candidates])
        self.snapshot('compiling')
        compiled = self.controller.query(f'compile-{number:02d}',
            'You are the DiscoveryRSI computational investigation compiler. Implement each of '
            'the four prespecified hypotheses as one complete NextItNet model file. Preserve '
            'candidate IDs exactly. Keep the declared comparator, mechanism and falsification '
            'rule coherent; do not add unrelated changes. A failed prior implementation may be '
            'repaired within these candidate slots. Return source code, not an experiment runner.\n'
            + json.dumps({'hypotheses': [asdict(c) for c in candidates]}, ensure_ascii=False)
            + self._context(), COMPILED)['models']
        if len(compiled) != 4 or {x['id'] for x in compiled} != {c.id for c in candidates}:
            raise RuntimeError('Compiler changed candidate IDs or count')
        compiled_by_id = {x['id']: x for x in compiled}
        specs, invalid = [], {}
        for candidate in candidates:
            result = compiled_by_id[candidate.id]
            path = self.controller.code / (candidate.id + '.py')
            path.write_text(result['source'])
            self.implementations[candidate.id] = result['implementation_note']
            try:
                source_parts(result['source'])
            except Exception as exc:
                receipt = {'id': candidate.id, 'status': 'invalid_source', 'error': str(exc),
                    'source': str(path), 'source_sha256': sha(path), 'allocated_gpu_seconds': 0}
                save(self.controller.root / 'invalid' / (candidate.id + '.json'), receipt)
                invalid[candidate.id] = receipt
            else:
                specs.append({'key': candidate.id, 'model': path})
        save(self.controller.root / 'rounds' / f'episode-{number:02d}-compiled.json', compiled)
        self.snapshot('training')
        trained = {r['id']: r for r in self.controller.jobs(specs)} if specs else {}
        for candidate in candidates:
            receipt = (invalid | trained)[candidate.id]
            self.receipts[candidate.id] = receipt
            self.round_receipts.append(receipt)
            candidate.evidence.append('HOST VALIDATION RECEIPT: ' + json.dumps(self.controller.brief(receipt)))
        self.snapshot('critiquing')
        return candidates

    def _critique(self, candidate):
        self.adapter.key, self.adapter.schema = 'critic-' + candidate.id, CRITIQUE
        # Preserve the native 0..10 scientific-quality score for native selection.
        # It is deliberately distinct from the validation metric used for final deployment.
        score, critique = super()._critique(candidate)
        if not 0 <= score <= 10:
            raise RuntimeError('Native critic score outside 0..10')
        save(self.controller.root / 'critiques' / (candidate.id + '.json'),
            {'id': candidate.id, 'score': score, 'critique': critique,
             'host_result': self.controller.brief(self.receipts[candidate.id])})
        return score, critique

    def finish_episode(self, episode):
        self.episodes_completed = episode.number
        selected = next(c for c in episode.candidates if c.id == episode.selected_id)
        save(self.controller.root / 'rounds' / f'episode-{episode.number:02d}.json',
            {'episode': asdict(episode), 'host_results': self.round_receipts,
             'retention_rule': 'upstream max((scientific_critic_score, candidate_id))',
             'retained_validation_hr10': self.receipts[selected.id].get('metrics', {}).get('HR@10'),
             'best_validation_id': choose_best(list(self.receipts.values()))['id']})
        self.controller.event('episode_completed', number=episode.number,
            retained_id=episode.selected_id, retained_scientific_score=selected.score,
            completed=len(self.receipts) - 1,
            best_validation=choose_best(list(self.receipts.values()))['metrics']['HR@10'])
        self.snapshot('between_episodes')

    def snapshot(self, phase):
        save(self.controller.root / 'progress.json', {'phase': phase,
            'episodes_completed': self.episodes_completed, 'episodes_total': self.iterations,
            'candidate_budget': self.iterations * 4, 'reserved_candidate_slots': self.reserved,
            'completed_candidates': len(self.receipts) - 1,
            'successful_candidates': sum(r['status'] == 'success' for k, r in self.receipts.items() if k != 'baseline'),
            'best_id': choose_best(list(self.receipts.values()))['id'],
            'best_validation': choose_best(list(self.receipts.values()))['metrics']['HR@10'],
            'nodes': [{'proposal': self.proposals[k], 'result': self.controller.brief(r)}
                      for k, r in self.receipts.items() if k != 'baseline']})

    def run(self):
        self.loop = DiscoveryLoop(lambda n, h: self._candidates(n, list(h)),
            self._critique, self.iterations, AuditReplay(self))
        return self.loop.run()


def config(root):
    c = copy.deepcopy(read(WORKSPACE / 'artifacts/mle-star-movies-2024/config.json'))
    c.update(run_dir=str(root), method='DiscoveryRSI public native hypothesis loop / frozen NextItNet adaptation',
        model='gpt-6.1-sol', effort='high', check_version=2, episodes=11,
        candidates_per_episode=4, candidate_budget=44, gpus=[0, 1, 2, 3],
        codex='/root/.nvm/versions/node/v24.22.0/bin/codex',
        native_retention='scientific critic score, then candidate ID',
        final_selection='highest validation HR@10 across all candidates and baseline; first on ties')
    return c


def run(root=ROOT):
    started = time.monotonic()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'controller.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if any((root / name).exists() for name in ['progress.json', 'calls', 'frozen.json']):
        raise RuntimeError('Existing experiment; inspect retained outcomes before recovery')
    c = config(root)
    c['upstream_commit'] = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if subprocess.check_output(['git', '-C', str(UPSTREAM), 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('Upstream source must remain unmodified')
    save(root / 'config.json', c)
    from mle_star_codex.worker import CHECK_VERSION
    if CHECK_VERSION != 2:
        raise RuntimeError('Evaluator version 2 required')
    pins = list(Path(__file__).parent.glob('*.py')) + list((UPSTREAM / 'src').rglob('*.py'))
    pins += list((UPSTREAM / 'plugins/discovery-rsi/roles').glob('*.md'))
    pins += [WORKSPACE / 'mle_star_codex/worker.py', WORKSPACE / 'mle_star_codex/run.py']
    reference = read(WORKSPACE / 'artifacts/mle-star-movies-2024/frozen.json')
    for name, expected in reference['data_hashes'].items():
        if sha(Path(c['data_dir']) / 'trainval' / name) != expected:
            raise RuntimeError('Frozen training/validation data changed: ' + name)
    if sha(Path(c['data_dir']) / 'test/Movies_and_TV_test.txt') != reference['heldout_test_sha256']:
        raise RuntimeError('Frozen test split changed')
    save(root / 'frozen.json', {'files': {str(p): sha(p) for p in pins},
        'config_sha256': sha(root / 'config.json'),
        'data_hashes': reference['data_hashes'], 'heldout_test_sha256': reference['heldout_test_sha256'],
        'upstream_commit': c['upstream_commit']})
    controller = CodexController(c)
    baseline_path = controller.code / 'baseline.py'
    baseline_path.write_text(baseline_source(Path(c['repo']) / 'gagc/templates/nextitnet_perdataset/train.py'))
    if sha(baseline_path) != '715e2a27453c8e4d1bdc4854154a71ab3cbdc9b492728307e46a3c4dde2c41be':
        raise RuntimeError('Baseline differs from prior comparison')
    baseline = controller.jobs([{'key': 'baseline', 'model': baseline_path}])[0]
    if baseline['status'] != 'success' or baseline['metrics']['HR@10'] != 0.395:
        raise RuntimeError('Fresh GPU baseline failed or differs from reference')
    save(root / 'baseline-summary.json', baseline)
    harness = MoviesHarness(controller, baseline)
    episodes = harness.run()
    if len(episodes) != 11 or len(harness.receipts) != 45:
        raise RuntimeError('44-attempt budget not completed')
    selected = choose_best(list(harness.receipts.values()))
    save(root / 'selection.json', {'id': selected['id'], 'validation': selected['metrics'],
        'source': selected['source'], 'source_sha256': selected['source_sha256'],
        'checkpoint_sha256': selected['checkpoint_sha256'],
        'selection_rule': c['final_selection'], 'test_not_used_for_selection': True,
        'native_retained_path': [candidate.id for candidate in harness.loop.history]})
    pins = read(root / 'frozen.json')
    if any(sha(path) != value for path, value in pins['files'].items()):
        raise RuntimeError('Frozen source changed during search')
    tests = controller.jobs([
        {'key': 'final-baseline', 'model': baseline_path, 'mode': 'test',
         'checkpoint': root / 'trials/baseline/checkpoint.pt'},
        {'key': 'final-selected', 'model': Path(selected['source']), 'mode': 'test',
         'checkpoint': root / 'trials' / selected['id'] / 'checkpoint.pt'}], final=True)
    if any(t['status'] != 'success' or t['rows_sha256'] != '7e2fb8a72364535829f8677ac7532232a4ac1aff5b37dbd564c5ad3cd817b5aa' for t in tests):
        raise RuntimeError('Final test failed or test rows differ from comparison')
    candidates = [r for k, r in harness.receipts.items() if k != 'baseline']
    summary = {'status': 'complete', 'method': c['method'], 'seed': 2024,
        'upstream_commit': c['upstream_commit'], 'check_version': 2,
        'candidate_budget': 44, 'completed_candidates': len(candidates),
        'training_calls': sum((root / 'trials' / r['id'] / 'request.json').exists() for r in candidates),
        'successful_candidates': sum(r['status'] == 'success' for r in candidates),
        'failed_candidates': sum(r['status'] != 'success' for r in candidates),
        'episodes': 11, 'llm_calls': len(list((root / 'calls').glob('*/response.json'))),
        'model': c['model'], 'effort': c['effort'], 'selected_id': selected['id'],
        'baseline_validation': baseline['metrics'], 'selected_validation': selected['metrics'],
        'baseline_test': tests[0]['metrics'], 'selected_test': tests[1]['metrics'],
        'allocated_gpu_seconds': controller.spent(), 'controller_wall_seconds': time.monotonic() - started,
        'limitations': ['single seed and previously used test split',
            'NextItNet task adapter to public hypothesis harness, not DiscoveryRSI-27B or HCO training',
            '44 candidate attempts; successful fits, tokens and actual GPU time are not matched',
            'historical methods searched with evaluator v1; v2 checkpoint reevaluation is not a fresh search',
            'earlier MLE-STAR/autoresearch reasoning effort was not explicitly pinned',
            'native scientific-quality retention is distinct from final validation checkpoint selection',
            'trusted process mode; no claim of complete filesystem isolation']}
    save(root / 'summary.json', summary)
    comparison = read(WORKSPACE / 'artifacts/evaluator-fix-v2/reevaluation/summary.json')['methods']
    comparison.append({'method': 'DiscoveryRSI', 'selected_id': selected['id'],
        'validation': selected['metrics'], 'test': tests[1]['metrics']})
    save(root / 'comparison.json', comparison)
    harness.snapshot('complete')
    controller.event('complete', summary=summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        run(root)
    except Exception as exc:
        save(root / 'failure.json', {'type': type(exc).__name__, 'error': str(exc), 'time': time.time()})
        raise


if __name__ == '__main__':
    main()
