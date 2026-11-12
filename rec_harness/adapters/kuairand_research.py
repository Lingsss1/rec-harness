"""Full method D with optional graph memory and LLM-selected study lengths.

The memory-disabled arm keeps chronological scientific receipts. It never gets
graph nodes, relationships, retrieved summaries, or another arm's outcomes.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import shutil
import subprocess
import time

from rec_harness.core import Artifacts, ContractError, canonical, digest, file_hash, identifier, read_json, write_json
from rec_harness.store import Store
from rec_harness.kuairand_baseline import MODELS, command, environment, lock
from rec_harness.kuairand_agent import call, validate_plan, validate_review
from rec_harness.kuairand_graph import append, view, observation, pareto, interaction
from rec_harness.runtime import node, edge

BASE_SOURCE = '''"""Pinned model, independently initialized by the host for each trial."""
from longseq_bench.models import ModelConfig, HSTUEncoder, DAREEncoder
from longseq_bench.ccformer import CCFormerEncoder

def build(config):
    cfg = ModelConfig(**config)
    cls = DAREEncoder if cfg.name == 'dare' else CCFormerEncoder if cfg.name == 'ccformer' else HSTUEncoder
    return cls(cfg)
'''
TREATMENT = {'optimizer': 'adamw', 'learning_rate': .001, 'weight_decay': 0.,
             'scheduler': 'constant', 'loss': 'sampled_softmax', 'temperature': .05}


class KuaiRandResearch:
    @staticmethod
    def init(root, config):
        root = Path(root).resolve(); root.mkdir(parents=True, exist_ok=True)
        if (root/'kuairand-auto.json').exists(): raise ContractError('campaign already frozen')
        if config['adapter'] != 'kuairand-research-v1': raise ContractError('unknown adapter')
        if config['memory'] not in {'graph', 'none'}: raise ContractError('memory must be graph or none')
        if config['gpus'] != ([0, 1, 2, 3] if config['memory'] == 'graph' else [4, 5, 6, 7]):
            raise ContractError('incorrect fixed GPU allocation')
        if config.get('method') != 'D' or config.get('max_rounds') is not None:
            raise ContractError('full method D requires LLM-selected lengths and unlimited global rounds')
        if config['codex'].get('model') != 'gpt-6.1-sol' or config['codex'].get('effort') != 'high':
            raise ContractError('frozen Codex model/effort mismatch')
        benchmark = Path(config['benchmark'])
        protocol = read_json(benchmark/'configs/benchmark.json')
        # The runtime version is shared between arms, independently of official packages.
        runtime = subprocess.check_output([config['python'], '-c',
            'import json,torch,numpy,pandas,platform; print(json.dumps(dict(torch=torch.__version__,numpy=numpy.__version__,pandas=pandas.__version__,python=platform.python_version())))'], text=True)
        config = {**config, 'runtime': __import__('json').loads(runtime), 'protocol': protocol,
            'protocol_hash': digest({'contract': protocol, 'split': 'leave-two-out', 'cache_events': 1027,
                'runtime': __import__('json').loads(runtime), 'scoring_panel': 'first-128-users-1-warmup-3-median'})}
        package = Path(__file__).resolve().parents[1]
        config['controller_files'] = {str(path.relative_to(package)): file_hash(path) for path in
            [*package.glob('kuairand*.py'), Path(__file__), package/'store.py', package/'core.py']}
        write_json(root/'kuairand-auto.json', config)
        write_json(root/'state.json', {'status': 'waiting_baselines', 'stage': 0,
            'records': [], 'plans': [], 'reviews': [], 'completed_slots': [], 'frontier': []})
        store = Store(root/'memory'); store.event('kuairand_created', {'configuration_hash': digest(config),
            'memory': config['memory'], 'method': 'D', 'max_rounds': None})
        store.close()

    def __init__(self, root, agent=None):
        self.root = Path(root).resolve(); self.config = read_json(self.root/'kuairand-auto.json')
        self.store = Store(self.root/'memory'); self.artifacts = Artifacts(self.root/'objects')
        self.workspace = Path(__file__).resolve().parents[2]
        self.benchmark = Path(self.config['benchmark']); self.python = self.config['python']
        self.task = 'kuairand-' + self.config['memory']; self.protocol = self.config['protocol_hash']
        self.agent = agent
        creation = self.store.events()[0]
        if creation['payload']['configuration_hash'] != digest(self.config):
            raise ContractError('campaign configuration differs from its creation event')
        package = Path(__file__).resolve().parents[1]
        if any(file_hash(package/name) != sha for name, sha in self.config['controller_files'].items()):
            raise ContractError('controller implementation changed; use a new explicitly versioned campaign')

    def state(self): return read_json(self.root/'state.json')
    def lock(self): return lock(self.root/'.controller.lock')
    def save(self, state): write_json(self.root/'state.json', state)
    def stopped(self): return (self.root/'STOP').exists()
    def status(self):
        state = self.state()
        return {'adapter': 'kuairand-research-v1', 'memory': self.config['memory'], 'method': 'D',
            'status': state['status'], 'stage': state['stage'], 'completed_models': len(state['records']),
            'frontier': state['frontier'], 'stop_requested': self.stopped(), 'integrity': self.store.verify()}
    def verify(self): return self.store.verify()
    def recover(self): return self.status()

    def model_folder(self, model_id):
        identifier(model_id); return self.root/'models'/model_id

    def context(self, state, query='', *, graph=True):
        # Exclude paths, checkpoints, original benchmark test metrics and host-only final outputs.
        result = {'task_id': self.task, 'protocol_hash': self.protocol,
            'protocol': {'history': 1024, 'dim': 128, 'global_batch': 128, 'negatives': 128,
                'precision': 'fp32', 'epochs': 101, 'seed': 42, 'partition': 'validation',
                'score': 'exhaustive-catalog-ascending-id-ties-seen-filter',
                'timestamps': 'same next-event timestamps as pinned baseline; frozen information boundary'},
            'method': 'D', 'max_rounds': None, 'round_policy': 'LLM requests study length and review checkpoints',
            'records': [{k: r[k] for k in ('id', 'valid', 'result', 'stage', 'slot', 'label')}
                for r in state['records'] if r in state['records'][-64:] or r['id'] in state['frontier'] or r['stage'] == 0],
            'plans': state['plans'][-16:], 'reviews': state['reviews'][-16:], 'frontier': state['frontier'],
            'history_total_records': len(state['records']),
            'sources': state.get('sources', []),
            'registered_training': {'optimizer': ['adamw', 'adam', 'sgd'], 'scheduler': ['constant', 'cosine'],
                'loss': ['sampled_softmax', 'bpr']},
            'instructions': 'Stage plans, reviews and diagnostics use full method D; no fixed experimental quota. '
                'Performance Pareto selection and single-seed mechanism interpretation are separate.'}
        if graph and self.config['memory'] == 'graph': result['evidence_graph'] = view(self.store, self.task, self.protocol, query)
        return copy.deepcopy(result)

    def sources(self, state, requests):
        from rec_harness.kuairand_literature import acquire
        for request in requests:
            result = acquire(request, self.root/'literature'/digest(request), self.artifacts)
            if result['id'] in {s['id'] for s in state.get('sources', [])}: continue
            state.setdefault('sources', []).append(result); self.save(state)
            if self.config['memory'] == 'graph':
                append(self.store, [node(result['id'], 'Source', {**result, 'partition': 'development'},
                    self.task, self.protocol, 'paper_claim')])
            self.store.event('kuairand_source_acquisition', result)

    def seed_components(self):
        if self.config['memory'] != 'graph': return
        existing = {n['id'] for n in self.store.snapshot()['nodes']}
        bindings = {
            'attention-gating': ('hstu', 'models.py', 'HSTUEncoder', 'SiLU attention, relative bias, U gate'),
            'sparse-window-truncation': ('ultra_hstu', 'models.py', 'HSTUEncoder', 'local/global mask and one causal post-truncation layer'),
            'decoupled-embedding': ('dare', 'models.py', 'DAREEncoder', 'separate attention and representation embeddings'),
            'target-conditioned-retrieval': ('dare', 'models.py', 'DAREEncoder', 'per-candidate Top-K and cross features'),
            'cross-field': ('ccformer', 'ccformer.py', 'CCFormerEncoder', 'user/history/target directed attention'),
            'historical-time': ('ccformer', 'ccformer.py', 'CCFormerEncoder', 'historical timestamp relative subspaces'),
            'hierarchical-compression': ('ccformer', 'ccformer.py', 'CCFormerEncoder', 'causal left-padded strided compression'),
        }
        for key, (model, file, symbol, behavior) in bindings.items():
            identity = 'component-'+key
            if identity in existing: continue
            source = self.benchmark/'longseq_bench'/file
            ref = self.artifacts.put(source)
            append(self.store, [node(identity, 'Component', {'name': key, 'model': model,
                'input': 'history item IDs, allowed timestamps and candidate IDs for conditional scoring',
                'output': 'causal prefix context or per-candidate scalar scores',
                'implementation': {'file': file, 'symbol': symbol, 'source_ref': ref},
                'behavior': behavior, 'train_behavior': 'all supervised causal positions',
                'inference_behavior': 'exhaustive catalog candidates', 'dependencies': [model],
                'combination_constraints': 'preserve all-position causality, candidate independence and frozen information',
                'qualification': 'Local implementation/adaptation; paper fidelity is a separate question',
                'partition': 'development'}, self.task, self.protocol)])

    def invoke(self, role, context, directory):
        directory = Path(directory)
        if (directory/'accepted.json').exists() and (directory/'context.json').exists():
            # A decision is resumed from its original visible snapshot, even if its graph commit already happened.
            frozen_context = read_json(directory/'context.json')
            context.clear(); context.update(frozen_context)
        elif role in {'plan', 'review', 'implement', 'source_review'}:
            state = self.state()
            history = {'records': [{k: r[k] for k in ('id', 'valid', 'result', 'stage', 'slot', 'label')}
                for r in state['records']], 'plans': state['plans'], 'reviews': state['reviews'], 'sources': state.get('sources', [])}
            write_json(directory/'evidence/chronological-history.json', history)
            context['evidence_archive'] = {'path': 'evidence/chronological-history.json',
                'sha256': file_hash(directory/'evidence/chronological-history.json'),
                'instructions': 'Complete chronological scientific history is available here; header shows a bounded recent/frontier view. '
                    'Read older records as needed. No graph-derived summary is in this archive.'}
        response = self.agent(role, context, directory) if self.agent else call(role, context, directory, self.config['codex'])
        files = [str(p.relative_to(directory)) for p in Path(directory).rglob('*') if p.is_file() and '__pycache__' not in str(p)]
        archive = self.artifacts.snapshot(directory, files)
        self.store.event('kuairand_role', {'role': role, 'context_hash': digest(context),
            'response_hash': digest(response), 'archive': archive})
        return response

    def ensure_cache(self):
        from rec_harness.kuairand_data import make_research_cache
        import sys
        sys.path.insert(0, str(self.benchmark))
        from longseq_bench.data import verify_cache, sha256
        data = Path(self.config['data_root']); cache = data/'research-cache-l1024'
        with lock(data/'.research-cache.lock'):
            if cache.exists():
                meta = verify_cache(cache)
                if meta.get('protocol') != 'kuairand-research-v1' or meta['max_history'] != 1024:
                    raise ContractError('research cache identity mismatch')
                if meta['source_csv_sha256'] != sha256(data/'processed/sasrec_format.csv'):
                    raise ContractError('research source changed')
                return cache
            stage = data/'research-cache-l1024.preparing'
            if stage.exists(): stage.rename(data/('research-cache-interrupted-'+str(time.time_ns())))
            source = data/'processed/sasrec_format.csv'
            make_research_cache(source, stage, 1024, read_json(source.with_suffix('.provenance.json')))
            stage.rename(cache)
        return cache

    def run_job(self, model_id, model, gpus, cache):
        folder = self.model_folder(model_id); output = folder/'training'
        env = environment(self.benchmark, self.workspace, gpus)
        common = ['--benchmark', str(self.benchmark), '--model', model, '--candidate', str(folder/'model.py'),
            '--config', str(self.root/'benchmark.json'), '--treatment', str(folder/'training.json')]
        if not (folder/'admission.json').exists():
            command([self.python, '-m', 'rec_harness.kuairand_worker', 'check', *common,
                '--output', str(folder/'admission.json')], folder/'admission-jobs', cwd=self.workspace, env=env, gpus=gpus[:1])
        if (output/'result.json').exists():
            result = read_json(output/'result.json')
            if (result.get('candidate_hash') != file_hash(folder/'model.py') or
                result.get('treatment_hash') != file_hash(folder/'training.json') or
                result.get('config_hash') != digest(self.config['protocol'])):
                raise ContractError('completed candidate, treatment or protocol changed')
            return result
        command([self.python, '-m', 'torch.distributed.run', '--standalone', '--nproc-per-node=2', '-m',
            'rec_harness.kuairand_worker', 'train', *common, '--cache', str(cache), '--output', str(output)],
            folder/'training-jobs', cwd=self.workspace, env=env, gpus=gpus)
        return read_json(output/'result.json')

    def record(self, state, model_id, result, stage, slot, label, model, parent=None):
        if any(r['id'] == model_id for r in state['records']): return
        folder = self.model_folder(model_id)
        archive = self.artifacts.snapshot(folder, ['model.py', 'training.json'] +
            (['training/last.pt'] if (folder/'training/last.pt').exists() else []) +
            [str(p.relative_to(folder)) for p in folder.rglob('*.json') if p.name not in {'request.json', 'context.json'}])
        valid = result.get('kind') == 'kuairand_research_validation' and result.get('epochs') == 101
        row = {'id': model_id, 'valid': valid, 'result': result, 'stage': stage, 'slot': slot, 'label': label, 'model': model}
        existing = {n['id']: n for n in self.store.snapshot()['nodes']}
        if model_id not in existing:
            observation(self.store, model_id, result, self.task, self.protocol, archive)
        elif existing[model_id]['attrs']['result'] != result:
            raise ContractError('resumed result differs from immutable graph fact')
        state['records'].append(row); state['frontier'] = pareto(state['records'])
        if self.config['memory'] == 'graph':
            version_id = model_id+'-version'
            if version_id not in existing:
                nodes = [node(version_id, 'ModelVersion', {'model': model, 'source_ref': archive['model.py'],
                    'training_ref': archive['training.json'], 'evidence_refs': [model_id], 'partition': 'development'}, self.task, self.protocol)]
                edges = [edge(model_id+'-evaluates', model_id, version_id, 'EVALUATES', [model_id], self.task, self.store.version+1)]
                for component in self.store.snapshot()['nodes']:
                    if parent is None and component['type'] == 'Component' and component['attrs'].get('model') == model:
                        edges.append(edge(version_id+'-'+component['id'], version_id, component['id'], 'HAS_COMPONENT',
                            [model_id], self.task, self.store.version+1))
                if parent is not None:
                    edges.append(edge(version_id+'-parent', version_id, parent+'-version', 'DERIVED_FROM',
                        [model_id], self.task, self.store.version+1))
                append(self.store, nodes, edges)
        self.save(state)
        self.store.event('kuairand_result', {'record': row, 'frontier': state['frontier']})

    def starts(self, state, cache):
        """Both arms train the same four independent research-protocol starts."""
        def slot(models, gpus):
            values = []
            for name in models:
                if self.stopped(): break
                model_id = 'start-' + name; folder = self.model_folder(model_id)
                folder.mkdir(parents=True, exist_ok=True)
                if not (folder/'model.py').exists():
                    (folder/'model.py').write_text(BASE_SOURCE); write_json(folder/'training.json', TREATMENT)
                result = self.run_job(model_id, name, gpus, cache)
                values.append((model_id, result, name))
            return values
        gpus = self.config['gpus']
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(slot, MODELS[::2], gpus[:2]), pool.submit(slot, MODELS[1::2], gpus[2:])]
            for future in futures:
                for model_id, result, name in future.result(): self.record(state, model_id, result, 0, 0, 'start', name)

    def plan(self, state):
        index = state['stage'] + 1
        directory = self.root/'stages'/str(index)/'plan'
        ctx = self.context(state, state['reviews'][-1]['next_question'] if state['reviews'] else 'four model mechanisms')
        plan = self.invoke('plan', ctx, directory)
        visible = {r['id'] for r in state['records']}
        visible.update(s['id'] for s in state.get('sources', []))
        if 'evidence_graph' in ctx: visible.update(n['id'] for n in ctx['evidence_graph']['nodes'])
        validate_plan(plan, {r['id'] for r in state['records'] if r['valid']}, visible)
        self.sources(state, plan['source_requests'])
        frozen = {'stage': index, 'plan': plan, 'visible_context_hash': digest(ctx),
            'graph_snapshot_id': ctx.get('evidence_graph', {}).get('graph_snapshot_id'), 'registered_at': time.time()}
        write_json(directory.parent/'frozen-plan.json', frozen)
        state['plans'].append(frozen); state['stage'] = index; self.save(state)
        if self.config['memory'] == 'graph':
            plan_id = 'plan-%d' % index
            nodes = [node(plan_id, 'Proposal', {'plan': plan,
                'evidence_refs': plan['evidence_refs'], 'partition': 'development'}, self.task, self.protocol, 'agent_hypothesis')]
            edges = []
            for i, explanation in enumerate(plan['competing_explanations']):
                key = plan_id+'-hypothesis-%d' % i
                nodes.append(node(key, 'GapHypothesis', {'statement': explanation, 'status': 'proposed',
                    'evidence_refs': plan['evidence_refs'], 'partition': 'development'}, self.task, self.protocol, 'agent_hypothesis'))
                edges.append(edge(key+'-plan', plan_id, key, 'TESTS', plan['evidence_refs'], self.task, self.store.version+1))
                if i:
                    edges.append(edge(key+'-alternative', key, plan_id+'-hypothesis-0', 'ALTERNATIVE_TO',
                        plan['evidence_refs'], self.task, self.store.version+1))
            append(self.store, nodes, edges)
        self.store.event('kuairand_plan_registered', frozen)
        return plan

    def trial(self, state, step, cache, *, diagnostic=None):
        stage, slot = state['stage'], step['slot']
        base_id = step['base_model_id']; base = next(r for r in state['records'] if r['id'] == base_id and r['valid'])
        variants = []
        for variant in step['variants']:
            identifier(variant['label'])
            model_id = 's%d-r%d-%s' % (stage, slot, variant['label'])
            folder = self.model_folder(model_id); folder.mkdir(parents=True, exist_ok=True)
            directory = self.root/'stages'/str(stage)/('round-%d' % slot)/variant['label']/'implement'
            source = self.model_folder(base_id)
            if not (directory/'accepted.json').exists():
                model_dir = directory/'model'; model_dir.mkdir(parents=True, exist_ok=True)
                for name in ('model.py', 'training.json'):
                    if not (model_dir/name).exists(): shutil.copyfile(source/name, model_dir/name)
            ctx = {'base_model_id': base_id, 'plan': state['plans'][-1]['plan'], 'step': step, 'variant': variant,
                'diagnostic': diagnostic, 'fixed_protocol': self.context(state, graph=False)['protocol'],
                'prior_observations': self.context(state, graph=False)['records'],
                'registered_training': self.context(state, graph=False)['registered_training'],
                'reference_modules': {p.name: p.read_text() for p in (self.benchmark/'longseq_bench').glob('*.py')
                    if p.name in {'models.py', 'ccformer.py'}},
                'allowed_files': ['model/model.py', 'model/training.json']}
            exact_repeat = diagnostic is not None and (variant['label'] == 'control' or diagnostic['kind'] == 'repeat')
            if exact_repeat:
                for name in ('model.py', 'training.json'): shutil.copyfile(source/name, folder/name)
            else:
                response = self.invoke('implement', ctx, directory)
                if not set(response['changed_files']) <= {'model.py', 'training.json', 'model/model.py', 'model/training.json'}:
                    raise ContractError('candidate changed a protected file')
                for name in ('model.py', 'training.json'):
                    shutil.copyfile(directory/'model'/name, folder/name)
            from rec_harness.kuairand_worker import validate_treatment
            validate_treatment(read_json(folder/'training.json'))
            variants.append({'id': model_id, 'label': variant['label'],
                'source': (folder/'model.py').read_text(), 'training': read_json(folder/'training.json')})
        implementation_hash = digest(variants)
        review_dir = self.root/'stages'/str(stage)/('round-%d' % slot)/'source-review'
        review = self.invoke('source_review', {'plan': state['plans'][-1]['plan'], 'step': step,
            'prior_observations': self.context(state, graph=False)['records'],
            'base': {'id': base_id, 'source': (self.model_folder(base_id)/'model.py').read_text(),
                'training': read_json(self.model_folder(base_id)/'training.json')},
            'variants': variants, 'implementation_hash': implementation_hash}, review_dir)
        if review['implementation_hash'] != implementation_hash:
            raise ContractError('independent source review refers to a different implementation')
        if review['verdict'] != 'approve' or review['blocking_findings']:
            # An implementation rejection is evidence about this attempt, not a reason to end unattended research.
            for v in variants:
                self.record(state, v['id'], {'kind': 'source_review_rejection',
                    'review_hash': digest(review), 'blocking_findings': review['blocking_findings'],
                    'mechanism_status': 'unknown'}, stage, slot, v['label'], base['model'], base_id)
            state['completed_slots'].append([stage, slot]); self.save(state)
            self.store.event('kuairand_rejected_implementation', {'stage': stage, 'slot': slot,
                'review_hash': digest(review), 'candidate_training_started': False})
            return
        gpus = self.config['gpus']
        def run_variant(v, group):
            try: return v, self.run_job(v['id'], base['model'], group, cache)
            except ContractError as error:
                return v, {'kind': 'execution_failure', 'reason': str(error), 'mechanism_status': 'unknown'}
        # At most two training tasks; all variants receive seed=42 and sampler seed=0.
        with ThreadPoolExecutor(2) as pool:
            for left in range(0, len(variants), 2):
                if self.stopped(): return
                fs = [pool.submit(run_variant, v, gpus[j*2:j*2+2]) for j, v in enumerate(variants[left:left+2])]
                for f in fs:
                    v, result = f.result(); self.record(state, v['id'], result, stage, slot, v['label'], base['model'], base_id)
        rows = [r for r in state['records'] if r['stage'] == stage and r['slot'] == slot]
        if step['control_design'] == 'factorial' and len(rows) == 4 and all(r['valid'] for r in rows):
            values = {r['label']: r['result']['metrics']['ndcg@10'] for r in rows}
            self.store.event('kuairand_interaction', {'stage': stage, 'slot': slot, 'metric': 'validation_ndcg@10',
                'delta': interaction(values['none'], values['only_a'], values['only_b'], values['both']),
                'evidence_refs': [r['id'] for r in rows], 'status': 'single_seed_provisional'})
        state['completed_slots'].append([stage, slot]); self.save(state)

    def review(self, state, slot):
        key = '%d-%d' % (state['stage'], slot)
        if any(r.get('review_key') == key for r in state['reviews']): return
        ctx = self.context(state, 'counterevidence competing explanations')
        ctx['review_slot'] = slot
        review = self.invoke('review', ctx, self.root/'stages'/str(state['stage'])/('review-%d' % slot))
        validate_review(review, state['records'])
        n = len(state['plans'][-1]['plan']['competing_explanations'])
        if any(not 0 <= entry['hypothesis_index'] < n for entry in review['interpretations']):
            raise ContractError('review names unknown competing explanation')
        review = {**review, 'review_key': key}
        # Graph interpretation is appended in a new identity; old explanations survive.
        if self.config['memory'] == 'graph':
            for i, explanation in enumerate(review['interpretations']):
                claim_id = 'explanation-%s-%d' % (key, i)
                nodes = [node(claim_id, 'Claim',
                    {'status': 'provisional', 'scope': 'seed 42 research validation only',
                     'statement': explanation['statement'], 'evidence_refs': explanation['evidence_refs'],
                     'hypothesis_index': explanation['hypothesis_index'],
                     'partition': 'development', 'mechanism_status': 'single_seed_provisional'},
                    self.task, self.protocol, 'agent_hypothesis')]
                relation = {'supports': 'SUPPORTS', 'contradicts': 'CONTRADICTS'}.get(explanation['assessment'])
                edges = []
                if relation:
                    for ref in explanation['evidence_refs']:
                        edges.append(edge(claim_id+'-'+ref, ref, claim_id, relation, [ref], self.task, self.store.version+1))
                if explanation['supersedes_review_key']:
                    previous = [n for n in self.store.snapshot()['nodes'] if n['type'] == 'Claim'
                        and n['id'].startswith('explanation-'+explanation['supersedes_review_key']+'-')
                        and n['attrs'].get('hypothesis_index') == explanation['hypothesis_index']]
                    if not previous: raise ContractError('interpretation revision names no prior matching explanation')
                    for old in previous:
                        edges.append(edge(claim_id+'-revises-'+old['id'], claim_id, old['id'], 'SUPERSEDES',
                            explanation['evidence_refs'], self.task, self.store.version+1))
                append(self.store, nodes, edges)
        state['reviews'].append(review); self.save(state)
        self.store.event('kuairand_direction_review', review)
        if review['decision'] == 'blocked': raise ContractError('LLM review found an unresolved blocking condition')

    def diagnostics(self, state, review, cache):
        for i, diagnostic in enumerate(review['diagnostics']):
            # Unique integer slots outside the prospective optimization slots.
            # Explicit plan length controls this offset; it is not a tail quota.
            slot = state['plans'][-1]['plan']['requested_rounds'] + 1 + i + 100000*int(review['review_key'].split('-')[-1])
            if [state['stage'], slot] in state['completed_slots']: continue
            if self.stopped(): return
            step = {'slot': slot, 'base_model_id': diagnostic['parent_model_id'], 'kind': 'diagnostic',
                'instruction': diagnostic['instruction'], 'condition': diagnostic['uncertainty'],
                'expected_observation': diagnostic['if_supported'], 'control_design': 'paired',
                'variants': [{'label': 'control', 'instruction': 'Exact independent from-scratch repeat of the parent; preserve model.py and training.json.'},
                    {'label': 'intervention', 'instruction': diagnostic['instruction']}]}
            self.trial(state, step, cache, diagnostic=diagnostic)

    def run(self):
        with lock(self.root/'.controller.lock'):
            state = self.state()
            if state.get('status') == 'blocked': raise ContractError('blocked campaign requires repair command with reason')
            write_json(self.root/'benchmark.json', self.config['protocol'])
            try:
                baseline = Path(self.config['baseline_output'])/'controller.json'
                while not self.stopped():
                    if baseline.exists():
                        baseline_state = read_json(baseline)
                        if baseline_state['status'] == 'completed': break
                        if baseline_state['status'] == 'blocked':
                            raise ContractError('shared original four-model baseline is blocked')
                    state['status'] = 'waiting_baselines'; self.save(state); time.sleep(10)
                if self.stopped(): return self.status()
                audit = read_json(self.config['graph_audit'])
                if audit.get('passed') is not True: raise ContractError('live graph readback acceptance required before either research arm')
                # Cache preparation uses the same ML interpreter as training.
                prepared = self.root/'cache-prepared.json'
                if not prepared.exists():
                    command([self.python, '-m', 'rec_harness.kuairand_control', 'prepare-cache', str(self.root)],
                        self.root/'jobs/cache', cwd=self.workspace,
                        env=environment(self.benchmark, self.workspace, []))
                cache = Path(read_json(prepared)['cache'])
                from rec_harness.kuairand_literature import SEEDS
                self.sources(state, SEEDS); self.seed_components()
                state['status'] = 'research-starts'; self.save(state); self.starts(state, cache)
                state['status'] = 'searching'; self.save(state)
                while not self.stopped():
                    plan = self.plan(state) if not state['plans'] or state['stage'] in state.get('closed_stages', []) or all(
                        [state['stage'], s['slot']] in state['completed_slots'] for s in state['plans'][-1]['plan']['steps']) else state['plans'][-1]['plan']
                    for step in plan['steps']:
                        if self.stopped(): break
                        if [state['stage'], step['slot']] not in state['completed_slots']: self.trial(state, step, cache)
                        if step['slot'] in plan['review_after']:
                            self.review(state, step['slot'])
                            matching = next(r for r in state['reviews'] if r['review_key'] == '%d-%d' % (state['stage'], step['slot']))
                            diagnostic_review = 0
                            while matching['diagnostics'] and not self.stopped():
                                self.diagnostics(state, matching, cache)
                                # The LLM can request more diagnostics after seeing outcomes.
                                diagnostic_review += 1
                                post_slot = 10000000 + step['slot'] + 100000000*diagnostic_review
                                self.review(state, post_slot)
                                matching = next(r for r in state['reviews'] if r['review_key'] == '%d-%d' % (state['stage'], post_slot))
                            if matching['decision'] in {'switch', 'revisit'}:
                                state.setdefault('closed_stages', []).append(state['stage']); self.save(state)
                                self.store.event('kuairand_direction_closed', {'stage': state['stage'],
                                    'completed_through_slot': step['slot'], 'decision': matching['decision'],
                                    'remaining_steps': 'cancelled prospectively, not fabricated as experiments'})
                                break
                    if self.config.get('automatic_frozen_final', True) and not self.stopped():
                        from rec_harness.kuairand_control import freeze_final
                        freeze_final(self, state)
                        from rec_harness.kuairand_control import final_run
                        for model_id in list(state['frontier']):
                            if self.stopped(): break
                            try: final_run(self.root, model_id, self.config['gpus'][:1])
                            except (ContractError, OSError) as error:
                                # Final failures stay in the host-only queue as well.
                                write_json(self.root/'host-only-final'/model_id/'attention.json',
                                    {'reason': str(error), 'requires_audit_before_retry': True})
                state['status'] = 'stopped'; self.save(state)
            except BaseException as error:
                state.update(status='blocked', blocking_reason=str(error)); self.save(state)
                self.store.event('kuairand_blocked', {'stage': state['stage'], 'reason': str(error)})
                raise
        return self.status()
