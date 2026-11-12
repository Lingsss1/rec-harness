import copy
import csv
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from rec_harness.core import ContractError, canonical, digest, write_json
from rec_harness.kuairand_agent import validate_plan, validate_review
from rec_harness.kuairand_graph import append, interaction, pareto, view
from rec_harness.store import Store
from rec_harness.runtime import node, edge
from rec_harness.graph_llm_audit import expected

BENCH = Path(__file__).resolve().parents[1]/'artifacts/kuairand-a37797a/benchmarks/kuairand-longseq'
sys.path.insert(0, str(BENCH))


class KuaiRandResearchTests(unittest.TestCase):
    def test_llm_controls_plan_length_and_review_points_without_fixed_tail(self):
        for n in (1, 3, 13):
            plan = {'requested_rounds': n, 'review_after': [n], 'evidence_refs': ['observed'],
                'steps': [{'slot': i, 'base_model_id': 'base', 'control_design': 'paired',
                           'variants': [{'label': 'control'}, {'label': 'intervention'}]} for i in range(1, n+1)]}
            validate_plan(plan, {'base'}, {'observed'})
            wrong = copy.deepcopy(plan); wrong['requested_rounds'] += 1
            with self.assertRaises(ContractError): validate_plan(wrong, {'base'}, {'observed'})
        plan['steps'][0]['control_design'] = 'factorial'
        with self.assertRaises(ContractError): validate_plan(plan, {'base'}, {'observed'})
        plan['steps'][0]['variants'] = [{'label': x} for x in ['none', 'only_a', 'only_b', 'both']]
        validate_plan(plan, {'base'}, {'observed'})

    def test_diagnostics_have_no_two_trial_quota_and_require_exact_parent(self):
        diagnostic = {'parent_model_id': 'base', 'paired_with': 'base', 'if_supported': 'continue', 'if_refuted': 'switch'}
        review = {'evidence_refs': ['base'], 'facts': [], 'interpretations': [], 'diagnostics': [diagnostic]*5}
        validate_review(review, [{'id': 'base'}])
        broken = copy.deepcopy(review); broken['diagnostics'][0]['paired_with'] = 'other'
        with self.assertRaises(ContractError): validate_review(broken, [{'id': 'base'}])

    def test_graph_counterevidence_revision_and_reference_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(tmp)
            try:
                facts = [node('negative', 'Experiment', {'counterevidence': True}, 'k', 'p'),
                    node('positive', 'Experiment', {}, 'k', 'p'),
                    node('old', 'Claim', {'status': 'provisional', 'scope': 'seed42', 'evidence_refs': ['positive']}, 'k', 'p', 'agent_hypothesis'),
                    node('revision', 'Claim', {'status': 'provisional', 'scope': 'seed42', 'evidence_refs': ['negative']}, 'k', 'p', 'agent_hypothesis'),
                    node('forbidden-final', 'Experiment', {'partition': 'test'}, 'k', 'p'),
                    node('bad-explanation', 'GapHypothesis', {'evidence_refs': ['forbidden-final']}, 'k', 'p', 'agent_hypothesis')]
                edges = [edge('counter', 'negative', 'old', 'CONTRADICTS', ['negative'], 'k', 1),
                         edge('revises', 'revision', 'old', 'SUPERSEDES', ['negative'], 'k', 1)]
                append(store, facts, edges)
                result = view(store, 'k', 'p', limit=8)
                self.assertNotIn('forbidden-final', canonical(result)); self.assertNotIn('bad-explanation', canonical(result))
                self.assertEqual(expected(result)['counterevidence_edge_ids'], ['counter'])
                self.assertEqual(expected(result)['revision_edge_ids'], ['revises'])
                for limit in (1, 2, 3, 4): expected(view(store, 'k', 'p', limit=limit))
                before = store.snapshot()
                with self.assertRaises(ContractError):
                    append(store, [node('old', 'Claim', {'status': 'supported'}, 'k', 'p', 'agent_hypothesis')])
                self.assertEqual(before, store.snapshot())
            finally: store.close()

    def test_pareto_retains_tradeoffs_and_does_not_erase_failures(self):
        def row(key, quality, cost, scoring):
            return {'id': key, 'valid': True, 'result': {'kind': 'kuairand_research_validation',
                'metrics': {'ndcg@10': quality}, 'training_gpu_seconds': cost, 'scoring_seconds': scoring}}
        rows = [row('quality', .5, 100, 10), row('fast', .4, 20, 2), row('dominated', .3, 50, 5),
                {'id': 'oom', 'valid': False, 'result': {'kind': 'execution_failure'}}]
        self.assertEqual(pareto(rows), ['fast', 'quality']); self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(interaction(.1, .2, .3, .5), .1)

    def test_no_graph_context_keeps_chronological_method_d_evidence_only(self):
        from rec_harness.adapters.kuairand_research import KuaiRandResearch
        c = object.__new__(KuaiRandResearch); c.task = 'k'; c.protocol = 'p'; c.config = {'memory': 'none'}
        state = {'records': [{'id': 'm', 'valid': True, 'result': {'kind': 'kuairand_research_validation'},
            'stage': 1, 'slot': 1, 'label': 'control', 'private_checkpoint': 'DO_NOT_SEND'}],
            'plans': [{'stage': 1}], 'reviews': [{'facts': ['past observation']}], 'frontier': ['m'], 'sources': []}
        packet = c.context(state)
        before = digest(packet); state['sources'].append({'id': 'future-source'})
        self.assertEqual(before, digest(packet))  # A later acquisition cannot change the pre-decision snapshot.
        self.assertNotIn('evidence_graph', packet); self.assertNotIn('DO_NOT_SEND', canonical(packet))
        self.assertEqual(packet['method'], 'D'); self.assertEqual(packet['reviews'], state['reviews'])


class KuaiRandSplitTests(unittest.TestCase):
    def test_research_cache_keeps_extra_training_position_and_two_holdouts(self):
        try: import torch
        except ImportError: self.skipTest('requires shared ML environment')
        from rec_harness.kuairand_data import ResearchDataset, make_research_cache
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'sequence.csv'
            with source.open('w') as f:
                writer = csv.writer(f); writer.writerow(['user_id', 'sequence_item_ids', 'sequence_timestamps'])
                writer.writerow([0, ','.join(map(str, range(10))), ','.join(str(1000*i) for i in range(10))])
            meta = make_research_cache(source, root/'cache', 4)
            self.assertEqual(meta['retained_events'], 7)
            train = ResearchDataset(root/'cache', 'train')[0]
            validation = ResearchDataset(root/'cache', 'validation')[0]
            final = ResearchDataset(root/'cache', 'final')[0]
            self.assertEqual(train['ids'].tolist(), [4, 5, 6, 7])
            self.assertEqual(train['labels'].tolist(), [5, 6, 7, 8])
            self.assertEqual(validation['target'].item(), 9); self.assertEqual(final['target'].item(), 10)
            self.assertNotIn(9, train['labels'].tolist()); self.assertNotIn(10, validation['ids'].tolist())
            self.assertEqual(validation['ids'].tolist(), [5, 6, 7, 8])
            self.assertEqual(final['ids'].tolist(), [6, 7, 8, 9])

    def test_model_only_contract_blocks_host_and_file_capabilities(self):
        from rec_harness.kuairand_source_contract import review_source
        from rec_harness.adapters.kuairand_research import BASE_SOURCE
        review_source(BASE_SOURCE)
        for suffix in ('\nimport os\n', '\nopen("private")\n', '\ntorch.load("checkpoint")\n', '\ntorch.fake = 1\n'):
            with self.assertRaises(ContractError): review_source(BASE_SOURCE+suffix)


class KuaiRandControllerTests(unittest.TestCase):
    def test_diagnostics_resume_and_two_slot_training_without_graph_feedback(self):
        try: import torch
        except ImportError: self.skipTest('requires shared ML environment')
        from rec_harness.autonomous import configuration
        from rec_harness.adapters.kuairand_research import KuaiRandResearch, BASE_SOURCE, TREATMENT
        from rec_harness.core import read_json
        workspace = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cfg = configuration(workspace/'examples/kuairand/no-graph.json')
            cfg.update(baseline_output=str(root/'baseline'), data_root=str(root/'data'),
                       graph_audit=str(root/'audit.json'), automatic_frozen_final=False)
            write_json(root/'baseline/controller.json', {'status': 'completed'})
            write_json(root/'audit.json', {'passed': True})
            campaign = root/'campaign'; KuaiRandResearch.init(campaign, cfg)
            write_json(campaign/'cache-prepared.json', {'cache': str(root/'cache')})
            c = KuaiRandResearch(campaign)
            self.addCleanup(c.store.close)
            state = c.state()
            def result(name):
                folder = c.model_folder(name)
                from rec_harness.core import file_hash
                return {'kind': 'kuairand_research_validation', 'epochs': 101, 'seed': 42,
                    'metrics': {'ndcg@10': .2}, 'training_gpu_seconds': 2., 'scoring_seconds': 1.,
                    'candidate_hash': file_hash(folder/'model.py'), 'treatment_hash': file_hash(folder/'training.json'),
                    'config_hash': digest(c.config['protocol']), 'qualification': 'SYNTHETIC UNIT FIXTURE'}
            for name in ('start-hstu', 'start-ultra_hstu', 'start-dare', 'start-ccformer'):
                folder = c.model_folder(name); folder.mkdir(parents=True)
                (folder/'model.py').write_text(BASE_SOURCE); write_json(folder/'training.json', TREATMENT)
                c.record(state, name, result(name), 0, 0, 'start', name[len('start-'):])
            calls, jobs = [], []
            def agent(role, context, directory):
                calls.append(role)
                if role in ('plan', 'review'): self.assertNotIn('evidence_graph', context)
                if role == 'plan':
                    return {'requested_rounds': 3, 'review_after': [1, 3], 'evidence_refs': ['start-hstu'],
                        'competing_explanations': ['capacity', 'mechanism'], 'source_requests': [],
                        'steps': [{'slot': i, 'base_model_id': 'start-hstu', 'kind': 'optimization',
                            'instruction': 'paired', 'condition': 'prospective', 'expected_observation': 'bounded',
                            'control_design': 'paired', 'variants': [{'label': 'control', 'instruction': 'keep'},
                            {'label': 'candidate', 'instruction': 'change'}]} for i in range(1, 4)]}
                if role == 'implement':
                    return {'changed_files': ['model.py'], 'checks': [], 'limitations': ['synthetic unit fixture']}
                if role == 'source_review':
                    return {'verdict': 'approve', 'implementation_hash': context['implementation_hash'],
                            'checks': [], 'blocking_findings': [], 'limitations': []}
                if role == 'review':
                    diagnostics = []
                    if context['review_slot'] == 1:
                        diagnostics = [{'parent_model_id': 'start-hstu', 'paired_with': 'start-hstu', 'kind': 'repeat',
                            'instruction': 'exact repeat', 'uncertainty': 'decision-changing',
                            'if_supported': 'continue', 'if_refuted': 'switch'} for _ in range(3)]
                    else: (campaign/'STOP').touch()
                    return {'facts': [], 'interpretations': [], 'uncertainties': [], 'decision': 'continue',
                        'rationale': 'fixture', 'next_question': 'next', 'diagnostics': diagnostics,
                        'evidence_refs': ['start-hstu']}
                raise AssertionError(role)
            c.agent = agent
            def job(model_id, model, gpus, cache): jobs.append((model_id, gpus)); return result(model_id)
            with patch.object(c, 'starts'), patch.object(c, 'sources'), patch.object(c, 'run_job', side_effect=job):
                c.run()
                first_ids = {r['id'] for r in c.state()['records']}
                self.assertEqual(len(jobs), 8)  # 2 main variants + 3 independent diagnostic pairs.
                self.assertEqual(calls.count('plan'), 1)
                (campaign/'STOP').unlink()
                c.run()
            self.assertEqual(calls.count('plan'), 1)  # same unfinished prospective plan resumes.
            self.assertEqual(len(jobs), 12)  # only rounds 2 and 3 added; completed diagnostics not repeated.
            self.assertTrue(first_ids <= {r['id'] for r in c.state()['records']})
            self.assertTrue(all(g in ([4, 5], [6, 7]) for _, g in jobs))
            packet = c.context(c.state())
            self.assertNotIn('host-only-final', canonical(packet)); self.assertNotIn('evidence_graph', packet)
            # Host-only final output must have no route into later role packets or graph facts.
            from rec_harness.kuairand_control import freeze_final
            for model_id in c.state()['frontier']:
                folder = c.model_folder(model_id)/'training'; folder.mkdir(exist_ok=True)
                (folder/'last.pt').write_bytes(b'SYNTHETIC CHECKPOINT FIXTURE')
                write_json(folder/'result.json', result(model_id))
            freeze_final(c, c.state())
            for model_id in c.state()['frontier']:
                write_json(c.root/'host-only-final'/model_id/'result.json', {'test_score': 'PRIVATE_FINAL_SENTINEL'})
            self.assertNotIn('PRIVATE_FINAL_SENTINEL', canonical(c.context(c.state())))
            self.assertNotIn('PRIVATE_FINAL_SENTINEL', canonical(c.store.snapshot()))


if __name__ == '__main__': unittest.main()
