"""Visibility, replay and frozen-runtime contracts; no ML install required."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rec_harness.core import ContractError, digest, read_json, write_json
from rec_harness.harness_prefix import historical_bundle
from rec_harness.runtime import Research, node
from rec_harness.semantic import receipt, validate
from rec_harness.semantic_encoder import LocalEncoder, freeze, freeze_options, pin_options, verify_options
from rec_harness.store import Store

IDENTITY = {'model_hash': 'a' * 64, 'implementation_hash': 'b' * 64, 'settings': {'fixture': True}}


def scored(nodes, query):
    return receipt(nodes, query, IDENTITY,
                   {key: 0.9 if key == 'semantic' else 0.1 for key in nodes})


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_encoder_sees_only_scoped_revealed_nodes_and_replay_uses_receipt(self):
        store = Store(self.root / 'memory')
        self.addCleanup(store.close)
        nodes = [node('lexical', 'Observation', {'text': 'needle'}, 't', 'p'),
                 node('semantic', 'Observation', {'text': 'related paraphrase'}, 't', 'p'),
                 node('failure', 'Failure', {}, 't', 'p'),
                 node('future', 'Observation', {'evidence_dependency_ids': ['later']}, 't', 'p'),
                 node('other-task', 'Observation', {}, 'other', 'p'),
                 node('other-protocol', 'Observation', {}, 't', 'other')]
        store.patch({'base_version': 0, 'nodes': nodes, 'edges': []})
        seen = []
        def encoder(eligible, query):
            seen.append(set(eligible))
            return scored(eligible, query)
        bundle = store.bundle('t', 'p', 'evidence', query='needle', limit=2,
                              revealed=['lexical', 'semantic', 'failure'], semantic=encoder)
        self.assertEqual(seen, [{'lexical', 'semantic', 'failure'}])
        self.assertEqual([n['id'] for n in bundle['nodes']], ['failure', 'semantic'])
        self.assertEqual(bundle['retrieval']['selection_reasons']['semantic'], 'semantic_seed')
        replay = historical_bundle(store.snapshot(), bundle, 3)
        self.assertEqual(set(replay['retrieval']['retrieved_node_ids']), seen[0])
        self.assertEqual(replay['retrieval']['semantic'], bundle['retrieval']['semantic'])
        self.assertEqual(len(seen), 1)
        self.assertEqual(historical_bundle(store.snapshot(), bundle, 2), bundle)
        lexical = store.bundle('t', 'p', 'evidence', query='needle', limit=2,
                               revealed=['lexical', 'semantic', 'failure'])
        self.assertEqual([n['id'] for n in lexical['nodes']], ['failure', 'lexical'])
        self.assertEqual(lexical['retrieval']['method'], 'role-lexical-two-hop-v2')

    def test_receipt_rejects_changed_query_documents_scores_and_encoder(self):
        nodes = {'semantic': node('semantic', 'Observation', {}, 't', 'p')}
        original = scored(nodes, 'q')
        with self.assertRaisesRegex(ContractError, 'query'): validate(original, nodes, 'different')
        changed = copy.deepcopy(nodes); changed['semantic']['attrs']['text'] = 'edited'
        with self.assertRaisesRegex(ContractError, 'documents'): validate(original, changed, 'q')
        for value in [{}, {'semantic': True}, {'semantic': 2}, {'semantic': .5, 'extra': .1}]:
            with self.subTest(value=value), self.assertRaises(ContractError):
                receipt(nodes, 'q', IDENTITY, value)
        for key in ['model_hash', 'implementation_hash', 'settings']:
            encoder = {**IDENTITY, key: 'invalid'}
            with self.assertRaisesRegex(ContractError, 'encoder'): receipt(nodes, 'q', encoder, {'semantic': .5})
        changed = copy.deepcopy(original); changed['scores']['semantic'] = .2
        with self.assertRaisesRegex(ContractError, 'changed'): validate(changed, nodes, 'q')
        for value in [float('nan'), float('inf')]:
            with self.assertRaises((ContractError, ValueError)):
                receipt(nodes, 'q', IDENTITY, {'semantic': value})

    def model(self):
        root = self.root / 'model'
        write_json(root / 'config.json', {'model_type': 'qwen3'})
        for name in ['tokenizer.json', 'tokenizer_config.json', 'model.safetensors']:
            (root / name).write_text('fixture only')
        return {'model_path': str(root), 'query': 'research question'}

    @patch('rec_harness.semantic_encoder._runtime', return_value={'fixture': '1'})
    def test_outer_pin_and_resume_fail_on_model_or_runtime_changes(self, runtime):
        options = {'semantic_encoder': self.model()}
        pin_options(options)
        frozen = copy.deepcopy(options); freeze_options(frozen)
        verify_options(options, frozen)
        LocalEncoder(frozen['semantic_retrieval']).verify()
        changed = copy.deepcopy(options); changed['semantic_encoder']['query'] = 'new query'
        with self.assertRaisesRegex(ContractError, 'outer'): freeze_options(changed)
        runtime.return_value = {'fixture': '2'}
        with self.assertRaisesRegex(ContractError, 'runtime'): LocalEncoder(frozen['semantic_retrieval']).verify()
        runtime.return_value = {'fixture': '1'}
        (Path(options['semantic_encoder']['model_path']) / 'model.safetensors').write_text('mutated')
        with self.assertRaisesRegex(ContractError, 'runtime'): LocalEncoder(frozen['semantic_retrieval']).verify()
        with self.assertRaisesRegex(ContractError, 'outer'): freeze_options(options)

    @patch('rec_harness.semantic_encoder._runtime', return_value={'fixture': '1'})
    def test_campaign_entry_freezes_query_and_calls_encoder_without_calling_agent(self, runtime):
        example = Path(__file__).resolve().parents[1] / 'examples/toy'
        config = read_json(example / 'campaign.json')
        config.update(source=str(example), semantic_encoder=self.model())
        Research.init(self.root / 'campaign', config)
        research = Research(self.root / 'campaign', agent=object())
        self.addCleanup(research.store.close)
        self.assertNotIn('semantic_encoder', research.config)
        with patch.object(LocalEncoder, '__call__', side_effect=scored) as encode:
            context = research.context('hypothesis', {'incumbent': 'root', 'baseline': None})
        self.assertEqual(encode.call_args.args[1], 'research question\nResearch role: hypothesis')
        self.assertEqual(context['evidence']['retrieval']['method'], 'role-neural-two-hop-v1')
        self.assertFalse(any(e['kind'] == 'agent_submitted' for e in research.store.events()))

    def test_malformed_and_injected_options_fail_closed(self):
        for spec in [None, {}, {'model_path': 'remote/model', 'query': 'q'},
                     {**self.model(), 'threads': True}, {**self.model(), 'batch_size': 0}]:
            with self.subTest(spec=spec), self.assertRaises(ContractError): freeze(spec)
        with self.assertRaisesRegex(ContractError, 'controller-owned'):
            freeze_options({'semantic_retrieval': {}})
        empty = {}; freeze_options(empty); self.assertEqual(empty, {})

    @patch('rec_harness.semantic_encoder._runtime', return_value={'fixture': '1'})
    def test_persistent_receipt_reuses_original_timing_without_loading_ml(self, runtime):
        frozen = freeze(self.model())
        nodes = {'semantic': node('semantic', 'Observation', {}, 't', 'p')}
        identity = {key: frozen[key] for key in ('model_hash', 'implementation_hash', 'runtime', 'settings')}
        original = receipt(nodes, 'q', identity, {'semantic': .8}, execution={
            'elapsed_seconds': 7, 'cpu_seconds': 10, 'document_count': 1,
            'query_truncated': False, 'truncated_documents': []})
        key = digest({'nodes': nodes, 'query': 'q', 'encoder': frozen})
        path = self.root / 'cache' / (key + '.json')
        write_json(path, original)
        # The test environment has no transformers; only this persisted path can succeed.
        self.assertEqual(LocalEncoder(frozen, self.root / 'cache')(nodes, 'q'), original)
        changed = copy.deepcopy(original); changed['encoder']['model_hash'] = 'c' * 64
        changed['hash'] = digest({k: v for k, v in changed.items() if k != 'hash'})
        write_json(path, changed)
        with self.assertRaisesRegex(ContractError, 'cached'):
            LocalEncoder(frozen, self.root / 'cache')(nodes, 'q')

    def test_amazon_selection_retains_readout_closure_and_rejects_private_seed_before_encoding(self):
        from rec_harness.adapters.amazon_semantic import retrieve
        report = {'schema': 'amazon-development-selection-v1', 'protocol_hash': 'p', 'rows': [{'seed': 42}]}
        report['hash'] = digest(report)
        entries = [{'round': 'round-1', 'development': report, 'design': {'mechanism': 'long history'}}]
        def encoder(nodes, query):
            return receipt(nodes, query, IDENTITY, {k: .9 if k.startswith('memory-') else .1 for k in nodes})
        history, selected, failures, audit = retrieve([], entries, [], 'p', [42], 'q', encoder, 1)
        self.assertEqual(history, [report]); self.assertEqual(selected[0]['development_hash'], report['hash'])
        self.assertNotIn('development', selected[0]); self.assertEqual(failures, [])
        self.assertEqual(len(audit['readout_closure_ids']), 1)
        with patch('rec_harness.semantic_encoder.LocalEncoder.__call__') as encode:
            with self.assertRaisesRegex(ContractError, 'non-development'):
                retrieve([], entries, [], 'p', [83], 'q', encode, 1)
            encode.assert_not_called()


if __name__ == '__main__':
    unittest.main()
