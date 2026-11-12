"""Parity of the inherited model and task-count/schema boundary checks."""
from collections import OrderedDict
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(PROJECT / 'UniRank'))
import torch
from unirank.features import FeatureMap
from common import aggregate
from model import CONFIG, Model


def original_class():
    path = PROJECT / 'artifacts/unirank-taobao-latest/best-model-multitask-audit/model.py'
    spec = importlib.util.spec_from_file_location('original_taobao', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, labels, fields=('cate_id', 'brand'), sequence=False):
        fmap = FeatureMap('fixture', str(self.folder))
        fmap.labels = labels
        fmap.features = OrderedDict([
            ('user_id', {'type': 'categorical', 'vocab_size': 13, 'padding_idx': 0}),
            ('item_id', {'type': 'categorical', 'vocab_size': 17, 'padding_idx': 0, 'source': 'item'}),
        ])
        for name in fields:
            if name is not None:
                fmap.features[name] = {'type': 'categorical', 'vocab_size': 7, 'source': 'item', 'padding_idx': 0}
        fmap.features['action'] = {'type': 'categorical', 'vocab_size': 2 ** len(labels) + 1,
                                   'source': 'action', 'padding_idx': 0}
        if sequence:
            fmap.features['list_context'] = {'type': 'sequence', 'vocab_size': 8, 'padding_idx': 0,
                                              'max_len': 5, 'feature_encoder': 'layers.MaskedAveragePooling()'}
        fmap.num_fields = len(fmap.features)
        fmap.default_emb_dim = 4
        mask = torch.tensor([[1., 1., 1., 0.], [1., 0., 0., 0.], [0., 0., 0., 0.], [1., 1., 1., 1.]])
        user = {'user_id': torch.tensor([1, 2, 3, 4])}
        if sequence:
            user['list_context'] = torch.tensor([[1, 2, 0, 0, 0], [0, 0, 0, 0, 0], [3, 4, 5, 0, 0], [7, 0, 0, 0, 0]])
        items = {name: torch.tensor([[1, 2, 3, 0, 3], [1, 0, 0, 0, 2], [0, 0, 0, 0, 4], [1, 2, 3, 4, 1]])
                 for name in fmap.features if fmap.features[name].get('source') in ['item', 'action']}
        items['action'][:, -1] = 0
        multi = [mask * ((torch.arange(4)[None, :] + i) % 2) for i in range(len(labels))]
        return fmap, (user, items, mask, multi)

    def create(self, cls, fmap, **kwargs):
        config = {**CONFIG, 'embedding_dim': 4, 'attention_hidden_units': [8, 4], 'tower_hidden_units': [8, 4], **kwargs}
        return cls(fmap, gpu=-1, verbose=0, metrics=['AUC'], model_root=str(self.folder),
                   enable_torch_compile=False, **config)

    def test_taobao_identical_initialization_forward_backward_and_buffers(self):
        fmap, inputs = self.fixture(['is_click', 'cart', 'fav', 'buy'])
        torch.manual_seed(123)
        original = self.create(original_class(), fmap)
        state_after_original = torch.get_rng_state().clone()
        torch.manual_seed(123)
        generalized = self.create(Model, fmap)
        self.assertTrue(torch.equal(state_after_original, torch.get_rng_state()))
        self.assertEqual(set(original.state_dict()), set(generalized.state_dict()))
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, generalized.state_dict()[key], rtol=0, atol=0)
        # Exercise nonzero gate/projection weights, not merely their all-zero initial state.
        with torch.no_grad():
            for name, parameter in original.named_parameters():
                if name.endswith('_weight') and parameter.count_nonzero() == 0:
                    parameter.normal_(std=0.02)
        generalized.load_state_dict(original.state_dict())
        original.train()
        generalized.train()
        before = original(inputs)
        after = generalized(inputs)
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
        sum(x.sum() for x in before.values()).backward()
        sum(x.sum() for x in after.values()).backward()
        for (name, a), (other, b) in zip(original.named_parameters(), generalized.named_parameters()):
            self.assertEqual(name, other)
            self.assertEqual(a.grad is None, b.grad is None, name)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)

    def test_native_task_counts_missing_match_and_sequence_context(self):
        for count in [2, 4, 5, 6]:
            with self.subTest(count=count):
                labels = ['task' + str(i) for i in range(count)]
                fields = ('metadata', None) if count == 4 else ('m1', 'm2')
                fmap, inputs = self.fixture(labels, fields, sequence=count == 2)
                model = self.create(Model, fmap, primary_task=labels[0], match_fields=fields)
                result = model(inputs)
                self.assertEqual(set(result), {x + '_pred' for x in labels})
                loss = sum(x.mean() for x in result.values())
                loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                evidence, channels, matches = model._history_evidence(inputs[1], inputs[2], inputs[3])
                self.assertEqual(evidence.shape, (4, (count + 1) * 8))
                self.assertTrue(torch.equal(evidence[2], torch.zeros_like(evidence[2])))
                if fields[1] is None:
                    self.assertEqual(matches[:, :, 2:].count_nonzero(), 0)

    def test_macro_requires_all_tasks(self):
        tasks = {'a': {'auc': 0.6, 'logloss': 0.2}, 'b': {'auc': 0.8, 'logloss': 0.4}}
        self.assertAlmostEqual(aggregate(tasks, ['a', 'b'])['auc'], 0.7)
        with self.assertRaises(ValueError):
            aggregate(tasks, ['a', 'b', 'c'])
        tasks['b']['auc'] = None
        self.assertIsNone(aggregate(tasks, ['a', 'b'], allow_undefined=True)['auc'])


if __name__ == '__main__':
    unittest.main()
