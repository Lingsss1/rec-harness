"""Preserve the click champion's forward, surrogate gradients, losses and task-count boundaries."""
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
from model import CONFIG, Model, _auxiliary_context_components, _CollectedContextTransfer


def original_class():
    source = PROJECT / 'artifacts/unirank-taobao-20261106T090344Z/topics/gpu-1/trials/g1-r0088-membership_full_first_mean_only/model.py'
    spec = importlib.util.spec_from_file_location('original_click_champion', source)
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

    def fixture(self, labels, fields=('item_id', 'cate_id', 'brand'), list_context=False, device='cpu'):
        fmap = FeatureMap('fixture', str(self.folder))
        fmap.labels = list(labels)
        fmap.features = OrderedDict([('user_id', {'type': 'categorical', 'vocab_size': 17, 'padding_idx': 0})])
        for name in fields:
            if name is not None:
                fmap.features[name] = {'type': 'categorical', 'source': 'item', 'vocab_size': 17, 'padding_idx': 0}
        fmap.features['action'] = {'type': 'categorical', 'source': 'action', 'vocab_size': 2 ** len(labels) + 1, 'padding_idx': 0}
        if list_context:
            fmap.features['list_context'] = {'type': 'sequence', 'vocab_size': 8, 'padding_idx': 0, 'max_len': 5,
                                            'feature_encoder': 'layers.MaskedAveragePooling()'}
        fmap.default_emb_dim = 16
        fmap.num_fields = len(fmap.features)
        lengths = torch.tensor([100, 73, 17, 0, 1, 50, 2, 99], device=device)
        mask = (torch.arange(100, device=device)[None, :] >= 100 - lengths[:, None]).float()
        user = {'user_id': torch.tensor([1, 2, 3, 1, 4, 2, 3, 1], device=device)}
        if list_context:
            user['list_context'] = torch.tensor([[1, 2, 0, 0, 0]] * 8, device=device)
            user['list_context'][3] = 0
        items = {name: torch.randint(0, spec['vocab_size'], (8, 101), device=device) for name, spec in fmap.features.items()
                 if spec.get('source') in ['item', 'action']}
        items['action'][:, -1] = 0
        for name in fields:
            if name is not None:
                items[name][:, -1] = 3
        multi = [mask * ((torch.arange(100, device=device)[None, :] + i) % 3 == 0)
                 for i in range(len(labels))]
        return fmap, (user, items, mask, multi)

    def create(self, cls, fmap, device='cpu', **kwargs):
        config = {**CONFIG, **kwargs}
        return cls(fmap, gpu=0 if device == 'cuda' else -1, verbose=0, metrics=['AUC'],
                   model_root=str(self.folder), enable_torch_compile=False, **config)

    def parity(self, device):
        torch.manual_seed(123)
        fmap, inputs = self.fixture(['is_click', 'cart', 'fav', 'buy'], device=device)
        torch.manual_seed(456)
        old = self.create(original_class(), fmap, device)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(456)
        new = self.create(Model, fmap, device)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        for name, tensor in old.state_dict().items():
            torch.testing.assert_close(tensor, new.state_dict()[name], rtol=0, atol=0)
        # Exercise context-transfer VJPs beyond their zero-initialized correction heads.
        with torch.no_grad():
            for name, parameter in old.named_parameters():
                if parameter.numel() and not parameter.count_nonzero() and 'weight' in name:
                    parameter.normal_(std=.03)
        new.load_state_dict(old.state_dict())
        target = torch.arange(32, device=device).reshape(8, 4).remainder(3).eq(0).float()
        predictions = []
        for model in [old, new]:
            model.train()
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                outputs = model(inputs)
                values = torch.stack([outputs[label + '_pred'].reshape(-1) for label in fmap.labels], -1).float()
            loss = model.training_loss(inputs, values, target)
            loss.backward()
            predictions.append((values.detach(), loss.detach()))
        torch.testing.assert_close(predictions[0][0], predictions[1][0], rtol=0, atol=0)
        torch.testing.assert_close(predictions[0][1], predictions[1][1], rtol=0, atol=0)
        for (name, a), (name_b, b) in zip(old.named_parameters(), new.named_parameters()):
            self.assertEqual(name, name_b)
            self.assertEqual(a.grad is None, b.grad is None, name)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
        old.eval()
        new.eval()
        with torch.no_grad():
            before, after = old(inputs), new(inputs)
            for label, value in before.items():
                torch.testing.assert_close(value, after[label], rtol=0, atol=0)

    def test_cpu_taobao_exact_parity(self):
        self.parity('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_bf16_taobao_exact_parity(self):
        self.parity('cuda')

    def test_native_task_counts_missing_metadata_and_empty_history(self):
        for count in [2, 4, 5, 6]:
            with self.subTest(count=count):
                labels = ['task' + str(i) for i in range(count)]
                fields = ('item_id', 'category', None) if count == 4 else ('item_id', 'metadata1', 'metadata2')
                fmap, inputs = self.fixture(labels, fields, list_context=count == 2)
                model = self.create(Model, fmap, primary_task=labels[0], match_fields=fields)
                values = torch.stack([v.reshape(-1) for v in model(inputs).values()], -1).float()
                loss = model.training_loss(inputs, values, torch.zeros_like(values))
                loss.backward()
                self.assertEqual(values.shape, (8, count))
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                if fields[-1] is None:
                    summary = model._represent(inputs)
                    self.assertEqual(summary[-2][:, -1].count_nonzero(), 0)

    def test_auxiliary_gradient_sum_keeps_native_order(self):
        occupied = torch.tensor([[True], [True], [False], [True]])
        for count in [1, 3, 4, 5]:
            gradients = [torch.randn(4, 7) for _ in range(count)]
            expected = torch.where(occupied, gradients[0].double(), 0.)
            for value in gradients[1:]:
                expected = expected + torch.where(occupied, value.double(), 0.)
            torch.testing.assert_close(_auxiliary_context_components(gradients, occupied)['full'], expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
