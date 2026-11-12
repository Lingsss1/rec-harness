"""Exact Taobao compatibility and native relation-group/task shapes."""
from collections import OrderedDict
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / 'UniRank/unirank').is_dir())
sys.path.insert(0, str(PROJECT / 'UniRank'))
import torch
from unirank.features import FeatureMap
from model import CONFIG, Model


class Contract(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, labels, fields, device='cpu', lists=False):
        fmap = FeatureMap('fixture', str(self.folder))
        fmap.labels = list(labels)
        fmap.features = OrderedDict([('user_id', {'type': 'categorical', 'vocab_size': 19, 'padding_idx': 0})])
        for name in fields:
            fmap.features[name] = {'type': 'categorical', 'source': 'item', 'vocab_size': 19, 'padding_idx': 0}
        fmap.features['action'] = {'type': 'categorical', 'source': 'action', 'vocab_size': 2 ** len(labels) + 1, 'padding_idx': 0}
        if lists:
            fmap.features['context_list'] = {'type': 'sequence', 'vocab_size': 9, 'max_len': 5,
                                             'padding_idx': 0, 'feature_encoder': 'layers.MaskedAveragePooling()'}
        fmap.num_fields = len(fmap.features)
        fmap.default_emb_dim = 16
        user = {'user_id': torch.arange(1, 9, device=device)}
        if lists:
            user['context_list'] = torch.tensor([[1, 3, 0, 0, 0]] * 8, device=device)
        items = {name: torch.randint(0, spec['vocab_size'], (8, 101), device=device)
                 for name, spec in fmap.features.items() if spec.get('source') in ['item', 'action']}
        items['action'][:, -1] = 0
        lengths = torch.tensor([100, 84, 37, 1, 0, 58, 99, 2], device=device)
        mask = (torch.arange(100, device=device)[None, :] >= 100 - lengths[:, None]).float()
        multi = [mask * ((torch.arange(100, device=device)[None, :] + i) % 3 == 0) for i in range(len(labels))]
        return fmap, (user, items, mask, multi)

    def create(self, cls, fmap, device='cpu', **kwargs):
        return cls(fmap, gpu=0 if device == 'cuda' else -1, verbose=0, metrics=['AUC'],
                   model_root=str(self.folder), enable_torch_compile=False, **{**CONFIG, **kwargs})

    def parity(self, device):
        source = Path(__file__).resolve().parent / 'taobao-original.py'
        if not source.exists():
            source = PROJECT / 'artifacts/unirank-taobao-macro-12h-20261109T043023Z/final-assessment-top3/suite/rank_1/model.py'
        spec = importlib.util.spec_from_file_location('original_current_best', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fields = ['item_id', 'cate_id', 'campaign_id', 'customer_id', 'brand', 'price_bucket']
        fmap, inputs = self.fixture(['is_click', 'cart', 'fav', 'buy'], fields, device)
        torch.manual_seed(123)
        old = self.create(module.Model, fmap, device)
        state = torch.get_rng_state().clone()
        torch.manual_seed(123)
        new = self.create(Model, fmap, device)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
        with torch.no_grad():
            for name, p in old.named_parameters():
                if 'weight' in name and not p.count_nonzero():
                    p.normal_(std=.02)
        new.load_state_dict(old.state_dict())
        predictions = []
        for model in [old, new]:
            model.train()
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                output = model(inputs)
                p = torch.stack([output[t + '_pred'].reshape(-1) for t in fmap.labels], -1).float()
            target = torch.arange(p.numel(), device=device).reshape_as(p).remainder(3).eq(0).float()
            loss = torch.nn.functional.binary_cross_entropy(p, target, reduction='none').mean(0).sum()
            loss.backward()
            predictions.append((p, loss))
        torch.testing.assert_close(predictions[0][0], predictions[1][0], rtol=0, atol=0)
        torch.testing.assert_close(predictions[0][1], predictions[1][1], rtol=0, atol=0)
        for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
            self.assertEqual(p.grad is None, q.grad is None, name)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        for name, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
        old.eval(); new.eval()
        with torch.no_grad():
            before, after = old(inputs), new(inputs)
            for name, value in before.items():
                torch.testing.assert_close(value, after[name], rtol=0, atol=0)

    def test_cpu_exact_taobao_parity(self):
        self.parity('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_bf16_exact_taobao_parity(self):
        self.parity('cuda')

    def test_native_field_and_task_cardinalities(self):
        for count, fields in [(2, 14), (4, 2), (5, 11), (6, 5)]:
            with self.subTest(tasks=count):
                labels = ['task' + str(i) for i in range(count)]
                fmap, inputs = self.fixture(labels, ['item_' + str(i) for i in range(fields)], lists=count == 2)
                model = self.create(Model, fmap)
                output = model(inputs)
                self.assertEqual(set(output), {t + '_pred' for t in labels})
                self.assertEqual(model.group_count, fields + 1)
                self.assertEqual(model.pair_bank.shape[0], fields * (fields - 1) // 2)
                self.assertEqual(model.feedback_memory.identity.shape[2], count)
                self.assertEqual(model.feedback_content_bank.shape[0], count)
                self.assertEqual(model.feedback_memory.group_count, fields + 1)
                self.assertEqual(len(model.task_scorers), count)
                empty = (inputs[0], inputs[1], inputs[2] * 0, [m * 0 for m in inputs[3]])
                self.assertTrue(all(torch.isfinite(v).all() for v in model(empty).values()))
                sum(p.sum() for p in output.values()).backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
