"""Verify that the host adapter preserves the width-80 HeMix implementation."""
from collections import OrderedDict
from pathlib import Path
import sys
import tempfile
import unittest

PROJECT = next(p for p in Path(__file__).resolve().parents if (p / 'UniRank/unirank').is_dir())
sys.path.insert(0, str(PROJECT / 'UniRank'))
import torch
from unirank.features import FeatureMap
from model import CONFIG, Model
from HeMix import HeMix


class Contract(unittest.TestCase):
    def parity(self, device):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder:
            fmap = FeatureMap('fixture', folder)
            fmap.labels = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward', 'long_view']
            fmap.features = OrderedDict([('user_id', {'type': 'categorical', 'vocab_size': 17, 'padding_idx': 0})])
            fields = ['item_id', 'video_type', 'primary_tag', 'music_type', 'duration_bucket']
            for name in fields:
                fmap.features[name] = {'type': 'categorical', 'source': 'item', 'vocab_size': 17, 'padding_idx': 0}
            fmap.features['action'] = {'type': 'categorical', 'source': 'action', 'vocab_size': 65, 'padding_idx': 0}
            fmap.num_fields = len(fmap.features)
            fmap.default_emb_dim = 16
            args = {**CONFIG, 'gpu': 0 if device == 'cuda' else -1, 'verbose': 0, 'metrics': ['AUC'],
                    'model_root': folder, 'enable_torch_compile': False, 'num_tasks': 6,
                    'task': 'binary_classification', 'loss': ['binary_crossentropy'] * 6,
                    'dense_optimizer': 'AdamW', 'dense_learning_rate': 1e-4,
                    'sparse_optimizer': 'Adagrad', 'sparse_learning_rate': 0.05}
            torch.manual_seed(61)
            old = HeMix(fmap, **args)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(61)
            new = Model(fmap, **args)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(set(old.state_dict()), set(new.state_dict()))
            for k, v in old.state_dict().items():
                torch.testing.assert_close(v, new.state_dict()[k], atol=0, rtol=0)
            user = {'user_id': torch.arange(1, 5, device=device)}
            items = {name: torch.randint(1, 16, (4, 101), device=device) for name in fields + ['action']}
            items['action'][:, -1] = 0
            lengths = torch.tensor([0, 1, 24, 100], device=device)
            valid = (torch.arange(100, device=device)[None, :] >= 100 - lengths[:, None]).float()
            inputs = (user, items, valid, [valid.clone() for _ in fmap.labels])
            results = []
            for model in [old, new]:
                model.train()
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                    result = model(inputs)
                    pred = torch.cat([result[t + '_pred'] for t in fmap.labels], dim=-1).float()
                self.assertEqual(pred.shape, (4, 6))
                self.assertTrue(torch.isfinite(pred).all())
                y = torch.arange(24, device=device).reshape(4, 6).remainder(3).eq(0).float()
                loss = torch.nn.functional.binary_cross_entropy(pred, y, reduction='none').mean(0).sum()
                loss.backward()
                results.append((pred, loss))
            for a, b in zip(results[0], results[1]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            for (name, a), (_, b) in zip(old.named_parameters(), new.named_parameters()):
                self.assertEqual(a.grad is None, b.grad is None, name)
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)

    def test_cpu_width80_exact_parity(self):
        self.parity('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_bf16_width80_exact_parity(self):
        self.parity('cuda')


if __name__ == '__main__':
    unittest.main()
