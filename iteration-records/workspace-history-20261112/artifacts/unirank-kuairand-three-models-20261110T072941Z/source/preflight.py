"""Numerical smoke checks and exact shared initialization for the depth control."""
import copy
from pathlib import Path
import sys
import tempfile

import torch

from common import read, write


def check(root):
    setup = read(root / 'setup.json')
    sys.path.insert(0, setup['upstream_path'])
    from unirank.features import FeatureMap
    from model import CONFIG, Model
    from base_model import Model as OriginalModel
    torch.set_num_threads(2)
    folder = root / 'datasets/KuaiRand'
    params = read(folder / 'parameters.json')
    fmap = FeatureMap(params['dataset_id'], str(folder / 'prepared'))
    fmap.load(str(folder / 'prepared/feature_map.json'), {'embedding_dim': 16})
    fixture = copy.deepcopy(fmap)
    for spec in fixture.features.values():
        if 'vocab_size' in spec:
            spec['vocab_size'] = min(spec['vocab_size'], 17)
    checks = []
    with tempfile.TemporaryDirectory() as temporary:
        for device in ['cpu', 'cuda']:
            gpu = 0 if device == 'cuda' else -1
            settings = {**params, **CONFIG, 'gpu': gpu, 'model_root': temporary}
            settings.pop('joint_layers')
            torch.manual_seed(123)
            base = OriginalModel(fixture, **settings)
            torch.manual_seed(123)
            one = Model(fixture, joint_layers=1, **settings)
            rng = torch.get_rng_state().clone()
            torch.manual_seed(123)
            three = Model(fixture, joint_layers=3, **settings)
            assert torch.equal(rng, torch.get_rng_state()), 'Depth changed host sampling RNG'
            assert len(three.interaction.blocks) == 3
            assert len({id(b) for b in three.interaction.blocks}) == 3
            for key, value in one.state_dict().items():
                torch.testing.assert_close(value, base.state_dict()[key], atol=0, rtol=0)
                mapped = 'interaction.blocks.0.' + key[len('interaction.'):] if key.startswith('interaction.') else key
                torch.testing.assert_close(value, three.state_dict()[mapped], atol=0, rtol=0)
            assert not torch.equal(three.interaction.blocks[0].qkv.weight,
                                   three.interaction.blocks[1].qkv.weight)
            user, items = {}, {}
            for name, spec in fixture.features.items():
                if spec['type'] == 'meta':
                    continue
                target = items if spec.get('source') in ['item', 'action'] else user
                shape = (4, 101) if target is items else (4, spec.get('max_len', 4)) if spec['type'] == 'sequence' else (4,)
                if 'vocab_size' in spec:
                    target[name] = torch.randint(0, spec['vocab_size'], shape, device=device)
                else:
                    target[name] = torch.ones(shape, device=device)
                if spec.get('source') == 'action':
                    target[name][:, -1] = 0
            lengths = torch.tensor([0, 1, 37, 100], device=device)
            valid = (torch.arange(100, device=device)[None, :] >= 100 - lengths[:, None]).float()
            inputs = (user, items, valid, [valid * ((torch.arange(100, device=device)[None, :] + i) % 3 == 0)
                                          for i in range(6)])
            outputs = []
            for model in [base, one, three]:
                model.train()
                with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                    output = model(inputs)
                    predictions = torch.cat([output[t + '_pred'] for t in fixture.labels], -1).float()
                assert predictions.shape == (4, 6) and torch.isfinite(predictions).all()
                target = torch.arange(24, device=device).reshape(4, 6).remainder(3).eq(0).float()
                loss = torch.nn.functional.binary_cross_entropy(predictions, target, reduction='none').mean(0).sum()
                loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                outputs.append((predictions, loss))
            for a, b in zip(outputs[0], outputs[1]):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
            for (name, a), (_, b) in zip(base.named_parameters(), one.named_parameters()):
                assert (a.grad is None) == (b.grad is None), name
                if a.grad is not None:
                    torch.testing.assert_close(a.grad, b.grad, atol=0, rtol=0)
            for block in three.interaction.blocks:
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters())
            checks.append({'device': device, 'token_dim': 256, 'embedding_dim': 16,
                           'one_layer_exact_base_parity': True, 'three_distinct_blocks_with_gradients': True,
                           'all_common_parameters_and_sampling_rng_preserved': True})
            del base, one, three
    write(root / 'interface-checks.json', {'status': 'passed', 'checks': checks})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    check(parser.parse_args().root.resolve())
