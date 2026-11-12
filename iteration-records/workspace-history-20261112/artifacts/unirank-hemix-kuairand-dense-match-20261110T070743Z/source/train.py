"""Fixed paired cross-dataset training; host-owned labels, selection and final test."""
import argparse
import copy
import importlib.util
from pathlib import Path
import random
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from common import aggregate, read, sha, write


def framework(root):
    upstream = Path(read(root / 'setup.json')['upstream_path'])
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))


def make_model(root, folder, kind, output, device=0):
    framework(root)
    from unirank.features import FeatureMap
    params = read(folder / 'parameters.json')
    source = root / 'source' / ('model.py' if kind == 'candidate' else 'baseline.py')
    spec = importlib.util.spec_from_file_location('fixed_' + kind, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = {**module.CONFIG, **read(folder / (kind + '-config.json'))}
    fmap = FeatureMap(params['dataset_id'], str(folder / 'prepared'))
    fmap.load(str(folder / 'prepared/feature_map.json'), {'embedding_dim': 16})
    model = module.Model(fmap, gpu=device, model_id=kind, model_root=str(output), **params, **config)
    return model, fmap, params, config, source


def loader(folder, fmap, params, split):
    from unirank.pytorch.dataloaders.unirank_dataloader import UniRankDataloader
    protocol = read(folder / 'protocol.json')
    kwargs = {key: value for key, value in params.items() if key not in ['split', 'shuffle']}
    return UniRankDataloader(fmap, data_path=params[split + '_data'], split=split,
                            batch_size=protocol['batch_size'], num_workers=0,
                            max_len=protocol['history_limit'], shuffle=split == 'train',
                            drop_last=split == 'train', **kwargs)


def features_only(inputs, labels, device):
    batch, items, mask, multi = inputs
    batch = {name: value.to(device, non_blocking=True) for name, value in batch.items()
             if name not in labels and name not in ['user_index', 'item_index', 'seq_len']}
    return (batch, {name: value.to(device, non_blocking=True) for name, value in items.items()},
            mask.to(device, non_blocking=True), [x.to(device, non_blocking=True) for x in multi])


def predict(model, features, labels):
    result = model(features)
    if set(result) != {name + '_pred' for name in labels}:
        raise ValueError('Every native task requires exactly one prediction')
    values = torch.stack([result[name + '_pred'].reshape(-1) for name in labels], -1).float()
    if values.shape != (len(features[2]), len(labels)) or not torch.isfinite(values).all():
        raise ValueError('Invalid prediction shape or nonfinite probabilities')
    if (values < 0).any() or (values > 1).any():
        raise ValueError('Predictions must be probabilities')
    return values


def binary_loss(y, p):
    total = 0.0
    for start in range(0, len(y), 1_000_000):
        target = y[start:start + 1_000_000]
        prob = np.clip(p[start:start + 1_000_000].astype(np.float64), 1e-7, 1 - 1e-7)
        total -= (target * np.log(prob) + (1 - target) * np.log1p(-prob)).sum()
    return float(total / len(y))


def score(model, iterator, labels, device, output, split, expected):
    started = time.monotonic()
    model.eval()
    # Preallocate to avoid retaining thousands of batch arrays or duplicate concatenations.
    y = np.empty((expected, len(labels)), dtype=np.uint8)
    p = np.empty((expected, len(labels)), dtype=np.float32)
    lengths = np.empty(expected, dtype=np.uint16)
    position = 0
    latencies = []
    with torch.inference_mode():
        for index, inputs in enumerate(iterator):
            targets = np.stack([inputs[0][label].numpy() for label in labels], axis=-1).reshape(-1, len(labels))
            if not np.isin(targets, [0, 1]).all():
                raise ValueError('Nonbinary label')
            batch_size = len(targets)
            if position + batch_size > expected:
                raise ValueError('Scoring exceeded released split sample count')
            features = features_only(inputs, labels, device)
            torch.cuda.synchronize()
            stamp = time.monotonic()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                values = predict(model, features, labels)
            torch.cuda.synchronize()
            latencies.append(time.monotonic() - stamp)
            y[position:position + batch_size] = targets
            p[position:position + batch_size] = values.cpu().numpy()
            lengths[position:position + batch_size] = inputs[2].sum(-1).numpy()
            position += batch_size
            if index % 20 == 0:
                write(output / 'progress.json', {'phase': split, 'batches': index + 1, 'samples': position,
                      'total_samples': expected, 'wall_seconds': time.monotonic() - started, 'at_unix': time.time()})
    if position != expected:
        raise ValueError(f'{split} coverage {position} != official {expected}')
    inference_seconds = time.monotonic() - started
    write(output / 'progress.json', {'phase': split + '_metrics', 'samples': position,
          'total_samples': expected, 'at_unix': time.time()})
    tasks = {}
    for i, label in enumerate(labels):
        if not (0 < y[:, i].sum() < expected):
            raise ValueError('Both classes required for ' + label)
        tasks[label] = {'auc': float(roc_auc_score(y[:, i], p[:, i])),
                       'logloss': binary_loss(y[:, i], p[:, i]),
                       'positives': int(y[:, i].sum()), 'samples': expected}
    slices = {}
    for name, selected in [('empty_history', lengths == 0), ('history_1_16', (lengths > 0) & (lengths <= 16)),
                           ('history_17_64', (lengths > 16) & (lengths <= 64)), ('history_gt64', lengths > 64)]:
        count = int(selected.sum())
        slice_tasks = {}
        if count:
            for i, label in enumerate(labels):
                sy = y[selected, i]
                sp = p[selected, i]
                positive = int(sy.sum())
                slice_tasks[label] = {'auc': float(roc_auc_score(sy, sp)) if 0 < positive < count else None,
                                     'logloss': binary_loss(sy, sp), 'positives': positive}
        slices[name] = {'samples': count, 'per_task': slice_tasks}
        if count:
            slices[name].update(aggregate(slice_tasks, labels, allow_undefined=True))
    return {**aggregate(tasks, labels), 'per_task': tasks, 'samples': expected, 'slices': slices,
            'inference_seconds': inference_seconds, 'scoring_seconds': time.monotonic() - started,
            'examples_per_second': expected / inference_seconds,
            'batch_latency_ms_p50': float(np.percentile(latencies[3:] or latencies, 50) * 1000),
            'batch_latency_ms_p95': float(np.percentile(latencies[3:] or latencies, 95) * 1000),
            'latency_unit': 'model-only batch; wall throughput includes data loading',
            'inference_batch_size': iterator._configured_batch_size}


def run(args):
    root = args.root.resolve()
    folder = root / 'datasets' / args.dataset
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'result.json').exists():
        raise ValueError('Completed runs are immutable; use a new attempt directory')
    protocol = read(folder / 'protocol.json')
    acquisition = read(folder / 'acquisition-status.json')
    if acquisition['phase'] != 'downloaded_and_verified':
        raise ValueError('Full pinned release verification is required before training')
    for name, expected_hash in protocol['source_sha256'].items():
        if sha(root / 'source' / name) != expected_hash:
            raise ValueError('Frozen source changed: ' + name)
    started = time.monotonic()
    seed = protocol['seed']
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    write(output / 'progress.json', {'phase': 'initializing', 'at_unix': time.time()})
    model, fmap, params, effective, source = make_model(root, folder, args.kind, output)
    labels = list(fmap.labels)
    if labels != protocol['tasks']:
        raise ValueError('Task order differs from frozen metadata')
    embedding_ids = {id(p) for module in model.modules() if isinstance(module, torch.nn.Embedding)
                     for p in module.parameters(recurse=False)}
    sparse = [p for p in model.parameters() if p.requires_grad and id(p) in embedding_ids]
    dense = [p for p in model.parameters() if p.requires_grad and id(p) not in embedding_ids]
    optimizers = [torch.optim.AdamW(dense, lr=protocol['dense_learning_rate'], weight_decay=0, foreach=False),
                  torch.optim.Adagrad(sparse, lr=protocol['embedding_learning_rate'], foreach=False)]
    write(output / 'progress.json', {'phase': 'loading_train', 'at_unix': time.time()})
    train = loader(folder, fmap, params, 'train')
    if train.global_num_samples != protocol['sample_size']['train']:
        raise ValueError('Training split coverage differs from publisher metadata')
    steps = 0
    training_loss_hook_calls = 0
    examples = 0
    epochs = []
    durations = []
    best = None
    best_key = None
    training_seconds = 0.0
    for epoch in range(protocol['epochs']):
        model.train()
        train.set_epoch(epoch)
        epoch_started = time.monotonic()
        previous = epoch_started
        loss_sum = 0.0
        epoch_examples = 0
        for inputs in train:
            targets = torch.stack([inputs[0][label].to(device).reshape(-1) for label in labels], -1).float()
            if not ((targets == 0) | (targets == 1)).all():
                raise ValueError('Nonbinary training label')
            features = features_only(inputs, labels, device)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                values = predict(model, features, labels)
            loss = torch.nn.functional.binary_cross_entropy(values, targets, reduction='none').mean(0).sum()
            if hasattr(model, 'training_loss'):
                loss = model.training_loss(features, values, targets)
                training_loss_hook_calls += 1
            if loss.ndim != 0:
                raise ValueError('Training loss must be one scalar')
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dense, protocol['gradient_clip'])
            for optimizer in optimizers:
                optimizer.step()
            torch.cuda.synchronize()
            steps += 1
            examples += len(targets)
            epoch_examples += len(targets)
            loss_sum += float(loss.detach()) * len(targets)
            stamp = time.monotonic()
            durations.append(stamp - previous)
            previous = stamp
            if steps % 10 == 0:
                avg = float(np.mean(durations[3:] or durations))
                write(output / 'progress.json', {'phase': 'calibration' if args.calibration_steps else 'train',
                      'epoch': epoch + 1, 'steps': steps, 'total_steps': len(train) * protocol['epochs'],
                      'examples': examples, 'loss': float(loss.detach()), 'mean_step_seconds': avg,
                      'eta_training_seconds': max(0, len(train) * protocol['epochs'] - steps) * avg,
                      'wall_seconds': stamp - started, 'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                      'at_unix': time.time()})
            if args.calibration_steps and steps >= args.calibration_steps:
                break
        seconds = time.monotonic() - epoch_started
        training_seconds += seconds
        if args.calibration_steps:
            result = {'status': 'success', 'calibration_only': True, 'steps': steps, 'training_examples': examples,
                      'wall_seconds': time.monotonic() - started, 'train_seconds': training_seconds,
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'embedding_parameters': sum(p.numel() for p in sparse),
                      'dense_parameters': sum(p.numel() for p in dense),
                      'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                      'training_loss_hook_calls': training_loss_hook_calls,
                      'steady_step_seconds': float(np.mean(durations[3:] or durations)),
                      'projected_train_seconds': len(train) * float(np.mean(durations[3:] or durations))}
            write(output / 'result.json', result)
            return
        # Drop optimizer gradients before validation, while preserving the exact checkpoint.
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        valid = loader(folder, fmap, params, 'valid')
        validation = score(model, valid, labels, device, output, 'valid', protocol['sample_size']['valid'])
        epochs.append({'epoch': epoch + 1, 'train_seconds': seconds,
                       'training_loss': loss_sum / epoch_examples, 'validation': validation})
        key = (-validation['auc'], validation['logloss'])
        if best_key is None or key < best_key:
            best_key = key
            best = copy.deepcopy(validation)
            torch.save(model.state_dict(), output / 'checkpoint.pt')
        del valid
    checkpoint_sha = sha(output / 'checkpoint.pt')
    # This immutable file is written BEFORE opening a test iterator or computing test predictions.
    write(output / 'selection.json', {'selected_unix': time.time(), 'criterion': protocol['selection'],
          'validation': best, 'checkpoint_sha256': checkpoint_sha, 'protocol_hash': protocol['identity'],
          'source_sha256': sha(source), 'test_used_for_selection': False})
    model.load_state_dict(torch.load(output / 'checkpoint.pt', map_location=device, weights_only=True))
    del optimizers
    torch.cuda.empty_cache()
    test = loader(folder, fmap, params, 'test')
    heldout = score(model, test, labels, device, output, 'test', protocol['sample_size']['test'])
    result = {'status': 'success', 'dataset': args.dataset, 'kind': args.kind, 'seed': seed,
              'source_sha256': sha(source), 'effective_config': effective, 'protocol_hash': protocol['identity'],
              'dataset_revision': protocol['revision'], 'validation': best, 'test': heldout, 'epochs': epochs,
              'optimizer_steps': steps, 'training_examples': examples, 'train_seconds': training_seconds,
              'training_loss_hook_calls': training_loss_hook_calls,
              'wall_seconds': time.monotonic() - started, 'parameters': sum(p.numel() for p in model.parameters()),
              'embedding_parameters': sum(p.numel() for p in sparse), 'dense_parameters': sum(p.numel() for p in dense),
              'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'hardware': torch.cuda.get_device_name(),
              'initialization': 'from_scratch', 'checkpoint_sha256': checkpoint_sha,
              'selection_policy': protocol['selection'], 'test_used_for_selection': False,
              'one_epoch_fixed_recipe': True, 'single_seed': True}
    write(output / 'result.json', result)
    write(output / 'progress.json', {'phase': 'complete', 'at_unix': time.time(),
          'test_macro_auc': heldout['auc'], 'wall_seconds': result['wall_seconds']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--kind', choices=['candidate', 'din'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--calibration-steps', type=int, default=0)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException as error:
        write(args.output / 'failure.json', {'status': 'failed', 'error': repr(error), 'at_unix': time.time()})
        raise
