from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_model_class = None


def _load(dataset):
    global _device, _model_class
    if dataset in _models:
        return
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if _device is None:
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if _model_class is None:
        spec = importlib.util.spec_from_file_location('fmlp_trial_architecture', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.FMLPRec
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as handle:
        meta = pickle.load(handle)
    if meta['architecture'] != 'fmlp' or meta.get('ablation') != 'no_pool_recency':
        raise ValueError('Unsupported model architecture or ablation')
    config = {key: meta[key] for key in ('itemnum', 'hidden_units', 'num_blocks', 'maxlen', 'dropout_rate')}
    model = _model_class(**config).to(_device)
    model.load_state_dict(torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location=_device, weights_only=True))
    model.eval()
    _models[dataset] = model
    _metas[dataset] = meta


def predict(user_id, history, candidates, *, dataset_name=None):
    if not candidates:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    _load(dataset)
    meta = _metas[dataset]
    model = _models[dataset]
    prefix = list(history)[-meta['maxlen']:]
    prefix = [int(item) if 0 < int(item) <= meta['itemnum'] else 0 for item in prefix]
    sequence = [0] * (meta['maxlen'] - len(prefix)) + prefix
    safe_candidates = [int(item) if 0 < int(item) <= meta['itemnum'] else 0 for item in candidates]
    with torch.inference_mode():
        seq = torch.tensor([sequence], dtype=torch.long, device=_device)
        cand = torch.tensor([safe_candidates], dtype=torch.long, device=_device)
        scores = model.predict(seq, cand)[0].cpu().tolist()
    return [score if item != 0 else -1e9 for score, item in zip(scores, safe_candidates)]
