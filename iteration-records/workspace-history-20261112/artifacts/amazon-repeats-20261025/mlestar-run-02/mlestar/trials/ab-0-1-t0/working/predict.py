from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_models = {}
_metas = {}
_device = None
_model_class = None


def _load_dataset(name):
    global _device, _model_class
    if name in _models:
        return
    if name not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(name))
    if _device is None:
        torch.set_num_threads(1)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        spec = importlib.util.spec_from_file_location('gsasrec_trial_model', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _model_class = module.SASRec
    with open(os.path.join(_HERE, name + '_meta.pkl'), 'rb') as f:
        meta = pickle.load(f)
    config = dict(meta['config'])
    config['dropout_rate'] = 0.0
    model = _model_class(**config).to(_device)
    state = torch.load(os.path.join(_HERE, name + '_model.pt'), map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    _models[name] = model
    _metas[name] = meta


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    name = dataset_name if dataset_name is not None else DATASETS[0]
    _load_dataset(name)
    if len(candidates) == 0:
        return []
    config = _metas[name]['config']
    maxlen = config['maxlen']
    hist = list(history[-maxlen:])
    seq = [0] * (maxlen - len(hist)) + hist
    with torch.inference_mode():
        seq_t = torch.tensor([seq], dtype=torch.long, device=_device)
        cand_t = torch.tensor([candidates], dtype=torch.long, device=_device)
        return _models[name].predict(seq_t, cand_t)[0].cpu().tolist()
