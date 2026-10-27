from __future__ import annotations
import importlib.util
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_SPEC = importlib.util.spec_from_file_location('duorec_exported_encoder', os.path.join(_HERE, 'model_def.py'))
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _load(dataset):
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if dataset not in _MODELS:
        checkpoint = torch.load(os.path.join(_HERE, dataset + '_model.pt'), map_location='cpu', weights_only=True)
        model = _MODULE.SASRec(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        model.to(_DEVICE).eval()
        _MODELS[dataset] = model
    return _MODELS[dataset]


def predict(user_id: int, history: list[int], candidates: list[int], *, dataset_name: str | None = None) -> list[float]:
    if not candidates:
        return []
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    model = _load(dataset)
    hist = [int(item) for item in history[-model.maxlen:]]
    seq = [0] * (model.maxlen - len(hist)) + hist
    seq_t = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    cand_t = torch.tensor([candidates], dtype=torch.long, device=_DEVICE)
    with torch.inference_mode():
        scores = model.predict(seq_t, cand_t)[0]
    return scores.cpu().tolist()
