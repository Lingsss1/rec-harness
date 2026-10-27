from __future__ import annotations
import importlib.util
from pathlib import Path
import torch

_HERE = Path(__file__).resolve().parent
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = None
_CLASS = None


def _load(dataset):
    global _DEVICE, _CLASS
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if dataset in _MODELS:
        return _MODELS[dataset]
    if _DEVICE is None:
        torch.set_num_threads(4)
        _DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        spec = importlib.util.spec_from_file_location('trial_recurrent_model', _HERE / 'recurrent_model.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLASS = module.RecurrentMemory
    checkpoint = torch.load(_HERE / (dataset + '_model.pt'), map_location='cpu', weights_only=True)
    model = _CLASS(**checkpoint['config'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(_DEVICE).eval()
    _MODELS[dataset] = model
    return model


@torch.inference_mode()
def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    model = _load(dataset)
    hist = list(history[-model.maxlen:])
    seq = hist + [0] * (model.maxlen - len(hist))
    seq_t = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    cand_t = torch.tensor([candidates], dtype=torch.long, device=_DEVICE)
    return model.predict(seq_t, cand_t)[0].float().cpu().tolist()
