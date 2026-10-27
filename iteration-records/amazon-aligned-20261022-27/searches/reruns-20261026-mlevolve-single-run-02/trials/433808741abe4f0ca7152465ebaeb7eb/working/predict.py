from __future__ import annotations
import importlib.util
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
             'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_MODULE = None


def _get_model(dataset_name):
    global _MODULE
    if dataset_name not in _DATASETS:
        raise ValueError('Unknown dataset_name: ' + str(dataset_name))
    if dataset_name not in _MODELS:
        if _MODULE is None:
            spec = importlib.util.spec_from_file_location(
                'trained_sasrec_definition', os.path.join(_HERE, '..', 'train.py'))
            _MODULE = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_MODULE)
        checkpoint = torch.load(
            os.path.join(_HERE, dataset_name + '_model.pt'),
            map_location=_DEVICE, weights_only=True)
        model = _MODULE.SASRec(**checkpoint['config']).to(_DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        _MODELS[dataset_name] = model
    return _MODELS[dataset_name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    model = _get_model(dataset_name if dataset_name is not None else _DATASETS[0])
    if len(candidates) == 0:
        return []
    seq_array = _MODULE.pack_histories([list(history)], model.maxlen)
    seq = torch.as_tensor(seq_array, dtype=torch.long, device=_DEVICE)
    cand = torch.as_tensor([list(candidates)], dtype=torch.long, device=_DEVICE)
    return model.predict(seq, cand)[0].float().cpu().tolist()
