from __future__ import annotations
import importlib.util
import os
import pickle
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAMES = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_METAS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_MODULE = None


def _load(dataset):
    global _MODULE
    if dataset in _MODELS:
        return
    if _MODULE is None:
        spec = importlib.util.spec_from_file_location(
            'relative_maskedrec_trial_model', os.path.join(_HERE, '..', 'train.py')
        )
        _MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MODULE)
    with open(os.path.join(_HERE, dataset + '_meta.pkl'), 'rb') as f:
        config = pickle.load(f)
    model = _MODULE.MaskedRec(**config).to(_DEVICE)
    state = torch.load(
        os.path.join(_HERE, dataset + '_model.pt'),
        map_location=_DEVICE, weights_only=True,
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    _MODELS[dataset] = model
    _METAS[dataset] = config


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    dataset = _NAMES[0] if dataset_name is None else dataset_name
    if dataset not in _NAMES:
        raise ValueError('Unknown dataset: ' + str(dataset))
    _load(dataset)
    config = _METAS[dataset]
    seq = _MODULE.inference_sequence(history, config['itemnum'], config['maxlen'])
    sequence = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    candidate_tensor = torch.tensor([list(candidates)], dtype=torch.long, device=_DEVICE)
    return _MODELS[dataset].predict(sequence, candidate_tensor)[0].float().cpu().tolist()
