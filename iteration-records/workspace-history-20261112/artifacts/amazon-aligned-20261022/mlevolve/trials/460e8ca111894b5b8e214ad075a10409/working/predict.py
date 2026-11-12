from __future__ import annotations
import importlib.util
import os
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
             'Electronics', 'CDs_and_Vinyl']
_models = {}
_module = None
_device = None


def _load(dataset_name):
    global _module, _device
    if _module is None:
        spec = importlib.util.spec_from_file_location(
            'trained_multvae_module', os.path.join(_HERE, '..', 'train.py'))
        _module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_module)
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(4)
    if dataset_name not in _models:
        saved = torch.load(os.path.join(_HERE, dataset_name + '_model.pt'),
                           map_location='cpu', weights_only=True)
        # Reconstruct the saved residual architecture; older checkpoints default
        # to the original architecture. History preprocessing remains identical.
        config = dict(saved['config'])
        config.setdefault('residual', False)
        model = _module.MultVAE(**config)
        model.load_state_dict(saved['state_dict'])
        sequential = torch.load(os.path.join(_HERE, dataset_name + '_sasrec.pt'),
                                map_location='cpu', weights_only=True)
        # Use the saved transition architecture and its shared predict path.
        sequential_config = dict(sequential['config'])
        sequential_config.setdefault('transition_dim', 32)
        sasrec = _module.SASRec(**sequential_config)
        sasrec.load_state_dict(sequential['state_dict'], strict=True)
        _models[dataset_name] = (model.to(_device).eval(),
                                 sasrec.to(_device).eval())
    return _models[dataset_name]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    ds = dataset_name if dataset_name is not None else _DATASETS[0]
    if ds not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(ds))
    if len(candidates) == 0:
        return []
    model, sasrec = _load(ds)
    # Identical binary, untruncated history representation to validation.
    row = np.asarray(sorted(set(int(i) for i in history)), dtype=np.int64)
    if row.size and (row.min() < 1 or row.max() > model.itemnum):
        raise ValueError('History item outside the trained catalog')
    ids, offsets, _ = _module.pack_rows([row])
    candidate_tensor = torch.tensor([candidates], dtype=torch.long, device=_device)
    if candidate_tensor.min() < 1 or candidate_tensor.max() > model.itemnum:
        raise ValueError('Candidate item outside the trained catalog')
    vae_scores = model.predict(ids.to(_device), offsets.to(_device), candidate_tensor)
    sequence = _module.chronological_rows([history], sasrec.maxlen, _device)
    sequential_scores = sasrec.predict(sequence, candidate_tensor)
    scores = _module.rank_fusion(vae_scores, sequential_scores)
    return scores[0].cpu().tolist()
