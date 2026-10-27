from __future__ import annotations
import importlib.util
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_NAMES = ['Movies_and_TV', 'Industrial_and_Scientific',
          'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_CLASS = None


def _load(name):
    global _CLASS
    if name not in _NAMES:
        raise ValueError('Unknown dataset: ' + str(name))
    if name in _MODELS:
        return _MODELS[name]
    if _CLASS is None:
        spec = importlib.util.spec_from_file_location(
            'attentive_similarity_train', os.path.join(_HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _CLASS = module
    experts = []
    for suffix, expert_class in (('model', _CLASS.AttentiveSimilarity),
                                ('sasrec', _CLASS.SASRec)):
        checkpoint = torch.load(os.path.join(_HERE, name + '_' + suffix + '.pt'),
                                map_location='cpu', weights_only=True)
        expert = expert_class(**checkpoint['config']).to(_DEVICE)
        expert.load_state_dict(checkpoint['state_dict'], strict=True)
        experts.append(expert.eval())
    model = _CLASS.RankEnsemble(*experts).eval()
    _MODELS[name] = model
    return model


@torch.inference_mode()
def predict(user_id: int, history: list[int], candidates: list[int],
            *, dataset_name: str | None = None) -> list[float]:
    name = dataset_name if dataset_name is not None else _NAMES[0]
    model = _load(name)
    if not candidates:
        return []
    hist = list(history)[-model.maxlen:]
    seq = [0] * (model.maxlen - len(hist)) + hist
    sequence = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    candidate_tensor = torch.tensor([candidates], dtype=torch.long, device=_DEVICE)
    scores = model.predict(sequence, candidate_tensor)[0]
    if not torch.isfinite(scores).all():
        raise FloatingPointError('Nonfinite prediction scores')
    return scores.cpu().tolist()
