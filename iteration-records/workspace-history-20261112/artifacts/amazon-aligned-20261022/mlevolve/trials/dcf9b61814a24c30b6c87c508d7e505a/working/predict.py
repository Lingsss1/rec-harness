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
        _CLASS = module.AttentiveSimilarity
    checkpoint = torch.load(os.path.join(_HERE, name + '_model.pt'),
                            map_location='cpu', weights_only=True)
    # Load the revised training class and require its learned recency parameter.
    if 'recency_bias' not in checkpoint['state_dict']:
        raise ValueError('Checkpoint is missing recency_bias: ' + str(name))
    model = _CLASS(**checkpoint['config']).to(_DEVICE)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
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
    outputs = []
    for start in range(0, len(candidates), 256):
        candidate_tensor = torch.tensor([candidates[start:start + 256]],
                                        dtype=torch.long, device=_DEVICE)
        outputs.extend(model.predict(sequence, candidate_tensor)[0].cpu().tolist())
    return outputs
