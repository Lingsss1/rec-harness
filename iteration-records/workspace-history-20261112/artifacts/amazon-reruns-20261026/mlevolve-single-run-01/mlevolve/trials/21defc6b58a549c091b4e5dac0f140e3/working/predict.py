from __future__ import annotations
import importlib.util
import os
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
             'Electronics', 'CDs_and_Vinyl']
_MODELS = {}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_MODEL_CLASS = None


def _get_model(dataset):
    global _MODEL_CLASS
    if dataset not in _DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if dataset not in _MODELS:
        if _MODEL_CLASS is None:
            spec = importlib.util.spec_from_file_location(
                'trained_item_attention', os.path.join(_HERE, '..', 'train.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            _MODEL_CLASS = module.ItemAttention
        state = torch.load(os.path.join(_HERE, dataset + '_model.pt'),
                           map_location=_DEVICE, weights_only=True)
        if state.get('architecture') != 'tied_item_attention_v1':
            raise ValueError('Expected a tied ItemAttention checkpoint')
        model = _MODEL_CLASS(**state['config']).to(_DEVICE)
        # Reconstruct exactly one tied network from its saved configuration.
        model.load_state_dict(state['state_dict'], strict=True)
        assert model.history_embedding is model.target_embedding
        assert model.history_embedding.padding_idx == 0
        assert torch.count_nonzero(model.history_embedding.weight[0]).item() == 0
        model.eval()
        _MODELS[dataset] = model
    return _MODELS[dataset]


@torch.inference_mode()
def predict(user_id, history, candidates, *, dataset_name=None):
    dataset = dataset_name if dataset_name is not None else _DATASETS[0]
    model = _get_model(dataset)
    hist = list(history[-model.maxlen:])
    seq = [0] * (model.maxlen - len(hist)) + hist
    seq_tensor = torch.tensor([seq], dtype=torch.long, device=_DEVICE)
    candidate_tensor = torch.tensor([candidates], dtype=torch.long, device=_DEVICE)
    return model.predict(seq_tensor, candidate_tensor)[0].float().cpu().tolist()
