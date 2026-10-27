from pathlib import Path
import importlib.util
import torch

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('rec_ensemble_training', _HERE.parent / 'train.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_loaded_name = None
_models = None
_meta = None

def predict(user_id, history, candidates, *, dataset_name=None):
    global _loaded_name, _models, _meta
    name = dataset_name if dataset_name is not None else _module.DATASETS[0]
    if name not in _module.DATASETS:
        raise ValueError(f'Unknown dataset: {name}')
    if _loaded_name != name:
        _models = None
        if _device.type == 'cuda':
            torch.cuda.empty_cache()
        _models, _meta = _module.load_ensemble(name, _HERE, _device)
        _loaded_name = name
    with torch.inference_mode():
        members = _module.score_members(_models, list(history), list(candidates),
                                        _meta['sas']['nitems'],
                                        _meta['sas']['maxlen'], _device)
        return _module.fused_scores(members, candidates).tolist()
