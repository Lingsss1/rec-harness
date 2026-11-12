import importlib.util
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('_residual_bridge_train', os.path.join(_HERE, '..', 'train.py'))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_predictor = _mod.Predictor(_HERE)
def predict(user_id, history, candidates, *, dataset_name=None):
    return _predictor.predict(user_id, history, candidates, dataset_name=dataset_name)
