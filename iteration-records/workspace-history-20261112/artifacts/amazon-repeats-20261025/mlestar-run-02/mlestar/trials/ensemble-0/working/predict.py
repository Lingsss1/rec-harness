import importlib.util
import numpy as np
_models = []
for i, path in enumerate(['/home/liuzhishan/lhr/rec-harness/artifacts/amazon-repeats-20261025/mlestar-run-02/mlestar/trials/refine-0-0-t0/working/predict.py', '/home/liuzhishan/lhr/rec-harness/artifacts/amazon-repeats-20261025/mlestar-run-02/mlestar/trials/refine-0-0-t1/working/predict.py', '/home/liuzhishan/lhr/rec-harness/artifacts/amazon-repeats-20261025/mlestar-run-02/mlestar/trials/refine-0-1-t2/working/predict.py', '/home/liuzhishan/lhr/rec-harness/artifacts/amazon-repeats-20261025/mlestar-run-02/mlestar/trials/merge-t3/working/predict.py']):
    spec = importlib.util.spec_from_file_location('ensemble_member_' + str(i), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _models.append(module.predict)
def predict(user_id, history, candidates, *, dataset_name=None):
    values = np.asarray([fn(user_id, history, candidates, dataset_name=dataset_name) for fn in _models], dtype=float)
    if 'standardize' == 'standardize':
        values = (values - values.mean(axis=1, keepdims=True)) / (values.std(axis=1, keepdims=True) + 1e-8)
    elif 'standardize' == 'rank':
        values = values.argsort(axis=1).argsort(axis=1).astype(float)
    return np.average(values, axis=0, weights=[0.4, 0.15, 0, 0.45]).tolist()
