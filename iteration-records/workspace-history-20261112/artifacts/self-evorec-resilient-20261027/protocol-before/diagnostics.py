"""Native Self-EvolveRec diagnostic probes on the shared trained checkpoints."""
from collections import Counter, defaultdict
import importlib.util
import json
from pathlib import Path
import pickle
import random
from types import SimpleNamespace

import numpy as np
import torch

from .core import save
from .prepare import DATASETS, REPO


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def histories(path):
    result = defaultdict(list)
    with path.open() as stream:
        for line in stream:
            user, item = map(int, line.split())
            result[user].append(item)
    return dict(result)


class ModelView:
    """Map the baseline's tensor interfaces to the upstream diagnostic contract."""
    def __init__(self, model, train, metadata, device):
        self.model, self.device = model, device
        self.user_train = {'History': {str(u): v for u, v in train.items()},
                           'Time': {str(u): [metadata.get('time', {}).get(i, {}).get(u, 0) or 0 for i in v]
                                    for u, v in train.items()}}
        self.meta_data = {str(i): {'title': str(title), 'description': metadata['description'].get(i, ''),
                                  'categories': [], 'average_rating': 'unavailable',
                                  'price': 'unavailable', 'store': 'unavailable'}
                          for i, title in metadata['title'].items()}
        self.review_data = {}

    @property
    def item_emb(self):
        return self.model.item_emb

    def eval(self):
        return self.model.eval()

    def log2feats(self, sequence):
        return self.model.log2feats(torch.as_tensor(sequence, dtype=torch.long, device=self.device))

    def predict_score(self, features, items):
        if hasattr(self.model, 'predict_score'):
            return self.model.predict_score(features, items)
        return (features.unsqueeze(-2) * self.item_emb(items)).sum(-1)


def collect(trial, data):
    predictor = load_module(trial / 'working/predict.py', '_self_evolve_predict')
    predictor._ensure_loaded()
    if not all(name in predictor._models for name in DATASETS):
        raise RuntimeError('Diagnostic adapter requires the per-dataset trained model registry')
    diagnosis = load_module(trial / 'diagnosis_tools.py', '_evolved_diagnosis')
    baseline = load_module(REPO / 'gagc/templates/perdataset/train.py', '_fixed_sampler')
    diagnoses, simulations = {}, []
    rng = random.Random(42)
    for dataset in DATASETS:
        train = histories(data / 'trainval' / (dataset + '_train.txt'))
        with (data / 'trainval' / (dataset + '_text_name_dict.json.gz')).open('rb') as stream:
            metadata = pickle.load(stream)
        model, meta = predictor._models[dataset], predictor._metas[dataset]
        view = ModelView(model, train, metadata, next(model.parameters()).device)
        def batches():
            for _ in range(10):
                seq, pos, neg = baseline.sample_batch(train, meta['itemnum'], 128, meta['maxlen'])
                yield None, torch.tensor(seq), torch.tensor(pos), torch.tensor(neg)
        probe = diagnosis.DiagnosisProbe(view, view.device)
        diagnoses[dataset] = probe.run_full_diagnosis(batches())
        if not {'metrics', 'metric_definitions'} <= set(diagnoses[dataset]):
            raise ValueError('Evolved diagnosis must preserve metrics and metric_definitions')
        selected = rng.sample(sorted(train), min(5, len(train)))
        pairs, used_items = [], set()
        for user in selected:
            hist = train[user]
            candidates = sorted(set(range(1, meta['itemnum'] + 1)) - set(hist))
            with torch.no_grad():
                scores = np.asarray(predictor.predict(user, hist, candidates, dataset_name=dataset))
            recommendations = [candidates[i] for i in np.argsort(-scores, kind='stable')[:20]]
            if len(recommendations) < 10:
                raise RuntimeError('Insufficient user-simulation recommendations')
            pairs.append({'user': str(user), 'recommendations': recommendations})
            used_items.update(hist)
            used_items.update(recommendations)
        counts = Counter(i for items in train.values() for i in items)
        total = sum(counts.values())
        subset_hist = {str(u): train[u] for u in selected}
        simulations.append({'dataset': dataset, 'pairs': pairs,
            'user_train': {'History': subset_hist, 'Time': {str(u): view.user_train['Time'][str(u)] for u in selected}},
            'meta_data': {str(i): view.meta_data.get(str(i), {'title': 'unavailable', 'categories': [],
                          'average_rating': 'unavailable', 'price': 'unavailable', 'store': 'unavailable'})
                          for i in used_items}, 'review_data': {},
            'popularity': {str(i): max(counts[i] / total, 0.0001) for i in used_items},
            'activity_thresholds': np.percentile([len(v) for v in train.values()], [60, 90]).tolist()})
    save(trial / 'diagnosis.json', diagnoses)
    save(trial / 'simulation-input.json', simulations)
    print('NATIVE_DIAGNOSTICS_READY', flush=True)
