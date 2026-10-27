from __future__ import annotations
import importlib.util
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific',
            'Electronics', 'CDs_and_Vinyl']
MODELS = {}
DEVICE = None
MODEL_CLASS = None
PREPARE_SEQUENCE = None


def load_model(dataset):
    global DEVICE, MODEL_CLASS, PREPARE_SEQUENCE
    if dataset in MODELS:
        return MODELS[dataset]
    if dataset not in DATASETS:
        raise ValueError('Unknown dataset: ' + str(dataset))
    if MODEL_CLASS is None:
        spec = importlib.util.spec_from_file_location(
            'convolutional_rec_training', os.path.join(HERE, '..', 'train.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        MODEL_CLASS = module.ConvRec
        PREPARE_SEQUENCE = module.prepare_sequence
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(4)
    checkpoint = torch.load(os.path.join(HERE, dataset + '_model.pt'),
                            map_location='cpu', weights_only=True)
    model = MODEL_CLASS(**checkpoint['config']).to(DEVICE)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    MODELS[dataset] = model
    return model


def predict(user_id, history, candidates, *, dataset_name=None):
    dataset = DATASETS[0] if dataset_name is None else dataset_name
    model = load_model(dataset)
    if not candidates:
        return []
    seq = torch.from_numpy(PREPARE_SEQUENCE(history, model.maxlen))
    seq = seq.unsqueeze(0).to(DEVICE)
    candidates = torch.tensor([candidates], device=DEVICE, dtype=torch.long)
    with torch.inference_mode():
        return model.predict(seq, candidates)[0].float().cpu().tolist()
