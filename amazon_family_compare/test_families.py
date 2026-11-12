import importlib.util
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .prepare import WORKSPACE, DATASETS
from .diagnostics import ModelView

ROOT = WORKSPACE/'artifacts/model-family-comparison-20261029'
NAMES = {'gru4rec': 'GRU4RecModel', 'nextitnet': 'NextItNetModel',
         'bert4rec': 'BERT4RecModel', 'hstu': 'HSTURec'}
torch.set_num_threads(2)


def load(path):
    spec = importlib.util.spec_from_file_location('probe_'+path.parent.name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model(family, module):
    kwargs = dict(itemnum=30, hidden_units=8, dropout_rate=0.)
    kwargs.update(num_layers=1) if family == 'gru4rec' else kwargs.update(num_blocks=2, maxlen=8)
    return getattr(module, NAMES[family])(**kwargs).eval()


@pytest.mark.parametrize('family', list(NAMES))
def test_short_history_forward_backward_and_diagnosis(family):
    module = load(ROOT/'baselines'/family/'train.py')
    m = model(family, module)
    seq, pos, neg = module.sample_batch({1: [1, 2, 3, 4], 2: [5, 6, 7]}, 30, 4, 8)
    seq, pos, neg = [torch.as_tensor(x, dtype=torch.long) for x in [seq, pos, neg]]
    if family == 'hstu':
        out = m(seq)
        positive, negative = (out*m.item_emb(pos)).sum(-1), (out*m.item_emb(neg)).sum(-1)
    else:
        positive, negative = m(seq, pos, neg)
    loss = torch.nn.functional.softplus(negative[pos != 0]-positive[pos != 0]).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    view = ModelView(m, {1:[1,2,3]}, {'time':{},'title':{},'description':{}}, torch.device('cpu'))
    features = view.log2feats(seq)
    margins = view.predict_score(features, torch.stack([pos, neg], dim=-1))
    torch.testing.assert_close(margins[...,0], positive)
    torch.testing.assert_close(margins[...,1], negative)


@pytest.mark.parametrize('family', ['gru4rec', 'nextitnet'])
def test_readout_is_last_nonpadding_not_count_minus_one(family):
    module = load(ROOT/'baselines'/family/'train.py')
    m = model(family, module)
    seq = torch.tensor([[0,0,0,0,0,1,2,3]])
    if family == 'gru4rec':
        out = m.gru(m.dropout(m.item_emb(seq)))[0]
        torch.testing.assert_close(m.encode(seq), out[:,7])
        assert not torch.allclose(m.encode(seq), out[:,2])
    else:
        out = m.encode_sequence(seq)
        expected = m.item_emb(torch.tensor([4,5])) @ out[0,7]
        torch.testing.assert_close(m.score_candidates(seq, torch.tensor([4,5])), expected)


def test_bert_masked_target_is_absent_at_supervised_positions_and_prediction_appends_mask():
    module = load(ROOT/'baselines/bert4rec/train.py')
    seq, pos, neg = module.sample_batch({1:[1,2,3,4]},30,32,8)
    assert np.all(seq[pos!=0] == 31) and np.all(pos[pos!=0] <= 30)
    assert np.all(neg[pos!=0] > 4)
    m = model('bert4rec', module)
    original = m.encode_sequence
    seen = []
    def encode(x):
        seen.append(x.clone())
        return original(x)
    m.encode_sequence = encode
    m.score_candidates(torch.tensor([[0,0,0,0,0,1,2,3]]),torch.tensor([4,5]))
    assert seen[-1].tolist() == [[0,0,0,0,1,2,3,31]]


@pytest.mark.parametrize('family', list(NAMES))
def test_generated_predictor_matches_checkpoint_model(tmp_path, monkeypatch, family):
    source = ROOT/'baselines'/family/'train.py'
    (tmp_path/'train.py').write_bytes(source.read_bytes())
    working = tmp_path/'working'
    working.mkdir()
    module = load(tmp_path/'train.py')
    m = model(family,module)
    meta = {'itemnum':30,'hidden_units':8,'dropout_rate':0.,'maxlen':8,
            'num_layers':1,'num_blocks':2,'num_heads':2,'kernel_size':3}
    for ds in DATASETS:
        torch.save(m.state_dict(),working/f'{ds}_model.pt')
        (working/f'{ds}_meta.pkl').write_bytes(pickle.dumps(meta))
    module.write_predict_script(str(working))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda:False)
    predictor = load(working/'predict.py')
    values = predictor.predict(1,[1,2,3],[4,5],dataset_name=DATASETS[0])
    seq = torch.tensor([[0,0,0,0,0,1,2,3]])
    with torch.no_grad():
        expected = (m.item_emb(torch.tensor([4,5]))@m(seq)[0,-1] if family=='hstu' else
                    m.score_candidates(seq,torch.tensor([4,5])))
    np.testing.assert_allclose(values,expected.numpy(),rtol=1e-5,atol=1e-6)
    from .single_model import model_audit
    data = tmp_path/'data/trainval'
    data.mkdir(parents=True)
    for ds in DATASETS:
        (data/f'{ds}_train.txt').write_text('1 1\n1 2\n1 3\n')
    monkeypatch.setenv('GAGC_MODEL_FAMILY', family)
    assert model_audit(tmp_path,data.parent)['eligible'] is True
    monkeypatch.setenv('GAGC_MODEL_FAMILY', 'bert4rec' if family!='bert4rec' else 'gru4rec')
    audit = model_audit(tmp_path,data.parent)
    assert audit['eligible'] is False and 'escaped registered model family' in audit['reason']


def test_eight_jobs_have_matched_budget_model_source_and_llm():
    plan = json.loads((ROOT/'plan.json').read_text())
    assert len(plan['jobs']) == 8 and plan['total_gpu_seconds'] == 345600
    assert {(j['family'],j['method']) for j in plan['jobs']} == {(f,m) for f in NAMES for m in ['mlestar','self_evolverec']}
    for family in NAMES:
        configs = [json.loads(Path(j['config']).read_text()) for j in plan['jobs'] if j['family']==family]
        assert len({c['baseline']['sha256'] for c in configs}) == 1
        assert all(c['search_budget_gpu_seconds']==43200 and c['single_model_search'] for c in configs)
        assert all(c['llm_model']=='gpt-6.1-sol' and c['llm_effort']=='medium' for c in configs)
        assert all(len(c['gpus'])==4 and len(c['cpu_cores'])==64 for c in configs)
