import sys
from pathlib import Path

import pytest

from recevobench.taac.common import file_hash, read, write
from recevobench.taac.engine import FeatureBank, construct, exact_topk, load_candidate
from recevobench.taac.loop import (
    baseline,
    context,
    finish,
    initialize,
    load,
    recover,
    run,
    stage_memory,
    trial,
)

torch = pytest.importorskip("torch")
pytest.importorskip("numpy")


def test_chunked_exact_retrieval_matches_dense_and_ties():
    torch.manual_seed(5)
    q, c = torch.randn(3, 8), torch.randn(19, 8)
    expected = torch.argsort(q @ c.T, dim=1, descending=True, stable=True)[:, :10]
    actual, _ = exact_topk(q, c, chunk_size=3)
    assert torch.equal(actual, expected)
    tied, _ = exact_topk(torch.zeros(2, 8), c, chunk_size=2)
    assert tied.tolist() == [list(range(10)), list(range(10))]
    c[0, 0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        exact_topk(q, c)


def test_upstream_causality_readout_and_loss_normalization(prepared, study_config):
    candidate = load_candidate(read(study_config)["candidate"])
    bank = FeatureBank(prepared / "public")
    model = construct(candidate, bank, "cpu").eval()
    e = {
        "user_id": "1",
        "history": [{"item_id": i, "action_type": 0, "timestamp": i} for i in [1, 2, 3]],
    }
    seq, mask, feat = bank.encode_inputs([e], "cpu")
    assert feat[0][-1]["history_timestamp"] == 3
    assert feat[0][-1]["history_action"] == 0
    with torch.no_grad():
        original = model.log2feats(seq, mask, feat)
        assert torch.equal(model.predict(seq, feat, mask), original[:, -1])
        seq[0, -1] = 4
        feat[0][-1] = bank.item(4)
        changed = model.log2feats(seq, mask, feat)
        torch.testing.assert_close(original[:, :-1], changed[:, :-1])
    pos, neg = torch.tensor([1.0, 2.0]), torch.tensor([[-1.0], [-2.0]])
    torch.testing.assert_close(
        candidate.compute_loss(pos, neg), candidate.compute_loss(pos, neg.repeat(1, 4))
    )
    bank.close()


AGENT = """import ast,json,sys
from pathlib import Path
request=json.loads(Path(sys.argv[1]).read_text())
ctx=request['context']
if request['phase']=='query':
    refs=['model.lr'] + [k for k in ctx['model_state']['facts'] if k.startswith('diagnostic.')]
    refs += ['trial:'+t['id'] for t in ctx['history']]
    text='negative sampling failure lr '+str(ctx['model_state']['facts']['model.lr'])
    text+=' history '+str(len(ctx['history']))
    Path('retrieval-plan.json').write_text(json.dumps({'queries':[{
       'text':text, 'intent':'counterevidence', 'keywords':['negative','sampling'],
       'conditions':[], 'rationale':'scripted integration query from current evidence',
       'evidence_refs':refs}]}))
elif request['phase']=='plan':
    p=Path('candidate.py')
    tree=ast.parse(p.read_text())
    for n in tree.body:
        if (isinstance(n,ast.Assign) and
            any(isinstance(t,ast.Name) and t.id=='SETTINGS' for t in n.targets)):
            settings=ast.literal_eval(n.value)
            settings['lr']=0.001/(len(ctx['history'])+1)
            n.value=ast.parse(repr(settings)).body[0].value
    p.write_text(ast.unparse(tree)+'\\n')
    d={'hypothesis':'Scripted test LR change; not an LLM research result',
       'change':'change learning rate only','falsification':'feedback score does not improve',
       'controls':['same source parent, data, seeds and steps'], 'budget_rationale':'two CPU steps'}
    if ctx['knowledge']:
        def judgment():
            return {'status':'unknown','rationale':'test only; needs evidence','evidence_refs':[]}
        d['knowledge_assessments']=[{
          'knowledge_id':card['id'], **judgment(), 'condition_checks':card['condition_checks'],
          'counterexample_checks':[{'counterexample':v,**judgment()}
                                   for v in card.get('counterexamples',[])],
          'linked_counterexample_checks':[{'knowledge_id':v,**judgment()}
                                          for v in card.get('counterexample_ids',[])]
        } for card in ctx['knowledge']]
        c=ctx['knowledge'][0]
        d['diffs']=[{'knowledge_id':c['id'],'current_refs':['model.lr'],
                    'condition_checks':c['condition_checks'], 'alternatives':['random noise'],
                    'missing_evidence':['replication'], 'experiment':'LR change',
                    'priority_reason':'integration test only',
                    'exploration_rationale':'test uncertainty', 'selected':True}]
    Path('design.json').write_text(json.dumps(d))
else:
    Path('analysis.json').write_text(json.dumps({'effect':'scripted test analysis',
       'alternatives':['random noise'], 'next_action':'inspect preceding result',
       'evidence_trials':[ctx['latest_trial']['id']]}))
"""


@pytest.mark.parametrize("enhanced", [False, True])
def test_real_cpu_training_two_rounds_and_frozen_final(tmp_path, study_config, enhanced, request):
    config = read(study_config)
    agent = tmp_path / "scripted_agent.py"
    agent.write_text(AGENT)
    if enhanced:
        config["retrieval"]["encoder"]["model_path"] = str(request.getfixturevalue("tiny_encoder"))
        config["min_delta"] = 1.0  # Force discard to test negative-history propagation.
    config.update(enhanced=enhanced, agent={"argv": [sys.executable, str(agent), "{request}"]})
    write(study_config, config)
    root = tmp_path / "study"
    initialize(study_config, root)
    state = run(root, rounds=2)
    assert len(state["trials"]) == 3
    assert all(t["status"] in {"keep", "discard"} for t in state["trials"])
    assert state["trials"][0]["status"] == "keep"
    for t in state["trials"]:
        d = root / "trials" / t["id"]
        assert read(d / "seed-42/train/training.json")["steps"] == 2
        assert (d / "seed-42/train/checkpoint.pt").exists()
        metric = read(d / "seed-42/metrics.json")
        assert metric["candidate_count"] == 16 and metric["queries"] == 30
        assert t["score"] == metric["score"]
        assert t["wall_seconds"] > 0
    ctx = context(root)
    assert not any("target" in k for k in ctx)
    if enhanced:
        assert all(
            (root / "trials" / t["id"] / "diagnostics.json").exists() for t in state["trials"]
        )
        requests = [read(p) for p in sorted((root / "agents").glob("*/request.json"))]
        assert [r["phase"] for r in requests] == ["query", "plan", "analyze"] * 2
        assert not requests[0]["context"]["knowledge"]
        assert len(requests[3]["context"]["history"]) == 2
        assert requests[3]["context"]["history"][-1]["status"] == "discard"
        retrieval_requests = [read(p) for p in sorted((root / "retrieval").glob("*/request.json"))]
        failed_card = next(c for c in retrieval_requests[1]["cards"] if c["id"] == "trial-t0001")
        assert failed_card["previous_status"] == "discard"
        assert failed_card["reason"]["analysis"]["next_action"]
        assert "analysis" in requests[3]["context"]["history"][-1]
        first = requests[1]["context"]["retrieval"]["plan"]
        second = requests[4]["context"]["retrieval"]["plan"]
        assert first != second
        assert requests[1]["context"]["knowledge"] == requests[2]["context"]["knowledge"]
        assert len(list((root / "retrieval").glob("*/ranking.json"))) == 2
        assert state["spent_seconds"] > sum(t["wall_seconds"] for t in state["trials"])
        assert ctx["knowledge"] == []  # Previous retrieval becomes stale after the trial.
        assert "analysis" in ctx["history"][-1]
        assert ctx["history"][-1]["analysis"]["next_action"]
    final = finish(root)
    assert state["incumbent"] in final
    assert set(final[state["incumbent"]]) == {"gate", "final"}
    with pytest.raises(ValueError, match="closed"):
        run(root)
    memory = stage_memory(root)
    assert len(memory) == 3
    assert (root / "results.tsv").read_text().count("\n") == 4


def test_trial_failure_retains_cost_and_incumbent(tmp_path, study_config):
    root = tmp_path / "study"
    initialize(study_config, root)
    initial = baseline(root)
    assert initial["status"] == "keep"
    bad = tmp_path / "bad.py"
    bad.write_text(
        Path(read(study_config)["candidate"]).read_text()
        + '\nraise RuntimeError("intentional failure fixture")\n'
    )
    design = tmp_path / "design.json"
    write(
        design,
        {
            "hypothesis": "failure accounting",
            "change": "intentional crash",
            "falsification": "crash",
            "controls": ["baseline"],
            "budget_rationale": "test",
        },
    )
    result = trial(root, bad, design)
    assert result["status"] == "crash" and result["score"] is None
    state = load(root)[2]
    assert state["incumbent"] == initial["id"]
    assert state["spent_seconds"] >= sum(t["wall_seconds"] for t in state["trials"])
    assert not state["pending"]


def test_recovery_charges_reservation_and_is_idempotent(tmp_path, study_config):
    root = tmp_path / "study"
    initialize(study_config, root)
    state = read(root / "state.json")
    state["pending"] = {
        "kind": "trial",
        "id": "t0000",
        "reserved_seconds": 12.5,
        "parent": None,
        "hypothesis": "interrupted test",
        "source_sha256": file_hash(root / "baseline.py"),
    }
    write(root / "state.json", state)
    with pytest.raises(ValueError, match="Unsettled"):
        run(root)
    recovered = recover(root)
    assert recovered["spent_seconds"] == 12.5
    assert recovered["trials"][0]["status"] == "interrupted"
    assert recover(root) == recovered


def test_frozen_protocol_and_source_mutation_rejected(tmp_path, study_config):
    root = tmp_path / "study"
    initialize(study_config, root)
    (root / "baseline.py").write_text("# changed")
    with pytest.raises(ValueError, match="Baseline source changed"):
        load(root)


def test_frozen_knowledge_cannot_be_overwritten(tmp_path, study_config, tiny_encoder):
    config = read(study_config)
    config["enhanced"] = True
    config["retrieval"]["encoder"]["model_path"] = str(tiny_encoder)
    write(study_config, config)
    root = tmp_path / "study"
    initialize(study_config, root)
    (root / "knowledge.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="read-only input changed"):
        context(root)


def test_runtime_settings_cannot_silently_disagree_with_diff(tmp_path, study_config):
    source = tmp_path / "candidate.py"
    source.write_text(
        Path(read(study_config)["candidate"]).read_text() + '\nSETTINGS["negatives"] = 99\n'
    )
    with pytest.raises(ValueError, match="runtime settings differ"):
        load_candidate(source)


def test_live_worker_prevents_recovery_and_timeout_is_accounted(tmp_path, study_config):
    import os
    import subprocess

    from recevobench.taac.common import bounded_run

    root = tmp_path / "study"
    initialize(study_config, root)
    work = root / "agents/0000-plan"
    work.mkdir()
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True
    )
    try:
        write(work / "agent.launch.json", {"pgid": worker.pid})
        state = read(root / "state.json")
        state["pending"] = {"kind": "agent", "id": work.name, "reserved_seconds": 2.0}
        write(root / "state.json", state)
        with pytest.raises(ValueError, match="process group still exists"):
            recover(root)
    finally:
        worker.kill()
        worker.wait()
    assert recover(root)["spent_seconds"] == 2.0
    result = bounded_run(
        [sys.executable, "-c", "import time; time.sleep(20)"],
        tmp_path,
        tmp_path / "timeout.log",
        0.1,
    )
    assert result["status"] == "timeout" and result["wall_seconds"] >= 0.1
    launch = read(tmp_path / "timeout.launch.json")
    with pytest.raises(ProcessLookupError):
        os.killpg(launch["pgid"], 0)
