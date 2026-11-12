import copy
import sys

import pytest

from recevobench.taac.common import append, read, write
from recevobench.taac.knowledge import validate_assessments
from recevobench.taac.loop import context, initialize, load, recover, research_context
from recevobench.taac.retrieval import (
    E5Encoder,
    HybridRetriever,
    bm25,
    freeze_config,
    load_cards,
    materialize,
    validate_plan,
)


def card(cid, method, **extra):
    return dict(
        id=cid,
        scene="ranking",
        method=method,
        gain=None,
        reason="unverified hypothesis",
        sources=[],
        conditions=[],
        counterexamples=[],
        **extra,
    )


def plan(text="negative sampling"):
    return {
        "queries": [
            {
                "text": text,
                "intent": "counterevidence",
                "keywords": [],
                "conditions": [],
                "rationale": "resolve a failure",
                "evidence_refs": ["model.lr", "diagnostic.mean_target_rank", "trial:t0001"],
            }
        ]
    }


def ctx():
    return {
        "model_state": {"facts": {"model.lr": 0.01, "diagnostic.mean_target_rank": 400}},
        "history": [{"id": "t0001", "status": "discard"}],
        "knowledge": [],
    }


def test_dynamic_plan_provenance_and_counterevidence():
    value = plan()
    validate_plan(value, ctx())
    value["queries"][0]["evidence_refs"].append("trial:invented")
    with pytest.raises(ValueError, match="resolve"):
        validate_plan(value, ctx())
    for ref, error in [
        ("model.lr", "model facts"),
        ("trial:t0001", "historical"),
        ("diagnostic.mean_target_rank", "diagnostics"),
    ]:
        value = plan()
        value["queries"][0]["evidence_refs"].remove(ref)
        with pytest.raises(ValueError, match=error):
            validate_plan(value, ctx())
    value = plan()
    value["queries"][0]["intent"] = "method"
    with pytest.raises(ValueError, match="counterevidence"):
        validate_plan(value, ctx())


class DirectionEncoder:
    def encode(self, texts, *, query):
        if query:
            return [[1, 0] if "failure" in t else [0, 1] for t in texts]
        return [[1, 0], [0, 1], [-1, 0]]


def test_hybrid_query_changes_ranking_and_keeps_lane_scores():
    cards = [card("a", "alternative"), card("b", "alternative"), card("c", "alternative")]
    retriever = HybridRetriever(DirectionEncoder())
    first = retriever.rank(cards, plan("failure"))
    second = retriever.rank(cards, plan("coverage"))
    assert first[0]["id"] == "a" and second[0]["id"] == "b"
    lexical = plan("failure")
    lexical["queries"][0]["keywords"] = ["specialterm"]
    cards[1]["method"] = "specialterm"
    fused = retriever.rank(cards, lexical)
    assert fused[0]["id"] == "b"  # Lexical lane actually affects fusion.
    assert {t["lane"] for t in fused[0]["trace"]} == {"semantic", "bm25"}
    cards[2]["conditions"] = [{"field": "model.lr", "op": "gt", "value": 0.1}]
    lexical["queries"][0]["conditions"] = cards[2]["conditions"]
    fused = retriever.rank(cards, lexical)
    assert any(t["lane"] == "conditions" for r in fused for t in r["trace"])
    both = {"queries": plan("failure")["queries"] + plan("coverage")["queries"]}
    result = retriever.rank([card(i, "same") for i in "abc"], both)
    assert [r["id"] for r in result[:2]] == ["a", "b"]  # Stable ID tie break.


def test_chinese_bm25_exact_terms():
    scores = bm25(["排查负采样中的假负例", "检查多模态覆盖率"], "假负例 负采样")
    assert scores[0] > scores[1] == 0


def test_linked_counterexamples_survive_topk_cycles_and_unknowns(tmp_path):
    a = card("a", "method", counterexample_ids=["b"])
    b = card("b", "negative evidence", counterexample_ids=["a", "c"])
    c = card("c", "different mechanism")
    a["conditions"] = [{"field": "absent", "op": "eq", "value": False}]
    c["conditions"] = [{"field": "model.lr", "op": "lt", "value": 0}]
    p = tmp_path / "cards"
    for entry in [a, b, c]:
        append(p, entry)
    cards = load_cards(p)
    selected = materialize(cards, ctx()["model_state"], [{"id": "a", "score": 1}], 1)
    assert [c["id"] for c in selected] == ["a", "b", "c"]
    assert selected[0]["condition_checks"][0]["status"] == "unknown"
    assert selected[2]["condition_checks"][0]["status"] == "not_satisfied"
    assert "counterexample_of:a" in selected[1]["retrieval_reasons"]
    a["counterexample_ids"] = ["missing"]
    p.write_text("")
    append(p, a)
    with pytest.raises(ValueError, match="Dangling"):
        load_cards(p)


def test_agent_assessment_cannot_drop_or_rewrite_counterevidence():
    from test_task import assessments

    cards = [card("a", "method", counterexample_ids=["b"]), card("b", "failure")]
    cards[0]["counterexamples"] = ["False negatives can dominate"]
    knowledge = materialize(cards, ctx()["model_state"], [{"id": "a", "score": 1}], 1)
    context = {**ctx(), "knowledge": knowledge}
    design = {"knowledge_assessments": assessments(knowledge)}
    validate_assessments(design, context)
    bad = copy.deepcopy(design)
    bad["knowledge_assessments"].pop()
    with pytest.raises(ValueError, match="every supplied card"):
        validate_assessments(bad, context)
    bad = copy.deepcopy(design)
    bad["knowledge_assessments"][0]["counterexample_checks"] = []
    with pytest.raises(ValueError, match="every inline"):
        validate_assessments(bad, context)
    bad = copy.deepcopy(design)
    bad["knowledge_assessments"][0]["linked_counterexample_checks"] = []
    with pytest.raises(ValueError, match="every linked"):
        validate_assessments(bad, context)
    bad = copy.deepcopy(design)
    bad["knowledge_assessments"][0]["status"] = "applicable"
    with pytest.raises(ValueError, match="cite evidence"):
        validate_assessments(bad, context)


def test_real_local_encoder_vectors_and_frozen_identity(tmp_path, tiny_encoder):
    np = pytest.importorskip("numpy")
    config = freeze_config(
        {"encoder": {"model_path": str(tiny_encoder), "max_length": 64}}, tmp_path
    )
    encoder = E5Encoder(config["encoder"])
    embeddings = encoder.encode(["negative sampling", "feature"], query=True)
    assert embeddings.shape == (2, 16)
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-6)
    config["encoder"]["files"]["config.json"] = "changed"
    with pytest.raises(ValueError, match="Frozen semantic"):
        E5Encoder(config["encoder"])
    with pytest.raises(ValueError, match="Local encoder"):
        freeze_config({"encoder": {"model_path": str(tmp_path / "absent")}}, tmp_path)


def test_invalid_vectors_do_not_fallback_to_lexical():
    class InvalidEncoder:
        def encode(self, texts, *, query):
            return [[float("nan")]] * len(texts)

    with pytest.raises(ValueError, match="Invalid semantic"):
        HybridRetriever(InvalidEncoder()).rank([card("a", "negative sampling")], plan())


QUERY_AGENT = """import json
from pathlib import Path
r=json.loads(Path('request.json').read_text())
c=r['context']
refs=['model.lr']+[k for k in c['model_state']['facts'] if k.startswith('diagnostic.')]
refs+=['trial:'+t['id'] for t in c['history']]
Path('retrieval-plan.json').write_text(json.dumps({'queries':[{'text':'failed negative sampling',
'intent':'counterevidence','keywords':['negative'],'conditions':[],
'rationale':'inspect current failures','evidence_refs':refs}]}))
"""


def command_study(tmp_path, study_config, script):
    config = read(study_config)
    agent = tmp_path / "query-agent.py"
    agent.write_text(QUERY_AGENT)
    backend = tmp_path / "backend.py"
    backend.write_text(script)
    config.update(
        enhanced=True,
        agent={"argv": [sys.executable, str(agent)]},
        retrieval={
            "backend": "command",
            "top_k": 1,
            "argv": [sys.executable, str(backend), "{request}", "{output}"],
            "artifacts": [str(backend)],
        },
    )
    write(study_config, config)
    root = tmp_path / "study"
    initialize(study_config, root)
    return root


RANKER = """import sys
from recevobench.taac.common import read,write,digest
r=read(sys.argv[1])
write(sys.argv[2],{'schema':'taac-retrieval-ranking-v1','request_digest':digest(r),
'ranking':[{'id':'ranking-negatives','score':1.0}]})
"""


def test_replaceable_backend_and_readonly_context(tmp_path, study_config):
    root = command_study(tmp_path, study_config, RANKER)
    assert context(root)["knowledge"] == []
    prepared = research_context(root)
    assert prepared["retrieval"]["backend"] == "command"
    assert prepared["knowledge"][0]["id"] == "ranking-negatives"
    assert any(c["id"] == "feature-fusion" for c in prepared["knowledge"])
    spent = load(root)[2]["spent_seconds"]
    assert context(root) == prepared
    assert load(root)[2]["spent_seconds"] == spent
    path = root / "research-context.json"
    altered = read(path)
    altered["knowledge"][0]["reason"] = "forged"
    write(path, altered)
    with pytest.raises(ValueError, match="context changed"):
        context(root)


@pytest.mark.parametrize(
    "script,match",
    [
        ("raise RuntimeError('intentional')", "Retrieval failed"),
        (RANKER.replace("digest(r)", "'wrong'"), "current request"),
        (RANKER.replace("'ranking-negatives'", "'invented'"), "unknown knowledge"),
    ],
)
def test_retrieval_failures_retain_cost_and_never_plan(tmp_path, study_config, script, match):
    root = command_study(tmp_path, study_config, script)
    with pytest.raises(ValueError, match=match):
        research_context(root)
    state = load(root)[2]
    assert state["spent_seconds"] > 0 and not state["pending"] and not state["trials"]
    assert context(root)["knowledge"] == []
    assert len(list((root / "retrieval").glob("*/failure.json"))) == 1


def test_retrieval_recovery_preserves_reservation(tmp_path, study_config):
    root = command_study(tmp_path, study_config, RANKER)
    state = load(root)[2]
    state["pending"] = {"kind": "retrieval", "id": "0000-query", "reserved_seconds": 3}
    write(root / "state.json", state)
    assert recover(root)["spent_seconds"] == 3
    assert recover(root)["spent_seconds"] == 3
