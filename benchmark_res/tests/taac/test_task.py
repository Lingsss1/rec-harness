import math
import sqlite3

import pytest

from recevobench.taac.common import append, read, rows, verify_data
from recevobench.taac.data import _lookup, partition, query_from_sequence
from recevobench.taac.evaluate import score
from recevobench.taac.knowledge import (
    applicability,
    memory_cards,
    model_facts,
    validate_design,
)
from recevobench.taac.retrieval import load_cards, materialize


def test_native_identity_mapping_and_no_target_exposure(prepared):
    np = pytest.importorskip("numpy")
    manifest = verify_data(prepared)
    assert manifest["identity"]
    meta = read(prepared / "public/metadata.json")
    assert meta["fixture"] and meta["candidate_mm_coverage"] == 1
    item_mm = np.load(prepared / "public/item_mm.npy")
    candidate_mm = np.load(prepared / "public/candidate_mm.npy")
    assert np.any(item_mm[1])
    np.testing.assert_equal(candidate_mm[0], item_mm[1])
    assert read(prepared / "public/candidate_ids.json")[0] == "20000000001"
    all_users = set()
    for part in ["feedback", "gate", "final"]:
        queries = list(rows(prepared / f"public/{part}.jsonl"))
        answers = {r["user_id"]: r["target"] for r in rows(prepared / f"private/{part}.jsonl")}
        assert queries and set(answers) == {r["user_id"] for r in queries}
        for q in queries:
            assert "target" not in q
            assert max(e["timestamp"] for e in q["history"]) < 7
            assert int(answers[q["user_id"]]) - 20000000000 not in {
                e["item_id"] for e in q["history"]
            }
            assert q["user_id"] not in all_users
            all_users.add(q["user_id"])
    with sqlite3.connect(prepared / "public/train.sqlite") as db:
        train_users = {str(r[0]) for r in db.execute("SELECT uid FROM examples")}
    assert not train_users & all_users


def test_query_rejects_ambiguous_exposure_and_preserves_time_boundary():
    events = [
        {"item_id": i, "action_type": a, "timestamp": t}
        for i, a, t in [(1, 0, 1), (1, 1, 2), (2, 0, 3), (2, 0, 4), (2, 1, 5)]
    ]
    assert query_from_sequence(events, {2}, 0.4)[1] == "ambiguous_or_missing_exposure"
    events.pop(2)
    q, reason = query_from_sequence(events, {2}, 0.5)
    assert reason is None
    assert [e["timestamp"] for e in q["history"]] == [1, 2]
    assert query_from_sequence(events, {99}, 0.5)[1] == "target_outside_candidates"
    events[-1]["timestamp"] = 0
    with pytest.raises(ValueError, match="chronological"):
        query_from_sequence(events, {2}, 0.5)


def test_split_validation_and_integer_string_ids():
    assert _lookup({100: 7}, "100") == 7
    assert _lookup({"100": 7}, 100) == 7
    assert _lookup({}, "abc", 0) == 0
    with pytest.raises(ValueError, match="sum"):
        partition("u", "salt", [0.5, 0.2, 0.1, 0.1])
    with pytest.raises(ValueError, match="positive"):
        partition("u", "salt", [0.5, 0.5, 0, 0])
    assert partition("u", "salt", [0.5, 0.2, 0.15, 0.15]) == partition(
        "u", "salt", [0.5, 0.2, 0.15, 0.15]
    )


def test_unknown_actions_are_not_relabelled_or_used_for_holdout_targets():
    events = [
        {"item_id": i, "action_type": a, "timestamp": t}
        for i, a, t in [(1, 0, 1), (1, 1, 2), (2, 0, 3), (2, 1, 4)]
    ]
    assert query_from_sequence(events, {2}, 0.5)[1] is None
    for position in range(len(events)):
        modified = [dict(e) for e in events]
        modified[position]["action_type"] = None
        assert query_from_sequence(modified, {2}, 0.5) == (None, "unknown_action")
        assert modified[position]["action_type"] is None
    modified[-1]["timestamp"] = 0
    with pytest.raises(ValueError, match="chronological"):
        query_from_sequence(modified, {2}, 0.5)


def test_prepare_preserves_unknown_training_actions_and_counts_excluded_queries(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    from recevobench.taac.data import prepare
    from recevobench.taac.fixture import make_fixture

    config_path = make_fixture(tmp_path / "input")
    config = read(config_path)
    sequence_path = tmp_path / "input/raw/seq/part-0.parquet"
    sequences = pq.read_table(sequence_path).to_pylist()
    selected = {}
    for row in sequences:
        part = partition(row["user_id"], config["split_salt"], config["fractions"])
        if part in {"train", "feedback"} and part not in selected:
            row["seq"][0]["action_type"] = None
            selected[part] = row["user_id"]
    pq.write_table(pa.Table.from_pylist(sequences), sequence_path)
    output = tmp_path / "prepared"
    metadata = prepare(config_path, output)
    assert metadata["counts"]["unknown_action_events"] == 2
    assert metadata["counts"]["users_with_unknown_action"] == 2
    assert metadata["counts"]["feedback_excluded_unknown_action"] == 1
    assert str(selected["feedback"]) not in {
        r["user_id"] for r in rows(output / "private/feedback.jsonl")
    }
    with sqlite3.connect(output / "public/train.sqlite") as db:
        import json

        events = json.loads(
            db.execute("SELECT events FROM examples WHERE uid=?", (selected["train"],)).fetchone()[0]
        )
    assert events[0]["action_type"] is None
    verify_data(output)


def test_null_multimodal_vectors_remain_missing_and_reduce_measured_coverage(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    np = pytest.importorskip("numpy")
    from recevobench.taac.data import prepare
    from recevobench.taac.fixture import make_fixture

    config = make_fixture(tmp_path / "input")
    path = tmp_path / "input/raw/mm_emb/emb_81_32_parquet/part-0.parquet"
    vectors = pq.read_table(path).to_pylist()
    vectors[0]["emb"] = None
    pq.write_table(pa.Table.from_pylist(vectors), path)
    output = tmp_path / "prepared"
    metadata = prepare(config, output)
    assert metadata["counts"]["missing_mm_vectors"] == 1
    assert metadata["counts"]["candidate_missing_mm_vectors"] == 1
    assert metadata["candidate_mm_coverage"] == pytest.approx(15 / 16)
    assert not np.load(output / "public/mm_seen.npy")[1]
    assert not np.any(np.load(output / "public/item_mm.npy")[1])
    assert not np.any(np.load(output / "public/candidate_mm.npy")[0])
    verify_data(output)


def test_independent_top10_score_and_rejections(tmp_path):
    pred, truth = tmp_path / "p.jsonl", tmp_path / "a.jsonl"
    ids = [str(i) for i in range(12)]
    append(pred, {"user_id": "u1", "items": ids[:10]})
    append(pred, {"user_id": "u2", "items": ids[:10]})
    append(truth, {"user_id": "u1", "target": "1"})
    append(truth, {"user_id": "u2", "target": "11"})
    metric = score(pred, truth, ids)
    assert metric["HitRate@10"] == 0.5
    assert metric["NDCG@10"] == pytest.approx(1 / math.log2(3) / 2)
    assert metric["score"] == pytest.approx(0.31 * 0.5 + 0.69 / math.log2(3) / 2)
    append(pred, {"user_id": "u1", "items": ids[:10]})
    with pytest.raises(ValueError, match="Duplicate"):
        score(pred, truth, ids)


def test_missing_prediction_not_silently_dropped(tmp_path):
    p, a = tmp_path / "p", tmp_path / "a"
    append(p, {"user_id": "u", "items": list(map(str, range(10)))})
    append(a, {"user_id": "v", "target": "1"})
    with pytest.raises(ValueError, match="exactly match"):
        score(p, a, list(map(str, range(10))))


def test_applicability_unknown_is_not_absence(prepared, study_config):
    config = read(study_config)
    state = model_facts(config["candidate"], read(prepared / "public/metadata.json"))
    cards = all_cards(config["knowledge"], state)
    card = next(c for c in cards if c["id"] == "action-conditioning")
    assert card["condition_checks"][0]["status"] == "unknown"
    assert (
        applicability({"conditions": [{"field": "x", "op": "gt", "value": 1}]}, {"x": 2})[0][
            "status"
        ]
        == "satisfied"
    )
    ctx = {"knowledge": cards, "model_state": state}
    design = {
        "hypothesis": "test",
        "change": "test",
        "falsification": "test",
        "controls": ["same protocol"],
        "budget_rationale": "test",
        "diffs": [
            {
                "knowledge_id": card["id"],
                "current_refs": ["model.negatives"],
                "condition_checks": card["condition_checks"],
                "alternatives": [],
                "missing_evidence": ["actual code inspection"],
                "experiment": "test",
                "priority_reason": "test",
                "selected": True,
            }
        ],
    }
    design["knowledge_assessments"] = assessments(cards)
    design["diffs"][0]["exploration_rationale"] = "test unknown applicability"
    validate_design(design, ctx, True)
    design["diffs"][0]["condition_checks"][0]["status"] = "satisfied"
    # Do not mutate the expected context while forging the report.
    ctx["knowledge"] = all_cards(config["knowledge"], state)
    with pytest.raises(ValueError, match="retain host applicability"):
        validate_design(design, ctx, True)


def test_changed_prepared_file_is_rejected(tmp_path, prepared):
    import shutil

    target = tmp_path / "data"
    shutil.copytree(prepared, target)
    (target / "public/metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_data(target)


def test_prior_agent_memory_preserves_failure_and_scope(prepared, study_config):
    config = read(study_config)
    meta = read(prepared / "public/metadata.json")
    export = {
        "schema": "taac-stage-memory-v1",
        "records": [
            {
                "scene": meta,
                "method": "historical failed candidate",
                "gain": -0.01,
                "reason": {"alternatives": ["noise"]},
                "status": "discard",
                "parent": "t0000",
                "evidence": {"receipt": "an-unverified-prior-receipt-hash"},
            }
        ],
    }
    cards = memory_cards(export)
    state = model_facts(config["candidate"], meta)
    retrieved = all_cards(config["knowledge"], state, extra_cards=cards)
    prior = next(c for c in retrieved if c["id"].startswith("prior-"))
    assert prior["previous_status"] == "discard" and prior["gain"] == -0.01
    assert prior["source_status"] == "prior_run_unverified"
    assert prior["condition_checks"][0]["status"] == "satisfied"
    state["facts"]["task.release"] = "another-scene"
    assert applicability(prior, state["facts"])[0]["status"] == "not_satisfied"


def all_cards(path, state, extra_cards=()):
    cards = load_cards(path, extra_cards)
    return materialize(cards, state, [{"id": c["id"], "score": 1.0} for c in cards], len(cards))


def assessments(cards):
    def judgment():
        return {"status": "unknown", "rationale": "requires evidence", "evidence_refs": []}

    return [
        {
            "knowledge_id": c["id"],
            **judgment(),
            "condition_checks": c["condition_checks"],
            "counterexample_checks": [
                {"counterexample": v, **judgment()} for v in c.get("counterexamples", [])
            ],
            "linked_counterexample_checks": [
                {"knowledge_id": v, **judgment()} for v in c.get("counterexample_ids", [])
            ],
        }
        for c in cards
    ]
