import json

import pytest

from recevobench.taac.__main__ import main
from recevobench.taac.common import digest, read, write
from recevobench.taac.retrieval import freeze_config, load_cards


def invoke(capsys, *args):
    assert main(list(args)) == 0
    return json.loads(capsys.readouterr().out)


def test_cli_preparation_and_readonly_study_commands(tmp_path, study_config, capsys):
    fixture = invoke(capsys, "fixture", "--output", str(tmp_path / "raw"))
    assert fixture["fixture"]
    data = tmp_path / "data"
    invoke(capsys, "prepare", "--config", fixture["config"], "--output", str(data))
    manifest = invoke(capsys, "verify", "--data", str(data))
    config = read(study_config)
    config["data"] = str(data)
    write(study_config, config)
    preflight = invoke(capsys, "preflight", "--config", str(study_config))
    assert preflight["data_exists"] and preflight["candidate_exists"]
    root = tmp_path / "study"
    initialized = invoke(capsys, "init", "--config", str(study_config), "--output", str(root))
    assert initialized["data_identity"] == manifest["identity"]
    ctx = invoke(capsys, "context", "--study", str(root))
    assert ctx["history"] == [] and ctx["knowledge"] == []
    state = invoke(capsys, "status", "--study", str(root))
    assert state["spent_seconds"] == 0 and state["trials"] == []
    assert invoke(capsys, "recover", "--study", str(root)) == state
    with pytest.raises(SystemExit) as error:
        main(["run", "--study", str(root), "--rounds", "0"])
    assert error.value.code == 2
    assert "Positive rounds" in capsys.readouterr().err


def test_cli_retrieval_worker_binds_response_to_request(
    tmp_path, study_config, tiny_encoder, capsys
):
    request = {
        "config": freeze_config(
            {"encoder": {"model_path": str(tiny_encoder), "max_length": 64}}, tmp_path
        ),
        "cards": load_cards(read(study_config)["knowledge"]),
        "plan": {"queries": [{"text": "negative sampling", "keywords": ["negative"]}]},
    }
    request_path, output = tmp_path / "request.json", tmp_path / "ranking.json"
    write(request_path, request)
    result = invoke(capsys, "_retrieve", "--request", str(request_path), "--output", str(output))
    assert result["output"] == str(output)
    ranking = read(output)
    assert ranking["request_digest"] == digest(request)
    assert {r["id"] for r in ranking["ranking"]} == {c["id"] for c in request["cards"]}
