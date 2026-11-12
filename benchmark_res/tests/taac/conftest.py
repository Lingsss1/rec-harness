from pathlib import Path

import pytest

from recevobench.taac.common import write


@pytest.fixture(scope="session")
def prepared(tmp_path_factory):
    pytest.importorskip("numpy")
    pytest.importorskip("pyarrow")
    from recevobench.taac.data import prepare
    from recevobench.taac.fixture import make_fixture

    root = tmp_path_factory.mktemp("taac")
    config = make_fixture(root / "input")
    prepare(config, root / "prepared")
    return root / "prepared"


@pytest.fixture
def study_config(tmp_path, prepared):
    project = Path(__file__).resolve().parents[2]
    example = project / "examples/recommendation_autoresearch"
    value = {
        "data": str(prepared),
        "candidate": str(example / "baseline/candidate.py"),
        "program": str(example / "program.md"),
        "device": "cpu",
        "seeds": [42],
        "train": {"epochs": 1, "batch_size": 4, "max_steps": 2, "cpu_threads": 1},
        "limits": {"trial_seconds": 90, "total_seconds": 900, "agent_seconds": 15, "max_trials": 5},
        "min_delta": 0.0,
        "enhanced": False,
        "retrieval": {
            "backend": "hybrid",
            "top_k": 2,
            "encoder": {"model_path": str(tmp_path / "unused-encoder"), "max_length": 64},
        },
        "agent": None,
        "knowledge": str(example / "knowledge.jsonl"),
    }
    path = tmp_path / "study-config.json"
    write(path, value)
    return path


@pytest.fixture(scope="session")
def tiny_encoder(tmp_path_factory):
    """Random tiny encoder tests actual local inference plumbing, not retrieval quality."""
    pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    root = tmp_path_factory.mktemp("tiny-encoder")
    vocab = {
        t: i
        for i, t in enumerate(
            [
                "[PAD]",
                "[UNK]",
                "[CLS]",
                "[SEP]",
                "[MASK]",
                "query",
                "passage",
                ":",
                "negative",
                "sampling",
                "feature",
                "ranking",
                "history",
                "failure",
                "lr",
            ]
        )
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]"
    ).save_pretrained(root)
    torch.manual_seed(123)
    BertModel(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=128,
        )
    ).save_pretrained(root)
    return root
