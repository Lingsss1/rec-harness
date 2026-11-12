"""Replaceable retrieval: evidence-bound plans, hybrid ranking, mandatory counterevidence.

Backends rank IDs only. The host owns corpus contents, conditions and linked-card expansion.
No graph index, online model downloads, or lexical fallback on encoder failure.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

from .common import digest, file_hash, read, rows, write
from .knowledge import applicability


def evidence_refs(context):
    return set(context["model_state"]["facts"]) | {
        "trial:" + t["id"] for t in context.get("history", [])
    }


def validate_plan(plan, context):
    queries = plan.get("queries")
    if not isinstance(queries, list) or not 1 <= len(queries) <= 3:
        raise ValueError("Retrieval plan requires 1 to 3 queries")
    seen_refs = set()
    for q in queries:
        for key in ["text", "rationale"]:
            if not isinstance(q.get(key), str) or not q[key].strip() or len(q[key]) > 4000:
                raise ValueError("Query requires bounded nonempty " + key)
        if q.get("intent") not in {"mechanism", "method", "counterevidence"}:
            raise ValueError("Query intent must be mechanism, method or counterevidence")
        keywords = q.get("keywords")
        if (
            not isinstance(keywords, list)
            or len(keywords) > 16
            or any(not isinstance(k, str) or not k.strip() or len(k) > 200 for k in keywords)
        ):
            raise ValueError("Query keywords must be a bounded string array")
        refs = q.get("evidence_refs")
        if not isinstance(refs, list) or not refs or not set(refs) <= evidence_refs(context):
            raise ValueError("Query evidence_refs must resolve to current facts/history")
        seen_refs.update(refs)
        for c in q.get("conditions", []):
            # Conditions select an additional recall lane; never discard other candidates.
            if set(c) != {"field", "op", "value"}:
                raise ValueError("Query conditions require field/op/value")
            applicability({"conditions": [c]}, context["model_state"]["facts"])
    if not any(r.startswith("model.") for r in seen_refs):
        raise ValueError("Retrieval plan must cite current model facts")
    if context.get("history") and not any(r.startswith("trial:") for r in seen_refs):
        raise ValueError("Retrieval plan must cite historical experiments")
    if any(r.startswith("diagnostic.") for r in evidence_refs(context)) and not any(
        r.startswith("diagnostic.") for r in seen_refs
    ):
        raise ValueError("Retrieval plan must cite available diagnostics")
    if not any(q["intent"] == "counterevidence" for q in queries):
        raise ValueError("Retrieval plan must include a counterevidence query")


def load_cards(path, extra_cards=()):
    cards = list(rows(path)) + list(extra_cards)
    ids = [c["id"] for c in cards]
    if not cards or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Knowledge corpus requires nonempty unique IDs")
    for c in cards:
        for key in ["scene", "method", "gain", "reason", "sources", "conditions"]:
            if key not in c:
                raise ValueError("Knowledge card missing " + key)
        applicability(c, {})
        for key in ["counterexamples", "counterexample_ids"]:
            values = c.get(key, [])
            if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
                raise ValueError("Card " + key + " must be a string array")
            if len(set(values)) != len(values):
                raise ValueError("Duplicate " + key)
        if not set(c.get("counterexample_ids", [])) <= set(ids):
            raise ValueError("Dangling counterexample_ids in " + c["id"])
    return cards


def history_cards(context):
    """Current-run outcomes remain searchable even after discard; explanations are provisional."""
    return [
        {
            "id": "trial-" + trial["id"],
            "scene": {"release": context["task"]["release"], "parent": trial["parent"]},
            "method": trial.get("design", {}).get("change", trial["hypothesis"]),
            "gain": trial.get("delta"),
            "reason": {
                "analysis": trial.get("analysis"),
                "failure": trial.get("failure"),
                "diagnostics": trial.get("diagnostics"),
            },
            "sources": [{"trial_id": trial["id"], "source_sha256": trial["source_sha256"]}],
            "source_status": "current_run_provisional",
            "conditions": [
                {"field": "task.release", "op": "eq", "value": context["task"]["release"]}
            ],
            "counterexamples": [
                "A single trial does not establish causality or independent replication"
            ],
            "previous_status": trial["status"],
        }
        for trial in context["history"]
    ]


def card_text(card):
    return json.dumps(
        {
            k: card[k]
            for k in [
                "scene",
                "method",
                "gain",
                "reason",
                "conditions",
                "counterexamples",
                "minimum_comparison",
                "previous_status",
            ]
            if k in card
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def tokens(text):
    # English identifiers + Chinese characters/bigrams, without a language-service dependency.
    result = re.findall(r"[a-z0-9_]+", text.lower())
    for word in re.findall(r"[\u3400-\u9fff]+", text):
        result.extend(word)
        result.extend(word[i : i + 2] for i in range(len(word) - 1))
    return result


def bm25(documents, query):
    counts = [Counter(tokens(d)) for d in documents]
    lengths = [sum(c.values()) for c in counts]
    avg = sum(lengths) / max(len(lengths), 1) or 1
    scores = [0.0] * len(documents)
    for token in set(tokens(query)):
        df = sum(token in c for c in counts)
        idf = math.log(1 + (len(counts) - df + 0.5) / (df + 0.5))
        for i, c in enumerate(counts):
            tf = c[token]
            scores[i] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * lengths[i] / avg))
    return scores


class Encoder(Protocol):
    def encode(self, texts: list[str], *, query: bool): ...


class RetrievalBackend(Protocol):
    def rank(self, cards: list[dict], plan: dict) -> list[dict]: ...


def encoder_files(path):
    path = Path(path)
    files = {
        str(p.relative_to(path)): file_hash(p)
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.suffix in {".json", ".safetensors", ".txt", ".model"}
    }
    if "config.json" not in files or not any(n.endswith(".safetensors") for n in files):
        raise ValueError("Local encoder needs config/tokenizer and safetensors weights")
    return files


def encoder_runtime():
    return {p: version(p) for p in ["torch", "transformers", "tokenizers", "safetensors"]}


def freeze_config(config, base):
    result = dict(config)
    if "research_query" in result:
        raise ValueError("Use an Agent retrieval plan, not a fixed research_query")
    top_k = result.setdefault("top_k", 4)
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("retrieval.top_k must be between 1 and 100")
    backend = result.setdefault("backend", "hybrid")
    if backend == "hybrid":
        enc = dict(result.get("encoder", {}))
        if not enc.get("model_path"):
            raise ValueError("Hybrid retrieval requires retrieval.encoder.model_path")
        enc["model_path"] = str((Path(base) / enc["model_path"]).resolve())
        for key, default, maximum in [
            ("batch_size", 8, 128),
            ("max_length", 512, 512),
            ("threads", 2, 64),
        ]:
            enc.setdefault(key, default)
            if type(enc[key]) is not int or not 1 <= enc[key] <= maximum:
                raise ValueError("Invalid encoder " + key)
        enc["files"] = encoder_files(enc["model_path"])
        enc["runtime"] = encoder_runtime()
        result["encoder"] = enc
    elif backend == "command":
        argv = result.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(a, str) for a in argv):
            raise ValueError("Command backend requires argv")
        artifacts = result.get("artifacts", [])
        if not artifacts:
            raise ValueError("Command backend must declare code/model artifacts to freeze")
        result["artifact_hashes"] = {
            str((Path(base) / p).resolve()): file_hash(Path(base) / p) for p in artifacts
        }
    else:
        raise ValueError("Unknown retrieval backend")
    return result


class E5Encoder:
    def __init__(self, config):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if (
            encoder_files(config["model_path"]) != config["files"]
            or encoder_runtime() != config["runtime"]
        ):
            raise ValueError("Frozen semantic encoder/runtime changed")
        self.config, self.torch = config, torch
        torch.set_num_threads(config["threads"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModel.from_pretrained(
                config["model_path"],
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
            )
            .float()
            .cpu()
            .eval()
        )

    def encode(self, texts, *, query):
        torch, cfg = self.torch, self.config
        results = []
        with torch.inference_mode():
            for start in range(0, len(texts), cfg["batch_size"]):
                inputs = self.tokenizer(
                    [
                        ("query: " if query else "passage: ") + t
                        for t in texts[start : start + cfg["batch_size"]]
                    ],
                    padding=True,
                    truncation=True,
                    max_length=cfg["max_length"],
                    return_tensors="pt",
                )
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                vector = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                results.append(torch.nn.functional.normalize(vector, dim=1).numpy())
        import numpy as np

        return np.concatenate(results)


class HybridRetriever:
    def __init__(self, encoder: Encoder):
        self.encoder = encoder

    def rank(self, cards, plan):
        import numpy as np

        docs = [card_text(c) for c in cards]
        queries = plan["queries"]
        vectors = []
        for matrix, count in [
            (self.encoder.encode(docs, query=False), len(docs)),
            (self.encoder.encode([q["text"] for q in queries], query=True), len(queries)),
        ]:
            a = np.asarray(matrix, dtype=float)
            if a.ndim != 2 or a.shape[0] != count or not a.shape[1] or not np.isfinite(a).all():
                raise ValueError("Invalid semantic vectors")
            norm = np.linalg.norm(a, axis=1, keepdims=True)
            if (norm == 0).any():
                raise ValueError("Zero semantic vectors")
            vectors.append(a / norm)
        if vectors[0].shape[1] != vectors[1].shape[1]:
            raise ValueError("Semantic vector dimension mismatch")
        semantic = vectors[1] @ vectors[0].T
        totals, traces = [0.0] * len(cards), [[] for _ in cards]
        for qi, q in enumerate(queries):
            lexical = bm25(docs, q["text"] + " " + " ".join(q["keywords"]))
            # A structural lane matches explicitly recorded scene conditions, not guessed facts.
            condition_matches = [
                sum(cond in c["conditions"] for cond in q.get("conditions", [])) for c in cards
            ]
            for lane, scores in [
                ("semantic", semantic[qi].tolist()),
                ("bm25", lexical),
                ("conditions", condition_matches),
            ]:
                ordered = sorted(range(len(cards)), key=lambda i: (-scores[i], cards[i]["id"]))
                ordered = [i for i in ordered if lane == "semantic" or scores[i] > 0]
                for rank, i in enumerate(ordered, 1):
                    totals[i] += 1 / (60 + rank)
                    traces[i].append({"query": qi, "lane": lane, "score": scores[i], "rank": rank})
        return [
            {"id": cards[i]["id"], "score": totals[i], "trace": traces[i]}
            for i in sorted(range(len(cards)), key=lambda i: (-totals[i], cards[i]["id"]))
        ]


def materialize(cards, state, ranking, limit):
    """Hydrate trusted cards and close explicit counterexample links, even outside top-k."""
    by_id = {c["id"]: c for c in cards}
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("Retrieval backend returned an empty/invalid ranking")
    ids = [r["id"] for r in ranking]
    if len(ids) != len(set(ids)) or not set(ids) <= set(by_id):
        raise ValueError("Ranking has duplicate/unknown knowledge IDs")
    if any(
        not isinstance(r.get("score"), (int, float)) or not math.isfinite(r["score"])
        for r in ranking
    ):
        raise ValueError("Ranking requires finite scores")
    if any(a["score"] < b["score"] for a, b in zip(ranking, ranking[1:], strict=False)):
        raise ValueError("Ranking must be sorted descending")
    selected = list(ids[:limit])
    reasons = {i: ["retrieved"] for i in selected}
    # Explicit ID references only: no inferred relationships or graph construction.
    for cid in selected:
        for linked in by_id[cid].get("counterexample_ids", []):
            if linked not in reasons:
                selected.append(linked)
                reasons[linked] = []
            reasons[linked].append("counterexample_of:" + cid)
    rank_by_id = {r["id"]: r for r in ranking}
    return [
        {
            **by_id[cid],
            "condition_checks": applicability(by_id[cid], state["facts"]),
            "source_status": by_id[cid].get("source_status", "external_unverified"),
            "retrieval_reasons": reasons[cid],
            "retrieval_rank": rank_by_id.get(cid),
        }
        for cid in selected
    ]


def worker(request, output):
    value = read(request)
    backend = HybridRetriever(E5Encoder(value["config"]["encoder"]))
    write(
        output,
        {
            "schema": "taac-retrieval-ranking-v1",
            "request_digest": digest(value),
            "ranking": backend.rank(value["cards"], value["plan"]),
        },
    )
