"""Pinned model, independently initialized by the host for each trial."""
from longseq_bench.models import ModelConfig, HSTUEncoder, DAREEncoder
from longseq_bench.ccformer import CCFormerEncoder

def build(config):
    cfg = ModelConfig(**config)
    cls = DAREEncoder if cfg.name == 'dare' else CCFormerEncoder if cfg.name == 'ccformer' else HSTUEncoder
    return cls(cfg)
