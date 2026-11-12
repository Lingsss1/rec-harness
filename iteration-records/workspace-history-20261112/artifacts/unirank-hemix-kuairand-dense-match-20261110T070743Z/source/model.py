"""Pinned UniRank HeMix with only token width reduced; the host owns training."""
from HeMix import HeMix

CONFIG = {
    'embedding_dim': 16, 'tower_hidden_units': [128, 64],
    'num_layers': 3, 'expansion_factor': 4, 'token_dim': 80,
    'num_glocal_token': 2, 'num_real_token': 2, 'num_mix_heads': 2,
    'low_rank_dim': 16, 'tower_activations': 'ReLU', 'net_dropout': 0,
    'real_seq_len': 20, 'accumulation_steps': 1,
}


class Model(HeMix):
    def __init__(self, feature_map, **kwargs):
        # Native metadata supplies task order/count. These constructor defaults
        # only initialize inherited loss/optimizer objects; the unchanged host
        # performs every actual step using the frozen comparison protocol.
        kwargs.setdefault('num_tasks', len(feature_map.labels))
        kwargs.setdefault('loss', ['binary_crossentropy'] * len(feature_map.labels))
        kwargs.setdefault('dense_optimizer', 'AdamW')
        kwargs.setdefault('dense_learning_rate', 1e-4)
        kwargs.setdefault('sparse_optimizer', 'Adagrad')
        kwargs.setdefault('sparse_learning_rate', 0.05)
        super().__init__(feature_map, **kwargs)
