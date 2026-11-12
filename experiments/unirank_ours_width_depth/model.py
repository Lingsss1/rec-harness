"""Width-256 versions of the frozen winner, with one or three joint blocks."""
import torch
from torch import nn
from base_model import CONFIG as BASE_CONFIG, Model as BaseModel, JointBlock

CONFIG = {**BASE_CONFIG, 'embedding_dim': 16, 'token_dim': 256, 'joint_layers': 1}


class JointStack(nn.Module):
    def __init__(self, first, width, heads, count):
        super().__init__()
        self.blocks = nn.ModuleList([first])
        for _ in range(count - 1):
            block = JointBlock(width, heads)
            # Match the inherited host's Linear initialization, rather than
            # leaving newly appended layers at PyTorch's constructor default.
            for module in block.modules():
                if type(module) is nn.Linear:
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            self.blocks.append(block)

    def forward(self, tokens, valid):
        for block in self.blocks:
            tokens = block(tokens, valid)
        return tokens


class Model(BaseModel):
    def __init__(self, feature_map, joint_layers=1, **kwargs):
        if joint_layers not in (1, 3):
            raise ValueError('This fixed comparison supports joint depth 1 or 3')
        kwargs.setdefault('embedding_dim', 16)
        kwargs.setdefault('token_dim', 256)
        if kwargs['embedding_dim'] != 16 or kwargs['token_dim'] != 256:
            raise ValueError('The requested dimensions are embedding16 and token256')
        super().__init__(feature_map, **kwargs)
        self.joint_layers = joint_layers
        if joint_layers == 3:
            # Preserve the initial common parameters and the host sampling RNG
            # stream. Added layers have independent, ordinary Xavier weights.
            with torch.random.fork_rng(devices=[]):
                self.interaction = JointStack(self.interaction, kwargs['token_dim'],
                                              kwargs.get('num_heads', 2), joint_layers)
            self.model_to_device()
