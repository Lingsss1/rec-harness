import unittest
from rec_harness.adapters.model_source_contract import review_source
from rec_harness.core import ContractError

SOURCE='''import torch
from torch import nn
CONFIG = 48
class NextItNetBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
    def forward(self, x):
        return self.linear(x)
class NextItNetModel(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.block = nn.Linear(hidden, hidden)
    def forward(self, x):
        return self.block(x)
'''

class SourceContractTests(unittest.TestCase):
    def test_position_specific_registered_block_calls(self):
        source=SOURCE.replace('self.block = nn.Linear(hidden, hidden)',
            'self.block = nn.Linear(hidden, hidden)\n        self.blocks = nn.ModuleList([NextItNetBlock(hidden)])')
        for body in ('for i, block in enumerate(self.blocks):\n            x = block(x) * (0.95 if i == 0 else 1.0)\n        return x',
                     'for i in range(len(self.blocks)):\n            x = self.blocks[i](x)\n        return x',
                     'return self.blocks[0](x)', 'return self.blocks[-1](x)'):
            with self.subTest(body=body):
                review_source(source,source.replace('return self.block(x)',body))

    def test_unrelated_enumerated_and_indirect_calls_still_rejected(self):
        source=SOURCE.replace('self.block = nn.Linear(hidden, hidden)',
            'self.block = nn.Linear(hidden, hidden)\n        self.blocks = nn.ModuleList([NextItNetBlock(hidden)])')
        for body in ('for i, block in enumerate(x):\n            x = block(x)\n        return x',
                     'return x[0](x)', 'return self.block[0](x)',
                     'return self.blocks[:](x)', 'return self.blocks[0.5](x)',
                     'return self.blocks[0].register_forward_hook(x)'):
            with self.subTest(body=body),self.assertRaises(ContractError):
                review_source(source,source.replace('return self.block(x)',body))
        with self.assertRaises(ContractError):
            review_source(SOURCE,SOURCE.replace('return self.block(x)','return self.blocks[0](x)'))

    def test_inline_block_uses_declared_modules_only(self):
        source=SOURCE.replace('self.block = nn.Linear(hidden, hidden)',
            'self.block = nn.Linear(hidden, hidden)\n        self.blocks = nn.ModuleList([NextItNetBlock(hidden)])')
        source=source.replace('return self.block(x)','for block in self.blocks:\n            x = block(x)\n        return x')
        candidate=source.replace('x = block(x)','x = block.linear(x)')
        review_source(source,candidate)
        helper_source=source.replace('return self.linear(x)',
            'return self.linear(x)\n    def _causal_conv(self, conv, x):\n        return conv(x)')
        review_source(helper_source,helper_source.replace('x = block(x)',
            'x = block._causal_conv(block.linear, x)'))
        with self.assertRaises(ContractError):
            review_source(source,source.replace('x = block(x)','x = block._causal_conv(block.linear, x)'))
        for call in ['block.unknown(x)','block.register_forward_hook(x)','block.linear.forward(x)',
                     'block.__getattribute__(x)','x.linear(x)']:
            with self.subTest(call=call),self.assertRaises(ContractError):
                review_source(source,source.replace('x = block(x)','x = '+call))

    def test_new_learnable_architecture_without_exact_template(self):
        candidate=SOURCE.replace('self.block = nn.Linear(hidden, hidden)',
            'self.block = nn.Linear(hidden, hidden)\n        self.gate = nn.Parameter(torch.zeros(hidden))')
        candidate=candidate.replace('return self.block(x)','return self.block(x) * torch.sigmoid(self.gate)')
        self.assertEqual(review_source(SOURCE,candidate)['contract'],'nextitnet-model-classes-v1')

    def test_changes_identify_actual_methods_and_fields_not_comments(self):
        candidate=SOURCE.replace('self.block = nn.Linear(hidden, hidden)',
            'self.block = nn.Linear(hidden, hidden)\n        self.gate = nn.Parameter(torch.zeros(hidden))')
        candidate=candidate.replace('return self.block(x)','return self.block(x) * torch.sigmoid(self.gate)')
        delta=review_source(SOURCE,candidate)['structural_changes']
        methods={row['method']:row for row in delta['methods']}
        self.assertEqual(set(methods),{'NextItNetModel.__init__','NextItNetModel.forward'})
        init=methods['NextItNetModel.__init__']
        self.assertEqual(init['before_written_fields'],['self.block'])
        self.assertEqual(init['after_written_fields'],['self.block','self.gate'])
        self.assertNotEqual(init['before_ast_hash'],init['after_ast_hash'])
        self.assertEqual(review_source(SOURCE,candidate+'\n# formatting comment\n')['structural_changes'],delta)
        self.assertEqual(review_source(SOURCE,SOURCE+'\n# comment\n')['structural_changes']['methods'],[])
        self.assertEqual(delta['candidate_ast_hash'],review_source(SOURCE,candidate)['source_ast_hash'])

    def test_forbidden_capabilities_and_protocol_edits(self):
        payloads=['open("/tmp/leak", "w")','torch.save(x, "/tmp/leak")',
                  'self.io = torch.save','x.__class__', 'eval("1")',
                  'self.forward = torch.load', 'import os', 'nn = x',
                  'self.library = torch\n        self.library.tensor = self.block',
                  'self.block.forward = x.__getattribute__']
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(ContractError):
                review_source(SOURCE,SOURCE.replace('return self.block(x)',payload+'\n        return self.block(x)'))
        with self.assertRaises(ContractError):review_source(SOURCE,SOURCE.replace('CONFIG = 48','CONFIG = 64'))
        with self.assertRaises(ContractError):review_source(SOURCE,SOURCE.replace('def forward(self, x):','def forward(self, x, y):'))
