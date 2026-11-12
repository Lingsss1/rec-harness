import unittest
try:
    import torch
except ImportError:
    raise unittest.SkipTest("PyTorch required for activation probe execution")
from rec_harness.activation_probe import activation_probe
from rec_harness.core import ContractError


class Scale(torch.nn.Module):
    def forward(self,x):return x*2


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.blocks=torch.nn.ModuleList([Scale(),torch.nn.Identity()])
    def encode_sequence(self,seq):
        x=seq.ne(0).float().unsqueeze(-1).expand(-1,-1,3)
        for block in self.blocks:x=block(x)
        return x


class ProbeTests(unittest.TestCase):
    def test_positive_null_controls_and_padding_mask(self):
        model=Model();sequence=torch.tensor([[0,0,1,2],[1,2,3,4]])
        result=activation_probe(model,sequence)
        self.assertEqual(result['blocks'][0]['activation_rms'],2)
        self.assertEqual(result['blocks'][0]['valid_positions'],6)
        self.assertEqual(result['blocks'][0]['bypass']['all']['relative_tail_change'],.5)
        self.assertEqual(result['blocks'][1]['bypass']['all']['tail_change_rms'],0)
        padded=torch.nn.functional.pad(sequence,(5,0))
        self.assertEqual(activation_probe(model,padded),result)
        self.assertTrue(model.training)

    def test_cleanup_on_invalid_activation_preserves_each_module_mode(self):
        model=Model();model.blocks[0].eval()
        def broken(module,args,out):return out*float('nan')
        external=model.blocks[1].register_forward_hook(broken)
        with self.assertRaises(ContractError):activation_probe(model,torch.tensor([[1,2]]))
        self.assertFalse(model.blocks[0].training);self.assertTrue(model.blocks[1].training)
        self.assertEqual(len(model.blocks[0]._forward_hooks),0)
        self.assertEqual(len(model.blocks[1]._forward_hooks),1)
        external.remove()

    def test_short_and_long_are_separate_and_invalid_histories_fail(self):
        result=activation_probe(Model(),torch.tensor([[0]*9+[1],[1]*10]))
        self.assertEqual(result['blocks'][0]['bypass']['short']['histories'],1)
        self.assertEqual(result['blocks'][0]['bypass']['long']['histories'],1)
        for seq in [torch.tensor([[0,0]]),torch.tensor([[1,0]]),torch.tensor([[1,0,1]])]:
            with self.assertRaises(ContractError):activation_probe(Model(),seq)


if __name__=='__main__':unittest.main()
