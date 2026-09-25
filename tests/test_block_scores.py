import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from rec_harness.core import ContractError
from rec_harness.adapters.score_worker import bypass_block


class Block:
    def __init__(self,factor):self.factor=factor;self.hooks=[]
    def register_forward_hook(self,hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda:self.hooks.remove(hook))
    def __call__(self,value):
        output=value*self.factor
        for hook in self.hooks:
            replacement=hook(self,(value,),output)
            if replacement is not None:output=replacement
        return output


class BlockScoreTests(unittest.TestCase):
    def test_known_identity_and_effectful_blocks_preserve_other_hooks_and_restore_after_failure(self):
        model=SimpleNamespace(blocks=[Block(1),Block(3)])
        seen=[];existing=model.blocks[1].register_forward_hook(lambda *args:seen.append('observed'))
        with bypass_block(model,0):self.assertEqual(model.blocks[0](2),2)
        with self.assertRaisesRegex(RuntimeError,'probe failure'):
            with bypass_block(model,1):
                self.assertEqual(model.blocks[1](2),2)
                raise RuntimeError('probe failure')
        self.assertEqual(model.blocks[1](2),6)
        self.assertEqual(seen,['observed','observed'])
        self.assertEqual(len(model.blocks[1].hooks),1);existing.remove()
        self.assertEqual(model.blocks[1].factor,3)

    def test_none_is_reference_and_invalid_indices_fail_without_hooks(self):
        model=SimpleNamespace(blocks=[Block(3)])
        with bypass_block(model,None):self.assertEqual(model.blocks[0](2),6)
        for value in [True,-1,1,0.0,'0']:
            with self.assertRaises(ContractError):
                with bypass_block(model,value):pass
        self.assertEqual(model.blocks[0].hooks,[])

    def test_host_refuses_intervention_on_confirmation_before_checkpoint_access(self):
        from rec_harness.adapters.amazon_rescore import rescore_job
        config={'datasets':['category'],'development_seeds':[42],'confirmation_seeds':[17]}
        with tempfile.TemporaryDirectory() as directory:
            with patch('rec_harness.adapters.amazon_rescore.load_config',return_value=config),patch('rec_harness.adapters.amazon_rescore.audit_job') as audit:
                for index,(seed,block) in enumerate([(17,0),(42,True),(42,-1)]):
                    with self.assertRaisesRegex(ContractError,'development seed'):
                        rescore_job(directory,'model','category',seed,Path(directory)/str(index),{},'unused',bypass_block=block)
                audit.assert_not_called()
