from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from amazon_family_compare.bootstrap import bootstrap
bootstrap()
from amazon_family_compare.controller import FamilyController
from amazon_family_compare.core import read, save, schema, TEXT
from amazon_family_compare.prepare import WORKSPACE
from .controller import RecoveryController
from .launch import audited_budget


class RecoveryTests(unittest.TestCase):
    def config(self, root, method='mlestar'):
        c = read(WORKSPACE/'artifacts/model-family-comparison-20261029/runs'/
                 ('bert4rec-'+method)/'config.json')
        return {**c,'root':str(root),'llm_slot_directory':str(Path(root)/'slots'),
                'llm_attempts':2,'retry_backoff_seconds':[0,0]}

    def receipt(self, c, key, source='x=1\n', cost=12):
        path = c.root/'trials'/key
        path.mkdir(parents=True)
        (path/'train.py').write_text(source)
        r = {'id':key,'status':'success','score':.55,'output':str(path),
             'allocated_gpu_seconds':cost}
        save(path/'receipt.json',r)
        return r

    def test_completed_wave_replays_even_when_budget_is_exhausted(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = RecoveryController(self.config(tmp),'mlestar')
            records = [self.receipt(c,'old-t'+str(t)) for t in range(4)]
            before = read(c.ledger.path)
            before['spent_gpu_seconds'] = 43199
            save(c.ledger.path,before)
            with patch.object(c,'_trial',side_effect=AssertionError('must not retrain')):
                actual = c.wave([{'id':r['id'],'source':'x=1\n'} for r in records])
            self.assertEqual(actual,records)
            self.assertEqual(read(c.ledger.path),before)

    def test_partial_wave_only_trains_missing_id_on_original_slot(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = RecoveryController(self.config(tmp),'mlestar')
            r = self.receipt(c,'old-t0')
            fresh = {'id':'new-t1','status':'success','score':.6}
            with patch.object(c,'_trial',return_value=fresh) as train:
                result = c.wave([{'id':'old-t0','source':'x=1\n'},
                                 {'id':'new-t1','source':'y=2\n'}])
            self.assertEqual(result,[r,fresh])
            self.assertEqual(train.call_count,1)
            self.assertEqual(train.call_args.args[:3],('new-t1','y=2\n',c.c['gpus'][1]))

    def test_changed_cached_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = RecoveryController(self.config(tmp),'mlestar')
            self.receipt(c,'old-t0')
            with self.assertRaisesRegex(RuntimeError,'mismatched'):
                c.wave([{'id':'old-t0','source':'x=2\n'}])

    def test_prior_exact_llm_cache_is_reused_by_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(tmp)
            c = FamilyController(config,'mlestar')
            out = schema({'value':TEXT})
            with patch.object(c,'run_attempt',return_value={'value':'cached'}):
                self.assertEqual(c.llm('fixture',out,key='fixed'),{'value':'cached'})
            resumed = RecoveryController(config,'mlestar')
            with patch.object(resumed,'run_attempt',side_effect=AssertionError('must reuse cache')):
                self.assertEqual(resumed.llm('fixture',out,key='fixed'),{'value':'cached'})

    def test_missing_block_selections_only_generate_missing_trajectories(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = RecoveryController(self.config(tmp),'mlestar')
            response = {'selections':[{'trajectory':t,'block':'loss','reason':'original'} for t in [0,1]]}
            prompt = 'Select one block\n'+json.dumps({'source':['def loss():\n pass\n']*4})
            def answer(instruction, out, search=False, key=None):
                t = out['properties']['trajectory']['enum'][0]
                return {'trajectory':t,'block':'loss','reason':'completed'}
            with patch.object(FamilyController,'llm',side_effect=answer) as call:
                fixed = c.complete_selection(prompt,response,'select-1')
            self.assertEqual(call.call_count,2)
            self.assertEqual([r['trajectory'] for r in fixed['selections']],[0,1,2,3])
            self.assertEqual(fixed['selections'][:2],response['selections'])

    def test_duplicate_or_absent_symbols_are_repaired(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = RecoveryController(self.config(tmp),'mlestar')
            response = {'selections':[{'trajectory':0,'block':'loss','reason':'duplicate'}]*2+
                        [{'trajectory':1,'block':'missing','reason':'wrong'},
                         {'trajectory':2,'block':'loss','reason':'valid'}]}
            prompt = 'Select one block\n'+json.dumps({'source':['def loss():\n pass\n']*4})
            def answer(instruction,out,search=False,key=None):
                return {'trajectory':out['properties']['trajectory']['enum'][0],
                        'block':'hyperparameters','reason':'repaired'}
            with patch.object(FamilyController,'llm',side_effect=answer) as call:
                fixed = c.complete_selection(prompt,response,'select-2')
            self.assertEqual(call.call_count,3)
            self.assertEqual(fixed['selections'][2],response['selections'][3])

    def test_missing_cli_launch_is_retryable_and_does_not_charge_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = RecoveryController(self.config(tmp),'mlestar')
            with patch.object(FamilyController,'run_attempt',side_effect=FileNotFoundError('codex')):
                with self.assertRaisesRegex(RuntimeError,'CLI launch failed'):
                    c.run_attempt(Path(tmp),'fixture',schema({'x':TEXT}),False)
            self.assertEqual(read(c.ledger.path)['spent_gpu_seconds'],0)

    def test_accounting_audit_rejects_unsettled_or_missing_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = RecoveryController(self.config(tmp),'mlestar')
            self.receipt(c,'old',cost=12)
            state = read(c.ledger.path)
            state.update(spent_gpu_seconds=12,settled={'old':{'allocated_gpu_seconds':12}})
            save(c.ledger.path,state)
            job = {'root':tmp,'method':'mlestar'}
            self.assertEqual(audited_budget(job)['spent_gpu_seconds'],12)
            state['reservations']={'active':30};save(c.ledger.path,state)
            with self.assertRaisesRegex(RuntimeError,'unsettled'):
                audited_budget(job)


if __name__=='__main__':unittest.main()
