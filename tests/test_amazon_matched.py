import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from rec_harness.core import ContractError,read_json,write_json
from rec_harness.store import Store
from rec_harness.adapters import amazon_matched
from rec_harness.adapters.amazon_matched import run_matched_development


class MatchedDevelopmentTests(unittest.TestCase):
    def test_interrupted_scoring_resume_terminal_verification_and_binding_rejection(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            study=Path(temp)/'study';out=Path(temp)/'output'
            plan={'proposal_id':'p','baseline':'base','handoff_hash':'h','reused_control_versions':['base'],
                  'versions':{'candidate':{'model_version':'candidate','source':{'sha256':'c'}},
                              'controls':[{'model_version':'base','source':{'sha256':'b'}}]}}
            store=Store(study/'memory');store.event('amazon_research_plan_registered',plan);store.close()
            cfg={'protocol_hash':'protocol','model_versions':{'base':{'sha256':'b'},'candidate':{'sha256':'c'}},
                 'datasets':['tiny'],'development_seeds':[42]}
            module='rec_harness.adapters.amazon_matched.'
            stack.enter_context(patch(module+'load_config',return_value=cfg))
            stack.enter_context(patch(module+'resolve_execution',return_value={'recipe':{'runtime':'frozen'}}))
            train=stack.enter_context(patch(module+'run_study'))
            phase=stack.enter_context(patch(module+'run_phase'))
            check=stack.enter_context(patch(module+'prepare_phase'))
            stack.enter_context(patch(module+'development_readout',return_value={'hash':'readout'}))
            launches=[];resume_flags=[];interrupt=True
            def score(study,version,dataset,seed,directory,recipe,uids,*,resume):
                nonlocal interrupt
                resume_flags.append((version,resume))
                if version=='candidate' and interrupt:
                    interrupt=False
                    raise RuntimeError('interrupted before candidate scoring')
                if not (directory/'inference').exists():
                    (directory/'inference').mkdir(parents=True);launches.append(version)
                return {'rank_disagreements':0,'hash':version+'-receipt'}
            scorer=stack.enter_context(patch(module+'rescore_job',side_effect=score))
            def run(execution='exec'):
                return run_matched_development(study,'p',out,execution_id=execution,recipe={'runtime':'frozen'},uid_registry=Path(temp)/'uids')
            with self.assertRaisesRegex(RuntimeError,'interrupted'):run()
            self.assertFalse((out/'acceptance.json').exists())
            result=run();self.assertEqual(launches,['base','candidate'])
            self.assertIn(('base',True),resume_flags)
            self.assertEqual(result['codex_calls'],0)
            train.reset_mock();phase.reset_mock()
            self.assertEqual(run(),result);train.assert_not_called();phase.assert_not_called()
            self.assertEqual(launches,['base','candidate']);check.assert_called_once()
            scorer.reset_mock()
            with self.assertRaisesRegex(ContractError,'binding changed'):run('changed-execution')
            scorer.assert_not_called();train.assert_not_called()
            accepted=read_json(out/'acceptance.json');accepted['codex_calls']=7
            write_json(out/'acceptance.json',accepted)
            with self.assertRaisesRegex(ContractError,'acceptance changed'):run()
            scorer.assert_not_called()
            write_json(out/'acceptance.json',result)
            (out/'rescores/base-tiny-42/inference').rmdir()
            with self.assertRaisesRegex(ContractError,'inference archive missing'):run()
            train.assert_not_called();scorer.assert_not_called()

    def test_rank_disagreement_cannot_publish_success(self):
        with tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
            study=Path(temp)/'study';output=Path(temp)/'output'
            plan={'proposal_id':'p','baseline':'b','reused_control_versions':['b'],
                  'versions':{'candidate':{'model_version':'c','source':{'sha256':'c'}},
                              'controls':[{'model_version':'b','source':{'sha256':'b'}}]}}
            store=Store(study/'memory');store.event('amazon_research_plan_registered',plan);store.close()
            cfg={'protocol_hash':'p','model_versions':{'b':{'sha256':'b'},'c':{'sha256':'c'}},'datasets':['tiny'],'development_seeds':[42]}
            for name,value in [('load_config',cfg),('resolve_execution',{'recipe':{}}),('run_study',{}),('run_phase',{}),('rescore_job',{'rank_disagreements':1})]:
                stack.enter_context(patch.object(amazon_matched,name,return_value=value))
            with self.assertRaisesRegex(ContractError,'rankings disagree'):
                run_matched_development(study,'p',output,execution_id='e',recipe={},uid_registry=Path(temp)/'uids')
            self.assertFalse((output/'acceptance.json').exists())
