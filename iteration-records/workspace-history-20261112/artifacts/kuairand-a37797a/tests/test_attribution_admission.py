"""Synthetic execution tests, not scientific attribution evidence."""
import sys
from unittest.mock import patch
import json
import tempfile
import unittest
from pathlib import Path
from rec_harness.core import ContractError
from rec_harness.evolution import Evolution, validation_stages
from rec_harness.attribution import validate


def report():
    return {'passed':True,'implementation_hash':'fixture','evaluation_ids':['source'],
        'total_cost':1,'identity':{},'attribution':{
        'gap_id':'gap','before':{'scale':'unresolved','shape':'unresolved'},
        'after':{'scale':'refuted','shape':'unresolved'},
        'observations':[{'evaluation_id':'source','measurement':'self_difference','value':0,
            'explanation_ids':['scale'],'interpretation':'Fixture interpretation only'}],
        'controls':[{'name':'self','passed':True,'observed':0,'expected':0}],
        'decision_before':'adjust scale','decision_after':'inspect shape',
        'rationale':'Fixture measurement narrows the decision','limitations':['synthetic fixture; no causal proof']}}


class AttributionTests(unittest.TestCase):
    def setUp(self):
        if sys.platform == 'darwin':
            # Linux process-recovery checks are outside this admission test.
            for target in ['rec_harness.agents.identity','rec_harness.process_identity.identity']:
                p=patch(target,side_effect=lambda pid:{'pid':int(pid),'observable':False})
                p.start();self.addCleanup(p.stop)

    def test_rejects_claim_only_unchanged_and_unbound_reports(self):
        for mutate in [lambda r:r.update(implementation_hash='other'),
                       lambda r:r['attribution'].update(observations=[]),
                       lambda r:r['attribution'].update(after=r['attribution']['before']),
                       lambda r:r['attribution']['controls'][0].update(passed=False),
                       lambda r:r['attribution']['observations'][0].update(evaluation_id='other')]:
            r=report();mutate(r)
            with self.assertRaises(ContractError):validate(r,{'implementation_hash':'fixture'})

    def test_local_offline_measurement_activates_without_independent_utility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);validator=root/'validate.py'
            # The validator executes the candidate on a fixed input without training.
            validator.write_text('import json,sys,runpy\nfrom pathlib import Path\n'
                'candidate=Path(sys.argv[1]); result=runpy.run_path(str(candidate/"diagnostic.py"))["measure"]([2,2])\n'
                'r='+repr(report())+'\n'
                'r["attribution"]["observations"][0]["value"]=result\n'
                'r["attribution"]["controls"][0]["observed"]=result\n'
                'r["attribution"]["controls"][0]["passed"]=result==0\n'
                'r["implementation_hash"]=json.loads(Path(sys.argv[3]).read_text())["implementation_hash"]\n'
                'Path(sys.argv[2]).write_text(json.dumps(r))\n')
            identity={k:'fixture' for k in ['base_llm','tools','start','graph','budget','protocol']}
            # Match the immutable identity in the validator report.
            validator.write_text(validator.read_text().replace("'identity': {}",repr('identity')+': '+repr(identity)))
            plan={'validation_basis':'attribution_increment','budget':10,'agent_cost':0,
                'tool_contract':'public-aggregate-measurement-v1','development_ids':['source'],'identity':identity,'rollback_version':'base',
                'stages':{'attribution':{'command':['{python}',str(validator),'{candidate}','{output}',str(root/'campaign/candidates/tool/candidate.json')],
                    'timeout':10,'cost':1,'maximum_total_cost':10}}}
            from rec_harness.project_tool_plans import freeze
            scheduled={'tool':{'plan':plan},'tool_followup_plans':[],'max_tool_updates':1}
            freeze(scheduled)
            class Agent:
                def run(self,role,context,directory):
                    p=Path(directory)/'model';p.mkdir(parents=True,exist_ok=True)
                    (p/'diagnostic.py').write_text('def measure(x): return max(x)-min(x)\n')
                    return {'changed_files':['diagnostic.py']}
            evolution=Evolution(root/'campaign')
            try:
                result=evolution.run('tool','tool',{},plan,agent=Agent())
                self.assertEqual(result['status'],'promoted',result)
                self.assertIsNotNone(evolution._active('tool'))
                evolution.verify()
                from rec_harness.diagnostics import freeze_tool
                from rec_harness.core import Artifacts
                registered=freeze_tool({'registry':str(root/'campaign'),'identity':identity,
                    'cost':1,'timeout':10,'execution':{'recipe':{},'python':sys.executable,'uid_registry':'fixture'}},Artifacts(root/'consumer'))
                self.assertEqual(registered['registration']['validation_basis'],'attribution_increment')
                self.assertEqual(set(registered['registration']['validation_stages']),{'attribution'})
                events=evolution.store.events()
                self.assertEqual([e['payload']['stage'] for e in events if e['kind']=='validation_passed'],['attribution'])
                self.assertEqual(evolution.run('tool','tool',{},plan,agent=Agent()),result)
            finally:evolution.store.close()

    def test_harness_cannot_skip_existing_gates(self):
        with self.assertRaises(ContractError):validation_stages('harness',{'validation_basis':'attribution_increment'})
        self.assertEqual(validation_stages('tool',{}),['correctness','synthetic','robustness','shadow','utility','cost_regression'])
