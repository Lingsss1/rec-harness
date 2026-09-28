import copy
import unittest
from pathlib import Path
from rec_harness.core import ContractError,digest,read_json,write_json
from rec_harness.evolution import STAGES,numerical_gate_failures
from rec_harness.autonomous import autoresearch
from rec_harness.store import Store
from rec_harness.tool_history import collect
import test_evolution_auto


class ToolGateFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_evolution_auto.EvolutionAutomationTests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    def test_exact_thresholds_pass_and_individual_failures_remain_distinct(self):
        rule={'minimum_gain':0.5,'maximum_regression':0.1,'maximum_total_cost':8}
        report={'gain':0.5,'regression':0.1,'total_cost':8}
        self.assertEqual(numerical_gate_failures(report,rule),[])
        for field,value,expected in [('gain',0.4,'minimum_gain'),('regression',0.2,'maximum_regression'),('total_cost',9,'maximum_total_cost')]:
            self.assertEqual(numerical_gate_failures({**report,field:value},rule),[expected])

    def test_rejection_history_and_graph_retain_all_failed_gates_without_scores(self):
        config=read_json(self.fixture.config());config.update(kind='tool',context={})
        rule=copy.deepcopy(config['plan']['stages']['prefix_audit'])
        rule.update(minimum_gain=2,maximum_total_cost=3)
        config['plan']['stages']={stage:copy.deepcopy(rule) for stage in STAGES['tool']}
        path=self.fixture.root/'tool.json';write_json(path,config)
        class Author:
            def run(self,role,context,directory):
                model=Path(directory)/'model';model.mkdir(parents=True)
                (model/'diagnostic.py').write_text('def measure(request): return {}\n')
                return {'changed_files':['diagnostic.py']+test_evolution_auto.tool_definition(context,directory)}
        result=autoresearch(self.fixture.campaign,path,agent=Author())
        self.assertEqual(result['status'],'rejected');self.assertEqual(result['rejection_kind'],'validation_gate')
        expected=['minimum_gain','maximum_total_cost']
        history,_=collect([{'campaign':str(self.fixture.campaign)}])
        row=history['candidates'][0]['validation'][0]
        self.assertEqual(row['failed_gates'],expected)
        self.assertNotIn('report',row);self.assertNotIn('gain',row);self.assertNotIn('total_cost',row)
        store=Store(self.fixture.campaign/'memory');self.addCleanup(store.close)
        event=next(e['payload'] for e in store.events() if e['kind']=='validation_rejected')
        self.assertEqual(event['failed_gates'],expected)
        graph=store.snapshot();node=next(n for n in graph['nodes'] if n['type']=='ToolValidation')
        self.assertEqual(node['attrs']['failed_gates'],expected)
        self.assertFalse(any(n['type']=='Claim' for n in graph['nodes']))
        before=store.verify()
        self.assertEqual(autoresearch(self.fixture.campaign,agent=object()),result)
        self.assertEqual(store.verify(),before)
        # A later committed record cannot claim a different reason than its
        # report and frozen thresholds, even if its event hash is valid.
        store.event('validation_rejected',{**event,'failed_gates':['maximum_regression']})
        with self.assertRaisesRegex(ContractError,'failed gates differ'):
            collect([{'campaign':str(self.fixture.campaign)}])
