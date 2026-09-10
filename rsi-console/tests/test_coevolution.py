"""Offline lifecycle integration: no training data or model dependencies."""
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       'skills' / 'model-diagnostic-tool-evolver' / 'scripts'))
import coevolve as loop
import manage_candidate as manager


class CoevolutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name)
        self.file('source.json', {'observation': 'new temporal layer'})
        self.file('measurements.json', {'delta': 0.3, 'control': 0.0})
        self.spec = {key: key for key in ('trigger_id', 'observation', 'component',
                    'gap', 'decision', 'baseline', 'budget', 'measurement',
                    'null_behavior', 'action', 'falsification')}
        self.spec.update(kind='structural_change', source='source.json',
                         code_identity='code-1', protocol_id='protocol-1')
        self.proposal = self.file('proposal.json', self.spec)
        self.id = loop.propose(self.args(input=self.proposal))['candidate_id']
        self.report = self.file('report.json', dict(code_identity='code-1',
            protocol_id='protocol-1', split='validation', conclusion='supported',
            finding='signal recovered', evidence='measurements.json'))

    def args(self, **kwargs):
        return argparse.Namespace(workspace=self.ws, id=getattr(self, 'id', None), **kwargs)

    def file(self, name, value):
        path = self.ws / name
        path.write_text(json.dumps(value))
        return path

    def accept(self):
        manager.command_paper(self.args(title='test method', url='https://example.org/method',
            version='1', code_url=None, license=None, role='chosen'))
        manager.command_precommit(self.args(opportunity_score=8,
            thresholds=self.file('thresholds.json', {'signal_min': 0.2}),
            input_schema='array', output_schema='delta', null_behavior='no edit',
            action_mapping='test a bounded change if delta > .2'))
        self.transition('researched')
        self.transition('implemented')
        for gate in manager.GATES:
            self.file(gate + '.json', {'test': gate, 'passed': True})
            manager.command_gate(self.args(gate=gate, result='pass',
                evidence=gate + '.json', note='offline fixture'))
        for status in ('synthetic_pass', 'shadow', 'accepted'):
            self.transition(status)

    def transition(self, target):
        return manager.command_transition(self.args(to=target, evidence='source.json', note='test'))

    def handoff(self):
        return loop.handoff(self.args(input=self.report))

    def task_args(self, task, **kwargs):
        return self.args(task_id=task['task_id'], code_identity='code-1',
                         protocol_id='protocol-1', **kwargs)

    def test_complete_loop_and_replay(self):
        self.assertTrue(loop.propose(self.args(input=self.proposal))['duplicate'])
        with self.assertRaises(ValueError):
            self.handoff()
        self.accept()
        task = self.handoff()
        self.assertEqual(task, self.handoff())
        self.assertEqual(loop.check(self.task_args(task))['action'], self.spec['action'])
        receipt = self.file('episode.json', dict(episode_id='episode-1',
            task_id=task['task_id'], outcome='discard', parent_code_identity='code-1',
            result_code_identity='trial-1', protocol_id='protocol-1', evidence='measurements.json'))
        self.assertEqual(loop.close(self.task_args(task, input=receipt))['status'], 'completed')
        self.assertEqual(self.handoff()['status'], 'completed')
        with self.assertRaises(ValueError):
            loop.check(self.task_args(task))
        events = [json.loads(line)['event'] for line in manager.registry_path(self.ws).read_text().splitlines()]
        self.assertEqual(events.count('edit_queued'), 1)
        self.assertEqual(events.count('edit_evaluated'), 1)

    def test_both_trigger_types(self):
        self.spec['kind'] = 'user_issue'
        result = loop.propose(self.args(input=self.file('user.json', self.spec)))
        self.assertNotEqual(result['candidate_id'], self.id)

    def test_missing_or_failed_gate_blocks_acceptance(self):
        state = manager.read_state(self.ws, self.id)
        for result in ('pending', 'fail'):
            state['validation']['cost']['result'] = result
            with self.assertRaises(ValueError):
                manager.require_transition_gates(state, 'accepted')

    def test_null_test_and_stale_reports_block_edits(self):
        self.accept()
        original = loop.read(self.report)
        for key, value in (('conclusion', 'null'), ('conclusion', 'inconclusive'),
                           ('conclusion', 'contradicted'), ('split', 'test'),
                           ('code_identity', 'code-2'), ('protocol_id', 'protocol-2')):
            with self.subTest(key=key, value=value):
                self.file('report.json', {**original, key: value})
                with self.assertRaises(ValueError):
                    self.handoff()

    def test_changed_gate_evidence_blocks_handoff(self):
        self.accept()
        self.file('cost.json', {'tampered': True})
        with self.assertRaises(ValueError):
            self.handoff()

    def test_evidence_and_context_rechecked_before_edit(self):
        self.accept()
        task = self.handoff()
        args = self.task_args(task)
        args.code_identity = 'new-code'
        with self.assertRaises(ValueError):
            loop.check(args)
        self.file('measurements.json', {'delta': 99})
        with self.assertRaises(ValueError):
            loop.check(self.task_args(task))

    def test_superseded_candidate_blocks_queued_edit(self):
        self.accept()
        task = self.handoff()
        self.transition('superseded')
        with self.assertRaises(ValueError):
            loop.check(self.task_args(task))

    def test_terminal_gates_immutable_and_missing_source_rejected(self):
        self.accept()
        with self.assertRaises(ValueError):
            manager.command_gate(self.args(gate='cost', result='fail', evidence='cost.json', note='change'))
        self.spec['source'] = '../missing.json'
        with self.assertRaises(ValueError):
            loop.propose(self.args(input=self.file('bad.json', self.spec)))

    def test_mismatched_episode_does_not_close(self):
        self.accept()
        task = self.handoff()
        receipt = self.file('episode.json', dict(episode_id='episode-1', task_id='other-task',
            outcome='keep', parent_code_identity='code-1', result_code_identity='trial-1',
            protocol_id='protocol-1', evidence='measurements.json'))
        with self.assertRaises(ValueError):
            loop.close(self.task_args(task, input=receipt))
        self.assertEqual(loop.check(self.task_args(task))['status'], 'ready')


if __name__ == '__main__':
    unittest.main()
