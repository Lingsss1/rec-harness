"""Final top-three selection, end-of-search trigger and closed-test boundaries."""
import copy
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from common import read, write, sha
from selection import POLICY, TASKS, aggregate
from assess_topk import ranked_unique, selection, topic_representatives, worker, watch, wait_for_gpus


class TopThreeAssessment(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'source').mkdir()
        (self.root / 'source/train.py').write_text('# host identity fixture\n')
        self.protocol = {'identity': 'p', 'data_identity': 'd', 'selection_policy': POLICY,
                         'primary': 'four-task macro ROC-AUC',
                         'source_sha256': {'train.py': sha(self.root / 'source/train.py')}}
        write(self.root / 'protocol.json', self.protocol)
        write(self.root / 'budget.json', {'gpus': [0, 1, 2, 3], 'seconds': 43200, 'deadline_unix': time.time() + 43200})
        self.din = self.row('baseline-din', [.6, .7, .7, .7], baseline=True)
        self.a = self.row('balanced-a', [.61, .82, .82, .82])
        self.b = self.row('balanced-b', [.62, .8, .8, .8])
        self.c = self.row('balanced-c', [.63, .79, .79, .79])
        self.click = self.row('click-only-winner', [.95, .4, .4, .4])
        self.click['topic_id'] = self.a['topic_id']
        write(self.root / 'references.json', {'trials': [self.din]})
        self.states([self.a, self.b, self.c, self.click])
        write(self.root / 'status.json', {'phase': 'topic_research_stopped'})

    def tearDown(self):
        self.temp.cleanup()

    def row(self, name, aucs, baseline=False):
        folder = self.root / name
        folder.mkdir()
        (folder / 'model.py').write_text('# source fixture ' + name + '\n')
        (folder / 'checkpoint.pt').write_bytes(('checkpoint identity fixture ' + name).encode())
        per_task = {task: {'auc': value, 'logloss': .2} for task, value in zip(TASKS, aucs)}
        row = {'id': name, 'status': 'success', 'role': 'baseline' if baseline else 'candidate',
               'model_path': str(folder / 'model.py'), 'source_sha256': sha(folder / 'model.py'),
               'topic_id': 'topic-' + name, 'round': 0,
               'output': str(folder), 'config': {}, 'seed': 42, 'wall_seconds': 1,
               'metrics': {'validation': {'per_task': per_task, **aggregate(per_task)},
                           'parameters': 100, 'checkpoint_sha256': sha(folder / 'checkpoint.pt'),
                           'effective_config': {}, 'data_identity': 'd', 'protocol_hash': 'p', 'train_seconds': 1}}
        if baseline:
            row['baseline_kind'] = 'din'
        return row

    def states(self, rows, phase='research_stopped'):
        for gpu in range(4):
            write(self.root / f'topics/gpu-{gpu}/state.json', {'phase': phase,
                  'trials': rows if gpu == 0 else [], 'completed_research_rounds': 1,
                  'round': max((r['round'] for r in rows), default=-1) + 1,
                  'topics': [{'id': t, 'title': t} for t in sorted({r['topic_id'] for r in rows})] if gpu == 0 else []})

    def test_top_three_use_macro_not_stale_alias_or_click(self):
        self.a['metrics']['validation']['auc'] = .01
        self.click['metrics']['validation']['auc'] = .99
        self.states([self.a, self.b, self.c, self.click])
        freeze = selection(self.root)
        self.assertEqual([r['id'] for r in freeze['suite'][:3]], ['balanced-a', 'balanced-b', 'balanced-c'])
        self.assertEqual(freeze['suite'][3]['id'], 'baseline-din')
        self.assertFalse(freeze['test_used_for_selection'])
        self.assertFalse((self.root / 'data').exists())

    def test_duplicates_require_code_config_and_weights_all_equal(self):
        duplicate = copy.deepcopy(self.a)
        duplicate['id'] = 'z-duplicate'
        different_function = copy.deepcopy(self.a)
        different_function['id'] = 'different-function'
        different_function['source_sha256'] = 'different code hash'
        ranked, duplicates = ranked_unique([self.a, duplicate, different_function])
        self.assertEqual(len(ranked), 2)
        self.assertEqual(duplicates, [{'id': 'z-duplicate', 'same_model_as': 'balanced-a'}])

    def test_selection_happens_after_end_and_includes_late_improvement(self):
        write(self.root / 'status.json', {'phase': 'topic_research'})
        with self.assertRaisesRegex(ValueError, 'finish normally'):
            selection(self.root)
        late = self.row('late-best', [.8, .9, .9, .9])
        self.states([self.a, self.b, self.c, late])
        write(self.root / 'status.json', {'phase': 'topic_research_stopped'})
        self.assertEqual(selection(self.root)['suite'][0]['id'], 'late-best')

    def test_watch_does_not_assess_while_running_and_respects_stop(self):
        assessment = self.root / 'assessment'
        write(self.root / 'status.json', {'phase': 'topic_research'})
        with patch('assess_topk.assess') as run, patch('assess_topk.time.sleep', side_effect=InterruptedError):
            with self.assertRaises(InterruptedError):
                watch(self.root, assessment, 3, True)
            run.assert_not_called()
        self.assertEqual(read(assessment / 'status.json')['phase'], 'waiting_for_research_completion')
        (self.root / 'STOP').touch()
        with patch('assess_topk.assess') as run:
            watch(self.root, assessment, 3, True)
            run.assert_not_called()
        self.assertEqual(read(assessment / 'status.json')['phase'], 'cancelled_with_research')

    def test_test_worker_requires_validation_gate_before_start(self):
        assessment = self.root / 'assessment'
        write(assessment / 'selection-freeze.json', selection(self.root))
        write(assessment / 'validation-replay-gate.json', {'status': 'failed', 'freeze_identity': 'wrong'})
        with self.assertRaisesRegex(ValueError, 'validation replay'):
            worker(self.root, assessment, 'rank_1', 'test')
        self.assertFalse((assessment / 'rank_1/test/worker-start.json').exists())

    def test_changed_weights_abort_selection(self):
        (Path(self.a['output']) / 'checkpoint.pt').write_bytes(b'changed fixture')
        with self.assertRaisesRegex(ValueError, 'source/checkpoint changed'):
            selection(self.root)

    def test_watcher_triggers_once_after_all_children_exit(self):
        assessment = self.root / 'assessment'
        with patch('assess_topk.live_research_children', return_value=[]), patch('assess_topk.assess') as run:
            watch(self.root, assessment, 3, True)
            run.assert_called_once_with(self.root, assessment, 3, True)

    def test_busy_gpu_delays_inference(self):
        assessment = self.root / 'assessment'
        response = type('Response', (), {})()
        response.stdout = '0, 4096\n1, 2\n'
        free = type('Response', (), {})()
        free.stdout = '0, 2\n1, 2\n'
        with patch('assess_topk.subprocess.run', side_effect=[response, free]), patch('assess_topk.time.sleep') as sleep:
            wait_for_gpus([0, 1], assessment)
            sleep.assert_called_once_with(10)

    def test_stop_during_gpu_wait_cancels_before_any_inference(self):
        assessment = self.root / 'assessment'
        response = type('Response', (), {})()
        response.stdout = '0, 4096\n'
        with patch('assess_topk.subprocess.run', return_value=response), patch('assess_topk.time.sleep',
                side_effect=lambda seconds: (self.root / 'STOP').touch()):
            with self.assertRaisesRegex(ValueError, 'cancelled'):
                wait_for_gpus([0], assessment, self.root)

    def test_only_one_final_representative_per_topic(self):
        second = self.row('same-topic-second', [.7, .85, .85, .85])
        first = self.row('same-topic-first', [.71, .86, .86, .86])
        second['topic_id'] = first['topic_id'] = self.a['topic_id']
        self.states([self.a, self.b, self.c, first, second])
        frozen = selection(self.root)
        self.assertEqual([r['id'] for r in frozen['suite'][:3]], ['same-topic-first', 'balanced-b', 'balanced-c'])
        self.assertEqual(len({r['topic_id'] for r in frozen['suite'][:3]}), 3)

    def test_round_trace_retains_best_and_excludes_unfinished_round(self):
        weaker = self.row('later-weaker', [.4, .6, .6, .6])
        pending = self.row('unfinished-stronger', [.9, .99, .99, .99])
        weaker.update(topic_id=self.a['topic_id'], round=1)
        pending.update(topic_id=self.a['topic_id'], round=2)
        self.states([self.a, weaker, pending, self.b, self.c])
        state = read(self.root / 'topics/gpu-0/state.json')
        state['round'] = 2
        representatives, summaries, trace = topic_representatives([state], [self.din])
        points = [x for x in trace if x['topic_id'] == self.a['topic_id']]
        self.assertEqual([x['best_trial_id'] for x in points], [self.a['id'], self.a['id']])
        self.assertNotIn(pending['id'], {x['id'] for x in representatives})

    def test_topic_keeps_entry_model_when_new_variants_regress(self):
        weaker = self.row('regressed-entry', [.4, .6, .6, .6])
        weaker['topic_id'] = 'new-topic'
        state = {'round': 1, 'trials': [weaker], 'topics': [
            {'id': 'new-topic', 'title': 'new topic', 'entry_trial_id': self.a['id'], 'import_trial_refs': [self.a['id']]}]}
        representatives, summaries, trace = topic_representatives([state], [self.din, self.a])
        self.assertEqual(representatives[0]['id'], self.a['id'])
        self.assertEqual(trace[0]['best_trial_id'], self.a['id'])
        self.assertEqual(summaries[0]['gain_vs_entry'], 0)

    def test_identical_models_representing_two_topics_are_evaluated_once(self):
        duplicate = copy.deepcopy(self.a)
        duplicate.update(id='z-same-model', topic_id='another-topic')
        self.states([self.a, self.b, self.c, duplicate])
        frozen = selection(self.root)
        first = frozen['suite'][0]
        self.assertEqual(first['id'], self.a['id'])
        self.assertEqual(set(first['topic_ids']), {self.a['topic_id'], 'another-topic'})


if __name__ == '__main__':
    unittest.main()
