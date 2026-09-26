import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from rec_harness.core import Artifacts, ContractError, canonical, write_json
from rec_harness.legacy import import_legacy
from rec_harness.runtime import node
from rec_harness.store import Store


class LegacyMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'legacy'; self.source.mkdir()
        self.store = Store(self.root / 'campaign/memory'); self.addCleanup(self.store.close)

    def import_all(self):
        return import_legacy(self.store, self.source, 'task', 'unknown')

    def populate(self):
        (self.source / 'results.tsv').write_text('experiment_id\tprimary_metric\tstatus\nexp-1\t0.4\tkeep\n')
        (self.source / 'episodes.jsonl').write_text(canonical({'experiment_id': 'exp-1',
            'metrics': {'primary': {'name': 'HR@10', 'value': 0.5}, 'guardrails': {'runtime_sec': 12}},
            'validity': {'valid': True}, 'budget': {'seeds': [1, 2]}, 'decision': 'keep'}) + '\n')
        (self.source / 'failures.jsonl').write_text(canonical({'experiment_id': 'exp-2', 'kind': 'oom',
            'signature': 'allocation failed', 'reproduction': 'python train.py --batch 2048',
            'avoid': 'reduce batch on constrained memory'}) + '\n')
        write_json(self.source / 'lessons.json', {'schema_version': 1, 'lessons': [{'lesson_id': 'lesson-1',
            'statement': 'method works', 'status': 'confirmed', 'confidence': 0.99,
            'supporting_episode_ids': ['exp-1'], 'counterevidence_episode_ids': ['exp-2', 'missing'],
            'scope': {'dataset': 'source'}}]})
        write_json(self.source / 'strategy.json', {'phase': 'exploit', 'incumbent': {'experiment_id': 'exp-1'}})

    def test_typed_schema_deduplicates_runs_and_retains_conflicting_summaries(self):
        self.populate(); first = self.import_all()
        nodes = self.store.snapshot()['nodes']
        experiments = [n for n in nodes if n['type'] == 'Experiment']
        self.assertEqual(len(experiments), 2)
        self.assertTrue(all(n['attrs']['valid'] is False and n['attrs']['independent_run_id'] is None for n in experiments))
        observations = [n for n in nodes if n['type'] == 'Observation']
        self.assertEqual(len(observations), 2)
        self.assertEqual({next(iter(n['attrs']['metrics'].values())) for n in observations}, {0.4, 0.5})
        claim = next(n for n in nodes if n['type'] == 'Claim')
        self.assertEqual(claim['attrs']['status'], 'legacy_unverified')
        self.assertEqual(claim['attrs']['reported_status'], 'confirmed')
        self.assertEqual(claim['attrs']['unresolved_external_refs'], ['missing'])
        self.assertEqual(len(claim['attrs']['evidence_refs']), 2)
        failure = next(n for n in nodes if n['type'] == 'Failure')
        self.assertEqual(failure['attrs']['reproduction'], 'python train.py --batch 2048')
        self.assertTrue(failure['attrs']['execution_evidence_only'])
        self.assertFalse(next(n for n in nodes if n['type'] == 'Context')['attrs']['historical_evidence'])
        before = self.store.verify()
        again = self.import_all()
        self.assertEqual(again['imported'], 0); self.assertEqual(again['typed_nodes_added'], 0)
        self.assertEqual(again['typed_nodes_updated'], 0); self.assertEqual(again['relations_added'], 0)
        self.assertEqual(self.store.verify(), before)
        with self.assertRaisesRegex(ContractError, 'independent'):
            self.store.patch({'base_version': self.store.version, 'nodes': [node('laundered', 'Claim', {
                'status': 'supported', 'scope': 'task', 'evidence_refs': [n['id'] for n in experiments]}, 'task', 'unknown')], 'edges': []})

    def test_exact_original_files_remain_recoverable_after_source_changes(self):
        self.populate(); original = (self.source / 'lessons.json').read_bytes(); self.import_all()
        manifest = next(e['payload'] for e in self.store.events() if e['kind'] == 'legacy_sources_archived')
        write_json(self.source / 'lessons.json', {'lessons': []})
        self.import_all()
        archive = Artifacts(self.root / 'campaign/artifacts/objects')
        self.assertEqual(archive.verify(manifest['files']['lessons.json']).read_bytes(), original)
        self.assertEqual(sum(n['type'] == 'Claim' for n in self.store.snapshot()['nodes']), 1)

    def test_reported_parent_sources_and_conflicts_survive_incremental_import(self):
        episodes=[{'experiment_id':'parent-a'},{'experiment_id':'parent-b'},
                  {'experiment_id':'child','parent_experiment_id':'parent-a'}]
        (self.source/'episodes.jsonl').write_text(''.join(canonical(row)+'\n' for row in episodes))
        self.import_all()
        def parents():
            graph=self.store.snapshot();nodes={n['id']:n for n in graph['nodes']}
            return nodes,[e for e in graph['edges'] if e['relation']=='DERIVED_FROM' and nodes[e['source']]['type']=='Experiment']
        nodes,relations=parents();self.assertEqual(len(relations),1)
        self.assertEqual(len(relations[0]['attrs']['legacy_source_record_ids']),1)
        (self.source/'results.tsv').write_text('experiment_id\tparent_experiment_id\nchild\tparent-a\nchild\tparent-b\n')
        result=self.import_all();self.assertEqual(result['relations_updated'],1)
        nodes,relations=parents();self.assertEqual(len(relations),2)
        self.assertEqual(sorted(len(e['attrs']['legacy_source_record_ids']) for e in relations),[1,2])
        for relation in relations:
            self.assertEqual(relation['attrs']['status'],'legacy_unverified')
            self.assertEqual(relation['attrs']['evidence_refs'],[])
            for reference in relation['attrs']['legacy_source_record_ids']:
                self.assertEqual(nodes[reference]['type'],'LegacyRecord')
                self.assertEqual(nodes[reference]['attrs']['original']['experiment_id'],'child')
        before=self.store.verify();again=self.import_all()
        self.assertEqual(again['relations_updated'],0);self.assertEqual(self.store.verify(),before)

    def test_late_references_resolve_without_fabricated_experiments(self):
        write_json(self.source / 'lessons.json', [{'lesson': 'tentative', 'supporting_episode_ids': ['later']}])
        self.import_all()
        first_version = self.store.version
        self.assertFalse(any(n['type'] == 'Experiment' for n in self.store.snapshot()['nodes']))
        (self.source / 'episodes.jsonl').write_text(canonical({'experiment_id': 'later', 'metrics': {'quality': 0.3}}) + '\n')
        result = self.import_all()
        self.assertEqual(result['typed_nodes_updated'], 1)
        old = next(n for n in self.store.snapshot(first_version)['nodes'] if n['type'] == 'Claim')
        current = next(n for n in self.store.snapshot()['nodes'] if n['type'] == 'Claim')
        self.assertEqual(old['attrs']['unresolved_external_refs'], ['later'])
        self.assertEqual(current['attrs']['unresolved_external_refs'], [])
        self.assertEqual(current['attrs']['status'], 'legacy_unverified')

    def test_concurrent_source_change_cannot_mismatch_archived_bytes_and_projection(self):
        path = self.source / 'lessons.json'
        write_json(path, [{'lesson': 'before'}]); original = path.read_bytes()
        original_put = Artifacts.put
        def change_source(archive, frozen):
            write_json(path, [{'lesson': 'after'}])
            return original_put(archive, frozen)
        with patch.object(Artifacts, 'put', autospec=True, side_effect=change_source):
            self.import_all()
        raw = next(n for n in self.store.snapshot()['nodes'] if n['type'] == 'LegacyRecord')
        self.assertEqual(raw['attrs']['original']['lesson'], 'before')
        archive = Artifacts(self.root / 'campaign/artifacts/objects')
        self.assertEqual(archive.verify(raw['attrs']['source_file_ref']).read_bytes(), original)

    def test_hidden_record_rejected_and_cross_task_identities_do_not_collide(self):
        write_json(self.source / 'lessons.json', [{'lesson': 'x', 'hidden_score': 1}])
        with self.assertRaises(ContractError): self.import_all()
        self.assertEqual(self.store.version, 0)
        write_json(self.source / 'lessons.json', [{'lesson': 'public'}]); self.import_all()
        other = import_legacy(self.store, self.source, 'other-task', 'unknown')
        self.assertEqual(other['imported'], 1)
        self.assertEqual({n['attrs']['task_id'] for n in self.store.snapshot()['nodes']}, {'task', 'other-task'})


if __name__ == '__main__':
    unittest.main()
