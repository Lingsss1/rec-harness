import tempfile
import fcntl
import unittest
from pathlib import Path

from rec_harness.core import ContractError, canonical, digest, read_json, write_json
from rec_harness.memory_policy import export_memory, import_memory, validate_package
from rec_harness.runtime import node, edge
from rec_harness.store import Store


class MemoryPolicyTests(unittest.TestCase):
    def test_source_lineage_and_unverified_status_survive_bounded_transfer(self):
        from rec_harness.method_sources import graph_records
        self.assertEqual(self.export()['schema'],'memory-policy-v3')
        method={'origin':'paper_claim','sources':['https://example.org/fictional-paper.pdf'],'mechanism':'fixture'}
        nodes,edges=graph_records(method,'cited-method','A','p',2)
        self.source.patch({'base_version':1,'nodes':[node('cited-method','Method',method,'A','p','paper_claim'),*nodes],'edges':edges})
        package=self.export();validate_package(package)
        self.assertEqual(package['schema'],'memory-policy-v4')
        reference=next(n for n in package['nodes'] if n['id']==nodes[0]['id'])
        self.assertEqual(reference['attrs']['verification_status'],'agent_reported_unverified')
        self.assertEqual(reference['attrs']['source_version'],'unknown')
        self.assertIsNone(reference['attrs']['retrieved_content_ref'])
        self.assertTrue(any(e['relation']=='DERIVED_FROM' and e['source']=='cited-method' and e['target']==reference['id'] for e in package['edges']))
        import_memory(self.target,self.package,'B','q')
        transferred=next(n for n in self.target.snapshot()['nodes'] if n['attrs']['source_node']['id']==reference['id'])
        self.assertEqual(transferred['attrs']['source_node'],reference)
        self.assertEqual(transferred['attrs']['status'],'unknown')
        self.assertEqual(transferred['attrs']['transfer_status'],'analogy_requires_target_validation')
        self.assertTrue(any(e['target']==transferred['id'] and e['attrs']['source_relation']=='DERIVED_FROM' for e in self.target.snapshot()['edges']))
        before=self.target.verify();import_memory(self.target,self.package,'B','q');self.assertEqual(self.target.verify(),before)
        self.assertEqual(self.export(limit=1)['nodes'],[])  # Do not detach source from method to fit capacity.
        package['schema']='memory-policy-v3';package['hash']=digest({k:v for k,v in package.items() if k!='hash'})
        with self.assertRaises(ContractError):validate_package(package)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.target = Store(self.root / 'source'), Store(self.root / 'target')
        self.addCleanup(self.source.close); self.addCleanup(self.target.close)
        self.package = self.root / 'package.json'
        nodes = [node('support', 'Experiment', {'valid': True, 'metrics': {'quality': 0.8},
                     'independent_run_id': 'run-a', 'checkpoint_ref': {'sha256': 'DO_NOT_EXPORT'},
                     'raw_users': ['DO_NOT_EXPORT'], 'failure_kind': None,
                     'evidence_dependency_ids': ['source-attempt-not-graph-node']}, 'A', 'p'),
                 node('counter', 'Experiment', {'valid': True, 'metrics': {'quality': 0.1}, 'independent_run_id': 'run-b'}, 'A', 'p'),
                 node('claim', 'Claim', {'status': 'provisional', 'scope': 'A only',
                      'statement': 'Conditional method may improve quality.', 'evidence_refs': ['support']}, 'A', 'p')]
        relations = [edge('support-edge', 'support', 'claim', 'SUPPORTS', ['support'], 'A', 1),
                     edge('counter-edge', 'counter', 'claim', 'CONTRADICTS', ['counter'], 'A', 1)]
        self.source.patch({'base_version': 0, 'nodes': nodes, 'edges': relations})

    def export(self, **kwargs):
        return export_memory(self.source, self.package, task_id='A', protocol_hash='p', **kwargs)

    def test_cutoff_freezes_original_values_and_package_hash(self):
        first = self.export(cutoff_seq=1)
        updated = node('claim', 'Claim', {'status': 'disputed', 'scope': 'A only',
                       'statement': 'Future interpretation', 'evidence_refs': ['counter']}, 'A', 'p')
        self.source.patch({'base_version': 1, 'nodes': [updated], 'edges': []})
        second = self.export(cutoff_seq=1)
        self.assertEqual(first, second)
        self.assertNotIn('Future interpretation', canonical(second))
        with self.assertRaisesRegex(ContractError, 'after source cutoff'):
            self.export(version=2, cutoff_seq=1)

    def test_counterevidence_closure_and_capacity_never_cherry_pick(self):
        package = self.export(limit=3)
        self.assertEqual({n['id'] for n in package['nodes']}, {'claim', 'support', 'counter'})
        validate_package(package)
        limited = self.export(limit=2)
        self.assertEqual(limited['nodes'], [])
        self.assertGreater(limited['omitted_roots']['capacity'], 0)

    def test_raw_artifacts_projected_out_and_smuggled_fields_rejected(self):
        package = self.export()
        self.assertNotIn('DO_NOT_EXPORT', canonical(package))
        package['nodes'][0]['attrs']['weights'] = [1, 2, 3]
        package['capacity']['payload_bytes'] = len(canonical({k: package[k] for k in ['nodes', 'edges']}).encode())
        package['hash'] = digest({k: v for k, v in package.items() if k != 'hash'})
        write_json(self.package, package)
        with self.assertRaisesRegex(ContractError, 'unapproved'):
            import_memory(self.target, self.package, 'B', 'q')

    def test_overlay_downgrades_source_evidence_and_preserves_counter_retrieval(self):
        self.export()
        first = import_memory(self.target, self.package, 'B', 'q')
        self.assertEqual(first['imported'], 3)
        self.assertEqual(import_memory(self.target, self.package, 'B', 'q')['imported'], 0)
        nodes = self.target.snapshot()['nodes']
        self.assertFalse(any(n['type'] in {'Experiment', 'Claim'} for n in nodes))
        bundle = self.target.bundle('B', 'q', 'hypothesis', limit=1)
        self.assertEqual(bundle['nodes'][0]['attrs']['source_node']['id'], 'counter')
        self.assertEqual(bundle['nodes'][0]['attrs']['evidence_refs'], [])
        reverse = export_memory(self.target, self.root / 'reverse.json', task_id='B', protocol_hash='q')
        self.assertEqual(reverse['nodes'], [])  # No source A -> B -> A laundering.

    def test_source_graph_and_started_target_cannot_receive_fresh_transfer(self):
        self.export()
        with self.assertRaisesRegex(ContractError, 'separate'):
            import_memory(self.source, self.package, 'B', 'q')
        self.target.patch({'base_version': 0, 'nodes': [node('b-run', 'Experiment', {}, 'B', 'q')], 'edges': []})
        with self.assertRaisesRegex(ContractError, 'before target research'):
            import_memory(self.target, self.package, 'B', 'q')

    def test_cross_task_evidence_excludes_whole_connected_claim(self):
        self.source.patch({'base_version': 1,
            'nodes': [node('target-method', 'Method', {'mechanism': 'target B experience'}, 'B', 'q')],
            'edges': [edge('mixed', 'claim', 'target-method', 'BASED_ON', [], 'A', 2)]})
        package = self.export()
        self.assertEqual(package['nodes'], [])
        self.assertGreater(package['omitted_roots']['ineligible_or_incomplete'], 0)

    def test_runtime_proposal_closure_keeps_gap_method_claim_and_counterevidence(self):
        self.source.patch({'base_version':1,'nodes':[
            node('gap','GapHypothesis',{'observation_refs':['support']},'A','p'),
            node('method','Method',{'mechanism':'explicit feature interaction'},'A','p'),
            node('proposal','Proposal',{'mechanism':'add an interaction','source_code':'DO_NOT_EXPORT'},'A','p'),
            node('claim','Claim',{'status':'provisional','scope':'A only','statement':'Conditional benefit',
                'evidence_refs':['support','proposal']},'A','p')],
            'edges':[edge('targets','proposal','gap','TARGETS',[],'A',2),
                     edge('uses','proposal','method','USES',[],'A',2)]})
        package=self.export();validate_package(package)
        self.assertEqual({n['id'] for n in package['nodes']},{'support','counter','claim','gap','method','proposal'})
        self.assertNotIn('DO_NOT_EXPORT',canonical(package))
        self.assertIn('USES',{e['relation'] for e in package['edges']})
        import_memory(self.target,self.package,'B','q')
        transferred=next(n for n in self.target.snapshot()['nodes'] if n['attrs']['source_node']['id']=='proposal')
        self.assertEqual(transferred['type'],'Source')
        self.assertEqual(transferred['attrs']['transfer_status'],'analogy_requires_target_validation')
        self.assertEqual(self.export(limit=5)['nodes'],[])
        package['schema']='memory-policy-v2'
        package['hash']=digest({k:v for k,v in package.items() if k!='hash'})
        with self.assertRaisesRegex(ContractError,'requires memory-policy-v3'):validate_package(package)

    def test_legacy_v2_packages_remain_readable(self):
        package=self.export();package['schema']='memory-policy-v2'
        package['hash']=digest({k:v for k,v in package.items() if k!='hash'})
        validate_package(package);write_json(self.package,package)
        self.assertEqual(import_memory(self.target,self.package,'B','q')['imported'],3)

    def test_tampered_cutoff_extra_payload_rejected(self):
        package = self.export()
        package['source_cutoff']['raw_users'] = [1, 2]
        package['hash'] = digest({k: v for k, v in package.items() if k != 'hash'})
        with self.assertRaisesRegex(ContractError, 'cutoff'):
            validate_package(package)

    def test_live_target_controller_blocks_import(self):
        self.export()
        with (self.target.root.parent / '.controller.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ContractError, 'controller is running'):
                import_memory(self.target, self.package, 'B', 'q')


if __name__ == '__main__':
    unittest.main()
