import copy
import unittest

from rec_harness.core import ContractError
from rec_harness.memory_view import paired_views, present, restore


class MemoryViewTests(unittest.TestCase):
    def bundle(self):
        return {'graph_snapshot_id': 2, 'graph_hash': 'snapshot', 'role': 'evidence',
                'nodes': [{'id': 'negative', 'type': 'Observation', 'attrs': {'counterevidence': True}}],
                'edges': [{'id': 'relation', 'source': 'negative', 'target': 'negative',
                           'relation': 'CONTRADICTS', 'attrs': {'origin': 'local_evidence'}}],
                'retrieval': {'counterevidence_omitted': 0}}

    def test_same_information_including_relations_and_provenance(self):
        bundle = self.bundle()
        pair = paired_views(bundle)
        for mode in ['graph', 'flat']:
            self.assertEqual(restore(pair[mode]), bundle)
            self.assertEqual(pair[mode]['presentation']['information_hash'], pair['information_hash'])
        self.assertNotIn('nodes', pair['flat'])
        self.assertEqual(len(pair['flat']['records']), 2)
        pair['flat']['records'][0]['value']['attrs']['counterevidence'] = False
        self.assertTrue(bundle['nodes'][0]['attrs']['counterevidence'])
        with self.assertRaisesRegex(ContractError, 'hash mismatch'):
            restore(pair['flat'])

    def test_removing_relationships_or_mixing_representations_is_rejected(self):
        flat = present(self.bundle(), 'flat')
        removed = copy.deepcopy(flat)
        removed['records'].pop()
        with self.assertRaisesRegex(ContractError, 'hash mismatch'):
            restore(removed)
        flat['nodes'] = []
        with self.assertRaisesRegex(ContractError, 'conflicting'):
            restore(flat)
