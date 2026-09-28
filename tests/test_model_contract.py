import tempfile
import unittest
from pathlib import Path

from rec_harness.core import ContractError, digest, file_hash, write_json
from rec_harness.model_graph import compare_sites, register_contract, validate_tensor_graph, parameter_sharing_summary
from rec_harness.store import Store

try:
    import torch
    from rec_harness.adapters.model_contract import check_contract, trace_encode, trace_operations, check_sequence_semantics
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch environment required')
class ModelContractTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(7)
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.item_emb = torch.nn.Embedding(33, 4, padding_idx=0)
                self.mix = torch.nn.Linear(4, 4)
            def encode_sequence(self, seq):
                return self.mix(self.mix(self.item_emb(seq)))
            def forward(self, seq, pos, neg):
                x = self.encode_sequence(seq)
                return (x * self.item_emb(pos)).sum(-1), (x * self.item_emb(neg)).sum(-1)
            def score_candidates(self, seq, candidates):
                return self.item_emb(candidates) @ self.encode_sequence(seq)[0, -1]
        self.model = Tiny()

    def test_correct_model_and_reused_module_callsite_identity(self):
        checks = check_contract(self.model, itemnum=32, length=8)
        self.assertTrue(checks['causal_input_gradient'])
        sequence = torch.tensor([[1, 2, 3, 4]])
        first = trace_encode(self.model, sequence, 'v1', 'code')
        repeated = trace_encode(self.model, sequence, 'v1', 'code')
        self.assertEqual(first, repeated)
        self.assertEqual(len({c['id'] for c in first['components']}), 3)
        mix = [c for c in first['components'] if c['module_path'] == 'mix']
        self.assertEqual([c['callsite'] for c in mix], ['encode_sequence:1', 'encode_sequence:2'])
        self.assertIn(mix[0]['id'], mix[1]['direct_input_producers'])
        second = trace_encode(self.model, sequence, 'v2', 'code')
        self.assertFalse({c['id'] for c in first['components']} & {c['id'] for c in second['components']})

    def test_parameter_ties_and_repeated_calls_are_distinct_from_equal_shapes(self):
        class Pair(torch.nn.Module):
            def __init__(self,tied):
                super().__init__()
                self.left=torch.nn.Linear(4,4,bias=False)
                self.right=torch.nn.Linear(4,4,bias=False)
                if tied:self.right.weight=self.left.weight
            def encode_sequence(self,x):return self.right(self.left(x))
        x=torch.ones(2,3,4)
        for tied in (False,True):
            model=Pair(tied)
            structure=trace_encode(model,x,'pair','source')
            report={'model_id':'pair','source_hash':'source','structure':structure}
            summary=parameter_sharing_summary(report)
            self.assertEqual(summary['registered_parameter_count'],1 if tied else 2)
            self.assertEqual(len(summary['shared_groups']),int(tied))
            if tied:
                group=summary['shared_groups'][0]
                self.assertEqual(group['registered_names'],['left.weight','right.weight'])
                self.assertEqual({use['module_path'] for use in group['observed_uses']},{'left','right'})
                structure['components'][1]['parameter_bindings']['weight']='unknown'
                with self.assertRaisesRegex(ContractError,'binding does not resolve'):parameter_sharing_summary(report)
        report={'model_id':'v1','source_hash':'source',
                'structure':trace_encode(self.model,torch.tensor([[1,2,3,4]]),'v1','source')}
        summary=parameter_sharing_summary(report)
        self.assertEqual(len(summary['shared_groups']),2)  # mix weight and bias, used twice.
        self.assertTrue(all(len(group['registered_names'])==1 for group in summary['shared_groups']))
        self.assertTrue(all(len(group['observed_uses'])==2 for group in summary['shared_groups']))
        self.assertNotIn(str(id(self.model.mix.weight)),str(summary))
        report['structure'].pop('parameter_sharing')
        self.assertEqual(parameter_sharing_summary(report)['status'],'unavailable')

    def test_future_leak_wrong_readout_and_disconnected_parameter_rejected(self):
        encode = self.model.encode_sequence
        self.model.encode_sequence = lambda seq: encode(seq) + encode(seq).flip(1)
        with self.assertRaisesRegex(ContractError, 'future token'):
            check_contract(self.model, itemnum=32, length=8)
        self.model.encode_sequence = encode
        score = self.model.score_candidates
        self.model.score_candidates = lambda seq, candidates: self.model.item_emb(candidates) @ encode(seq)[0, 0]
        with self.assertRaisesRegex(ContractError, 'tail candidate'):
            check_contract(self.model, itemnum=32, length=8)
        self.model.score_candidates = score
        self.model.unused = torch.nn.Parameter(torch.ones(2))
        with self.assertRaisesRegex(ContractError, 'disconnected'):
            check_contract(self.model, itemnum=32, length=8)

    def test_model_graph_identity_and_mapping_do_not_assert_equivalence(self):
        structure = trace_encode(self.model, torch.tensor([[1, 2, 3, 4]]), 'v1', 'code')
        report = {'model_id': 'v1', 'source_hash': 'code', 'settings': {}, 'structure': structure,
                  'checks': check_contract(self.model, itemnum=32, length=8),
                  'inspector_hash': 'fixture', 'input_protocol': {'length': 8, 'seed': 7}}
        with tempfile.TemporaryDirectory() as directory:
            for label in ['source', 'inspector']:
                code = Path(directory) / (label + '.py'); code.write_text('# unit fixture only\n')
                report[label + '_filename'] = code.name
                report[label + '_hash'] = file_hash(code)
            report['structure']=trace_encode(self.model,torch.tensor([[1,2,3,4]]),'v1',report['source_hash'])
            report['tensor_graph']=trace_operations(self.model,torch.tensor([[1,2,3,4]]),'v1',report['source_hash'])
            report['hash'] = digest(report)
            path = Path(directory) / 'report.json'; write_json(path, report)
            store = Store(Path(directory) / 'memory')
            try:
                first = register_contract(store, path, 'task', 'protocol')
                self.assertTrue(first['imported'])
                registered_model=next(n for n in store.snapshot()['nodes'] if n['id']==first['model_id'])
                self.assertEqual(len(registered_model['attrs']['parameter_sharing']['shared_groups']),2)
                graph_ids={n['id'] for n in store.snapshot()['nodes']}
                self.assertTrue(all(use['component_id'] in graph_ids
                    for group in registered_model['attrs']['parameter_sharing']['shared_groups'] for use in group['observed_uses']))
                component=next(n for n in store.snapshot()['nodes'] if n['type']=='Component' and n['attrs'].get('module_path')=='mix')
                self.assertEqual(set(component['attrs']['parameter_bindings']),{'weight','bias'})
                self.assertFalse(register_contract(store, path, 'task', 'protocol')['imported'])
                self.assertEqual(sum(n['type'] == 'Component' for n in store.snapshot()['nodes']), 3+len(report['tensor_graph']['operations']))
                self.assertTrue(any(e['attrs'].get('measurement')=='observed_tensor_operation_or_storage_mutation' for e in store.snapshot()['edges']))
                self.assertTrue(any(e['relation'] == 'FLOWS_TO' for e in store.snapshot()['edges']))
                checkpoint = Path(directory) / 'checkpoint.pt'
                torch.save(self.model.state_dict(), checkpoint)
                trained = {k: v for k, v in report.items() if k != 'hash'}
                trained['checkpoint_hash'] = file_hash(checkpoint)
                trained['hash'] = digest(trained)
                trained_path = Path(directory) / 'trained.json'; write_json(trained_path, trained)
                with self.assertRaisesRegex(ContractError, 'matching checkpoint'):
                    register_contract(store, trained_path, 'task', 'protocol')
                registered = register_contract(store, trained_path, 'task', 'protocol', checkpoint_path=checkpoint)
                self.assertNotEqual(first['model_id'], registered['model_id'])
                altered = {**report, 'source_hash': 'different'}; write_json(path, altered)
                with self.assertRaisesRegex(ContractError, 'hash mismatch'):
                    register_contract(store, path, 'task', 'protocol')
            finally: store.close()
        mapping = compare_sites(report, report)
        self.assertTrue(all(c['observed_shape_compatible'] for c in mapping['correspondences']))
        self.assertFalse(any(c['automatic_insertion_allowed'] for c in mapping['correspondences']))

    def test_tensor_graph_includes_functional_residual_and_alias_mutations(self):
        class Functional(torch.nn.Module):
            def encode_sequence(self, seq):
                x=seq.float()
                view=x[:, :]
                view.add_(1)
                return x + torch.nn.functional.pad(x.cumsum(1)[:, :-1], (1,0))
        model=Functional();sequence=torch.tensor([[1,2,3,4]])
        first=trace_operations(model,sequence,'v1','source')
        self.assertEqual(first,trace_operations(model,sequence,'v1','source'))
        ops=first['operations'];names=[row['operator'] for row in ops]
        self.assertIn('aten.cumsum.default',names)
        self.assertIn('aten.constant_pad_nd.default',names)
        mutation=next(row for row in ops if row['operator']=='aten.add_.Tensor')
        residual=next(row for row in ops if row['operator']=='aten.add.Tensor')
        self.assertIn(mutation['id'],residual['mutation_dependencies'])
        self.assertTrue(mutation['written_storages'])
        known=set();by_tensor={row['id']:row for row in first['tensors']}
        for row in ops:
            self.assertTrue(set(row['mutation_dependencies'])<=known)
            for ref in row['inputs']:
                producer=by_tensor[ref]['producer']
                self.assertTrue(producer is None or producer in known)
            known.add(row['id'])
        self.assertEqual(by_tensor[first['outputs'][0]]['producer'],residual['id'])
        self.assertEqual(validate_tensor_graph(first,'v1','source'),by_tensor)
        import copy
        broken=copy.deepcopy(first)
        broken['operations'][0]['mutation_dependencies']=[residual['id']]
        broken['hash']=digest({k:v for k,v in broken.items() if k!='hash'})
        with self.assertRaisesRegex(ContractError,'forward mutation'):
            validate_tensor_graph(broken,'v1','source')
        broken=copy.deepcopy(first)
        next(row for row in broken['operations'] if row['id']==residual['id'])['mutation_dependencies']=[]
        broken['hash']=digest({k:v for k,v in broken.items() if k!='hash'})
        with self.assertRaisesRegex(ContractError,'storage mutation dependencies'):
            validate_tensor_graph(broken,'v1','source')


        with self.assertRaisesRegex(ContractError,'operation limit'):
            trace_operations(model,sequence,'v1','source',max_operations=1)
        self.assertEqual(first,trace_operations(model,sequence,'v1','source'))

    def test_full_sequence_semantics_reject_cross_user_prefix_mixing(self):
        encode=self.model.encode_sequence
        def mixed(seq):
            value=encode(seq)
            # Leaves the tail readout and mid-sequence causal-gradient probe unchanged.
            return torch.cat([value[:,:1]+value[:,:1].mean(0,keepdim=True),value[:,1:]],dim=1)
        self.model.encode_sequence=mixed
        with self.assertRaisesRegex(ContractError,'other batch users'):
            check_contract(self.model,itemnum=32,length=8)

    def test_sequence_semantics_reject_input_mutation(self):
        encode=self.model.encode_sequence
        def modifies(seq):
            value=encode(seq)
            seq[:,0]=0
            return value
        self.model.encode_sequence=modifies
        self.model.eval()
        with self.assertRaisesRegex(ContractError,'modified input tokens'):
            check_sequence_semantics(self.model,torch.tensor([[1,2,3],[4,5,6]]))

    def test_checkpoint_hash_and_full_tensor_finiteness(self):
        from rec_harness.adapters.model_contract import load_checkpoint
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pt'
            torch.save(self.model.state_dict(), path)
            weights, itemnum = load_checkpoint(path, file_hash(path))
            self.assertEqual(itemnum, 32)
            self.assertTrue(torch.equal(weights['item_emb.weight'], self.model.item_emb.weight))
            with self.assertRaisesRegex(ContractError, 'hash mismatch'):
                load_checkpoint(path, 'wrong')
            weights['item_emb.weight'][-1, 0] = float('nan')
            torch.save(weights, path)
            with self.assertRaisesRegex(ContractError, 'nonfinite'):
                load_checkpoint(path, file_hash(path))

    def test_zero_initialized_readout_must_remain_valid_after_learning(self):
        import copy
        from rec_harness.adapters.training_smoke import stress_contract
        broken=copy.deepcopy(self.model)
        original_score=broken.score_candidates
        broken.readout=torch.nn.Parameter(torch.zeros(4,4))
        def forward(seq,pos,neg):
            x=broken.encode_sequence(seq);x=x+x@broken.readout
            return (x*broken.item_emb(pos)).sum(-1),(x*broken.item_emb(neg)).sum(-1)
        def score(seq,candidates):
            x=broken.encode_sequence(seq)[0,-1];x=x+x@broken.readout
            return broken.item_emb(candidates)@x
        broken.forward=forward;broken.score_candidates=score
        self.assertTrue(check_contract(broken,itemnum=32,length=8)['tail_readout'])
        with self.assertRaisesRegex(ContractError,'tail candidate'):
            stress_contract(broken,itemnum=32,length=8)
        with torch.no_grad():broken.readout.zero_()
        broken.score_candidates=original_score
        with self.assertRaisesRegex(ContractError,'forward scoring'):
            stress_contract(broken,itemnum=32,length=8)
        valid=self.model;valid.readout=torch.nn.Parameter(torch.zeros(4,4));encode=valid.encode_sequence
        def transformed(seq):
            x=encode(seq);return x+x@valid.readout
        valid.encode_sequence=transformed
        self.assertEqual(stress_contract(valid,itemnum=32,length=8)['steps'],3)


if __name__ == '__main__':
    unittest.main()
