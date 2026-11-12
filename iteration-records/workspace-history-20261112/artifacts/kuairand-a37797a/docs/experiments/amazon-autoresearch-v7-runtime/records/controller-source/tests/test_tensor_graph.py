import copy
import unittest
from rec_harness.core import ContractError,digest
from rec_harness.model_graph import tensor_graph_summary,validate_tensor_graph,sequence_semantics_summary


def semantics_fixture():
    return {condition:{'input_tokens_unchanged':True,'batch_independent':True,
        'batch_permutation_equivariant':True,'input_axes':['user','time'],
        'representation_axes':['user','time','feature']} for condition in ['dense','right_aligned_histories']}


def report_fixture():
    metadata={'shape':[1,4],'dtype':'torch.float32','requires_grad':False}
    graph={'schema':'observed-tensor-graph-v1','model_id':'candidate','source_hash':'source',
        'tensors':[{'id':'input','metadata':metadata,'storage':'s0','producer':None,'external_name':'sequence'},
                   {'id':'output','metadata':metadata,'storage':'s1','producer':'op'}],
        'operations':[{'id':'op','ordinal':0,'operator':'aten.add.Tensor','module_path':'',
            'inputs':['input','input'],'outputs':['output'],'mutation_dependencies':[],'written_storages':[]}],
        'outputs':['output'],'scope':'Concrete synthetic encode path only.'}
    graph['hash']=digest(graph)
    return {'model_id':'candidate','source_hash':'source','tensor_graph':graph}


class TensorGraphTests(unittest.TestCase):
    def test_sequence_semantics_requires_all_conditions_and_honors_legacy_absence(self):
        self.assertEqual(sequence_semantics_summary({})['status'],'unavailable')
        with self.assertRaises(ContractError):sequence_semantics_summary({},required=True)
        with self.assertRaises(ContractError):sequence_semantics_summary({'checks':{'sequence_semantics':[]}})
        report={'checks':{'sequence_semantics':semantics_fixture()}}
        self.assertEqual(sequence_semantics_summary(report,required=True)['status'],'verified')
        for condition in ['dense','right_aligned_histories']:
            for field in ['input_tokens_unchanged','batch_independent','batch_permutation_equivariant']:
                broken=copy.deepcopy(report);broken['checks']['sequence_semantics'][condition][field]=False
                with self.assertRaises(ContractError):sequence_semantics_summary(broken)

    def test_summary_binds_source_and_exposes_coverage(self):
        report=report_fixture();summary=tensor_graph_summary(report,'candidate','source',required=True)
        self.assertEqual(summary['operation_count'],1)
        self.assertEqual(summary['operator_counts'],{'aten.add.Tensor':1})
        self.assertEqual(summary['graph_hash'],report['tensor_graph']['hash'])
        self.assertNotIn('tensors',summary)
        for model,source in [('other','source'),('candidate','other')]:
            with self.assertRaises(ContractError):tensor_graph_summary(report,model,source)

    def test_legacy_absence_is_explicit_and_new_gate_requires_graph(self):
        self.assertEqual(tensor_graph_summary({},'candidate','source')['status'],'unavailable')
        with self.assertRaisesRegex(ContractError,'required'):
            tensor_graph_summary({},'candidate','source',required=True)

    def test_rehashed_invalid_dependency_cannot_pass(self):
        report=report_fixture();graph=report['tensor_graph']
        cases=[]
        missing=copy.deepcopy(graph);missing['operations'][0]['inputs']=['missing'];cases.append(missing)
        future=copy.deepcopy(graph);future['operations'][0]['mutation_dependencies']=['op'];cases.append(future)
        unbound=copy.deepcopy(graph);unbound['operations'][0]['outputs']=[];cases.append(unbound)
        for broken in cases:
            broken['hash']=digest({k:v for k,v in broken.items() if k!='hash'})
            with self.assertRaises(ContractError):validate_tensor_graph(broken,'candidate','source')
