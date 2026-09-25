import copy
import unittest
from rec_harness.activation_contract import public_measurement
from rec_harness.core import ContractError


class ContractTests(unittest.TestCase):
    def setUp(self):
        effect={'histories':2,'tail_change_rms':1.,'tail_reference_rms':2.,'relative_tail_change':.5}
        self.report={'schema':'masked-residual-probe-v1','source_hash':'source','probe_hash':'probe',
            'initialization':'synthetic','settings':{'NUM_BLOCKS':1},'unchanged_state':True,
            'scope':'UNTRUSTED_TEXT','limitations':['DO_NOT_FORWARD'],
            'blocks':[{'block_index':0,'activation_rms':2.,'residual_rms':1.,'zero_fraction':0.,'valid_positions':4,
                'bypass':{'all':copy.deepcopy(effect),'short':copy.deepcopy(effect)}}]}
        self.expected={**{k:self.report[k] for k in ['source_hash','probe_hash','initialization','settings']},
            'slice_counts':{'all':2,'short':2},'valid_positions':4}

    def test_projection_omits_untrusted_text_and_preserves_defined_measurement(self):
        result=public_measurement(self.report,self.expected)
        self.assertNotIn('DO_NOT_FORWARD',str(result));self.assertNotIn('UNTRUSTED_TEXT',str(result))
        self.assertEqual(result['measurements']['block_0.all.relative_tail_change'],.5)

    def test_tampered_identity_population_ratio_and_nonfinite_fail(self):
        values=[]
        for key,value in [('source_hash','other'),('unchanged_state',False),('user_ids',[1])]:
            report=copy.deepcopy(self.report);report[key]=value;values.append(report)
        for key,value in [('histories',3),('relative_tail_change',.75),('tail_change_rms',float('nan'))]:
            report=copy.deepcopy(self.report);report['blocks'][0]['bypass']['all'][key]=value;values.append(report)
        for report in values:
            with self.assertRaises(ContractError):public_measurement(report,self.expected)

    def test_undefined_ratio_is_not_reported_as_zero_effect(self):
        for effect in self.report['blocks'][0]['bypass'].values():
            effect['tail_reference_rms']=0;effect['relative_tail_change']=None
        result=public_measurement(self.report,self.expected)['measurements']
        self.assertEqual(result['block_0.all.relative_defined'],0)
        self.assertNotIn('block_0.all.relative_tail_change',result)
        self.assertEqual(result['block_0.all.tail_change_rms'],1)

    def test_real_history_identity_is_required_and_scope_is_host_authored(self):
        self.expected['history_hash']='frozen-history'
        with self.assertRaises(ContractError):public_measurement(self.report,self.expected)
        self.report['history_hash']='other'
        with self.assertRaisesRegex(ContractError,'history input identity'):
            public_measurement(self.report,self.expected)
        self.report['history_hash']='frozen-history'
        result=public_measurement(self.report,self.expected)
        self.assertIn('Sampled training-history',result['limitations'][0])
        self.assertNotIn('history_hash',result)
        self.assertNotIn('UNTRUSTED_TEXT',str(result))

    def test_slice_arithmetic_and_reference_consistency_fail(self):
        other=copy.deepcopy(self.report['blocks'][0]);other['block_index']=1
        for effect in other['bypass'].values():effect['tail_reference_rms']=4;effect['relative_tail_change']=.25
        self.report['blocks'].append(other);self.expected['settings']={'NUM_BLOCKS':2};self.report['settings']={'NUM_BLOCKS':2}
        with self.assertRaisesRegex(ContractError,'aggregate'):public_measurement(self.report,self.expected)
