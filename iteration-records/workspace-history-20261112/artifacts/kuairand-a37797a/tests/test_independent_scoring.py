import copy
import unittest

from rec_harness.adapters.independent_scoring import prepare_request, score_response
from rec_harness.core import ContractError, digest


class IndependentScoringTests(unittest.TestCase):
    def fixture(self):
        request, key = prepare_request([{'user_id': 71, 'history': [101,102], 'candidates': list(range(1,101))},
                                        {'user_id': 92, 'history': [103], 'candidates': list(range(100,0,-1))}], seed=4)
        response = {'schema': 'candidate-scores-response-v1', 'request_hash': digest(request),
                    'rows': [{'row_id': r['row_id'], 'scores': list(map(float,r['candidates']))} for r in request['rows']]}
        return request, key, response

    def test_only_host_knows_positive_and_computes_rank(self):
        request, key, response = self.fixture()
        for row in request['rows']:
            self.assertEqual(set(row), {'row_id','history','candidates'})
        self.assertNotEqual(key['rows'][0]['positive_index'],0)
        result = score_response(request,key,response)
        self.assertEqual(result['ranks'],[99,0])
        self.assertEqual(result['metrics']['HR@10'],0.5)
        response['rows'].reverse()
        self.assertEqual(score_response(request,key,response)['ranks'],[99,0])

    def test_nonfinite_missing_duplicate_and_forged_metrics_rejected(self):
        request,key,response = self.fixture()
        bad = copy.deepcopy(response); bad['rows'][0]['scores'][0] = float('nan')
        with self.assertRaisesRegex(ContractError,'finite'): score_response(request,key,bad)
        bad = copy.deepcopy(response); bad['rows'][0]['scores'][0] = 10**500
        with self.assertRaisesRegex(ContractError,'finite'): score_response(request,key,bad)
        bad = copy.deepcopy(response); bad['metrics'] = {'HR@10':1}
        with self.assertRaisesRegex(ContractError,'identity'): score_response(request,key,bad)
        bad = copy.deepcopy(response); bad['rows'][1]['row_id'] = bad['rows'][0]['row_id']
        with self.assertRaisesRegex(ContractError,'duplicate'): score_response(request,key,bad)
        bad = copy.deepcopy(response); bad['rows'].pop()
        with self.assertRaisesRegex(ContractError,'count'): score_response(request,key,bad)

    def test_frozen_tie_bias_is_visible_not_silently_changed(self):
        request,key,response = self.fixture()
        for row in response['rows']: row['scores'] = [0.0]*100
        result = score_response(request,key,response)
        self.assertEqual(result['ranks'],[0,0])
        self.assertEqual(result['positive_score_ties'],[99,99])
        self.assertEqual(result['tied_user_count'],2)
