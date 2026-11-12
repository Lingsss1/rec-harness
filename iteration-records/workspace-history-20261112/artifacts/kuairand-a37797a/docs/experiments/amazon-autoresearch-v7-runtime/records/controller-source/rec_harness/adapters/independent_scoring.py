"""Host-owned ranking from label-free, permuted candidate score responses.

Only the request crosses into model execution. The private key stays with the
controller. This does not itself isolate the model process or admit its code.
"""
import math
import random
import secrets
import platform
import sys

from rec_harness.core import ContractError, digest, file_hash


def scoring_runtime():
    """Python and math-extension identity; not a full host OS environment lock."""
    return {'implementation':sys.implementation.name,'version':sys.version,'machine':platform.machine(),
            'executable_sha256':file_hash(sys.executable),
            'math_module_sha256':file_hash(math.__file__) if getattr(math,'__file__',None) else None}


def require_scoring_runtime(expected):
    if expected!=scoring_runtime():
        raise ContractError('host scoring runtime changed or absent; replay with the original scoring interpreter')


def prepare_request(rows, *, seed=None):
    if not isinstance(rows, list) or not rows:
        raise ContractError('nonempty evaluation rows required')
    # A seed supports reproducible acceptance fixtures. Production must archive
    # the private key separately and never reveal the permutation seed.
    rng = random.Random(seed) if seed is not None else secrets.SystemRandom()
    public, private, users = [], [], set()
    for index, row in enumerate(rows):
        user, history, candidates = row['user_id'], row['history'], row['candidates']
        if type(user) is not int or user in users:
            raise ContractError('evaluation users must be unique integer IDs')
        users.add(user)
        if not isinstance(history, list) or not history or any(type(x) is not int or x <= 0 for x in history):
            raise ContractError('evaluation history must contain positive item IDs')
        if not isinstance(candidates, list) or len(candidates) != 100 or len(set(candidates)) != 100 or any(type(x) is not int or x <= 0 for x in candidates):
            raise ContractError('evaluation requires 100 distinct positive candidate IDs')
        order = list(range(100)); rng.shuffle(order)
        token = 'row-' + str(index)
        public.append({'row_id': token, 'history': list(history), 'candidates': [candidates[i] for i in order]})
        private.append({'row_id': token, 'user_id': user, 'positive_index': order.index(0)})
    request = {'schema': 'candidate-scores-request-v1', 'rows': public}
    key = {'schema': 'candidate-scores-private-v1', 'request_hash': digest(request), 'rows': private}
    return request, key


def score_response(request, key, response):
    if request.get('schema') != 'candidate-scores-request-v1' or key.get('schema') != 'candidate-scores-private-v1' or digest(request) != key.get('request_hash'):
        raise ContractError('score request/private key identity mismatch')
    if not isinstance(response, dict) or set(response) != {'schema', 'request_hash', 'rows'} or response['schema'] != 'candidate-scores-response-v1' or response['request_hash'] != key['request_hash']:
        raise ContractError('score response identity mismatch')
    if not isinstance(response['rows'], list) or len(response['rows']) != len(key['rows']) or len(request['rows']) != len(key['rows']):
        raise ContractError('score response row count mismatch')
    def finite_score(value):
        try:
            return type(value) in (float, int) and math.isfinite(value)
        except OverflowError:
            return False
    by_id = {}
    for row in response['rows']:
        if not isinstance(row, dict) or set(row) != {'row_id', 'scores'} or not isinstance(row['row_id'], str) or row['row_id'] in by_id:
            raise ContractError('invalid or duplicate score response row')
        scores = row['scores']
        if not isinstance(scores, list) or len(scores) != 100 or any(not finite_score(x) for x in scores):
            raise ContractError('scores must be 100 finite numbers')
        by_id[row['row_id']] = scores
    if set(by_id) != {r['row_id'] for r in key['rows']}:
        raise ContractError('score response row IDs differ')
    ranks, ties, users = [], [], []
    for public, private in zip(request['rows'], key['rows']):
        if public['row_id'] != private['row_id'] or type(private['positive_index']) is not int or not 0 <= private['positive_index'] < 100:
            raise ContractError('score private key alignment invalid')
        scores = by_id[private['row_id']]; position = private['positive_index']; positive = scores[position]
        ranks.append(sum(s > positive for s in scores))
        ties.append(sum(s == positive for i, s in enumerate(scores) if i != position))
        users.append(private['user_id'])
    metrics = {}
    for k in (5, 10, 20):
        metrics[f'HR@{k}'] = sum(r < k for r in ranks) / len(ranks)
        metrics[f'NDCG@{k}'] = sum(1 / math.log2(r + 2) if r < k else 0 for r in ranks) / len(ranks)
    body = {'schema': 'host-ranking-v1', 'request_hash': key['request_hash'], 'response_hash': digest(response),
            'user_ids': users, 'ranks': ranks, 'metrics': metrics, 'positive_score_ties': ties,
            'tie_policy': 'strictly greater negatives; preserves frozen optimistic tie rule',
            'tied_user_count': sum(n > 0 for n in ties),
            'qualification': 'host-computed ranking; not candidate admission or independent test data'}
    return {**body, 'hash': digest(body)}
