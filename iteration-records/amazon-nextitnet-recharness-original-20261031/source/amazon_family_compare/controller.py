"""Family-constrained comparisons with the proven retrying transport."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from .core import Controller, TEXT, schema, read, save
from .resilient_self import ResilientSelfController, TransportUnavailable
from .protocol_fixes import SINGLE_MODEL_CONTRACT


class FamilyController(ResilientSelfController):
    def environment(self, gpu):
        env = super().environment(gpu)
        env['GAGC_MODEL_FAMILY'] = self.c['model_family']
        return env

    def llm(self, instruction, output_schema=None, search=False, key=None):
        from .mlestar import CANDIDATES
        if self.method == 'mlestar' and output_schema == CANDIDATES:
            prefix, context = instruction.rsplit('\nCONTEXT:\n', 1)
            return self.llm_candidates(prefix, json.loads(context), key)
        family = self.c['model_family']
        contract = (f'COMPARISON MODEL FAMILY: {family}. Start from the supplied corrected baseline; '
            'keep its backbone family throughout search, including initial proposals, merging and debugging. '
            'Do not fall back to SASRec or another model family. Keep seed 42, maximum epochs 200, '
            'batches per epoch 200 and the supplied splits/evaluator fixed. All models use LEFT-padded '
            'histories: the readout must use the final nonpadding position, never count-1. '
            'BERT4Rec uses a reserved mask token with masked-target training and appended-mask inference; '
            'do not expose a shifted prediction target through bidirectional input. Preserve checkpoint '
            'and embedded predictor consistency. Diagnosis uses family-specific sample_batch and ModelView; '
            'keep log2feats(sequence), item_emb and predict_score(features, items) compatible. '
            'One trained model/checkpoint per category; no independent score fusion.\n')
        scope = key or self.scope.get()
        with self.request_scope(scope):
            while True:
                try:
                    return super().llm(contract + SINGLE_MODEL_CONTRACT + '\n' + instruction,
                                       output_schema, search, key)
                except TransportUnavailable as exc:
                    self.event('llm_cooldown', scope=scope, error=str(exc), seconds=60)
                    time.sleep(60)

    def trial(self, key, source, gpu, requested=None, files=None, smoke=False):
        if self.method != 'mlestar':
            return super().trial(key, source, gpu, requested, files, smoke)
        # MLE-STAR parses trajectory IDs from trial IDs; retain their identities.
        path = self.root/('smoke' if smoke else 'trials')/key
        if path.exists():
            receipt = path/'receipt.json'
            if not receipt.exists() or (path/'train.py').read_text() != source:
                raise RuntimeError('Incomplete or mismatched MLE-STAR trial requires review')
            record = read(receipt)
            if record['status'] == 'success' and (self.best is None or record['score'] > self.best['score']):
                self.best = record
            return record
        return Controller.trial(self, key, source, gpu, requested, files, smoke)

    def llm_candidates(self, instruction, items, key):
        def one(item):
            response = self.llm(instruction + '\nGenerate ONLY this trajectory; preserve its exact ID.\n' + json.dumps(item),
                schema({'id': {'type': 'string', 'enum': [item['id']]}, 'hypothesis': TEXT, 'source': TEXT}),
                key=key+'-'+item['id'])
            return response
        with ThreadPoolExecutor(max_workers=min(4, len(items))) as pool:
            return {'candidates': list(pool.map(one, items))}
