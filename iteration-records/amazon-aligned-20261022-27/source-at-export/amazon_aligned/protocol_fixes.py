"""Versioned safeguards for the fresh, explicitly authorized October 26 reruns."""
import ast
from copy import deepcopy
import json

from .prepare import WORKSPACE

SINGLE_MODEL_CONTRACT = '''
MANDATORY SINGLE-MODEL EXPERIMENT: train one independent model per category.
Do not implement ensembles, bagging, boosting of independent predictors, stacking,
multi-seed/checkpoint averaging, score/rank voting, mixtures of independently fitted
experts, or neural scores combined with separately fitted transition/popularity scores.
Each category must load exactly one independently trained model/checkpoint at prediction.
Jointly trained internal layers/features of one network are allowed. Four category
models are necessary for routing and do not constitute an ensemble. All hypotheses,
code generation, debugging, reviews and memory use must respect this constraint even
if an upstream generic prompt suggests an ensemble. The host audits the actual
predictor before accepting validation scores; violations fail the trial and consume
their training budget. Do not read other trials or reuse their trained checkpoints.
'''


def restore_diagnosis_prompt(source):
    """Restore only the pinned presentation function, preserving evolved probes."""
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == 'diagnosis_interpreter_prompt']
    if functions and len(functions[-1].args.args) == 1:
        return source, False
    path = WORKSPACE / 'artifacts/Self-EvolveRec-code/examples/sasrec_cds/initial_code/diagnosis_tools.py'
    original = path.read_text()
    node = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)
                and n.name == 'diagnosis_interpreter_prompt')
    pinned = ast.get_source_segment(original, node)
    # The upstream function uses json, which evolved probe files may omit.
    return source.rstrip() + '\n\n# Host-restored upstream diagnostic presentation contract.\nimport json\n' + pinned + '\n', True


def compact_diagnosis(raw, max_chars=120000):
    """Bound example-level records; retain measured headline values and definitions."""
    def compact(value, width, depth=0):
        if isinstance(value, dict):
            # The metric/definition names at the first two levels are never dropped.
            items = list(value.items())
            chosen = items if depth < 2 or len(items) <= width else items[:width]
            result = {k: compact(v, width, depth + 1) for k, v in chosen}
            if len(chosen) < len(items):
                result['__host_omitted_entries__'] = len(items) - len(chosen)
            return result
        if isinstance(value, (list, tuple)):
            result = [compact(v, width, depth + 1) for v in value[:width]]
            if len(value) > width:
                result.append({'__host_omitted_entries__': len(value) - width,
                               'original_count': len(value)})
            return result
        if isinstance(value, str) and len(value) > 4000:
            return value[:4000] + '\n[host omitted remaining explanatory text]'
        return value
    original_chars = len(json.dumps(raw, ensure_ascii=False))
    if original_chars <= max_chars:
        return raw, {'compacted': False, 'original_chars': original_chars}
    for width in [32, 16, 8, 4, 2, 1]:
        result = compact(raw, width)
        count = len(json.dumps(result, ensure_ascii=False))
        if count <= max_chars:
            return result, {'compacted': True, 'original_chars': original_chars,
                            'sent_chars': count, 'example_width': width,
                            'policy': 'preserve top-level metric names; cap nested raw examples, never fabricate measurements'}
    raise ValueError('Diagnostic payload remains too large; reduce per-example diagnostic output')


def diagnosis_presentation(module, raw):
    """Keep the actual rendered prompt below the transport limit, including indent."""
    for limit in [120000, 60000, 20000]:
        payload, info = compact_diagnosis(raw, max_chars=limit)
        prompt = module.diagnosis_interpreter_prompt(payload)
        if not isinstance(prompt, str):
            raise ValueError('diagnosis_interpreter_prompt must return a string')
        if len(prompt) <= 650000:
            return prompt, payload, {**info, 'rendered_prompt_chars': len(prompt)}
    raise ValueError('Diagnostic prompt is too large; use measured summary statistics instead of raw examples')


def bounded_history(history, max_chars=700000):
    """Drop complete old exchanges; never truncate the current code or diff."""
    result = deepcopy(history)
    removed = 0
    while len(json.dumps(result, ensure_ascii=False)) > max_chars and len(result) > 1:
        result.pop(0)
        removed += 1
        while len(result) > 1 and result[0].get('role') != 'user':
            result.pop(0)
            removed += 1
    if len(json.dumps(result, ensure_ascii=False)) > max_chars:
        raise ValueError('Current native message exceeds the bounded transport; reduce raw diagnostic examples or inspirations')
    return result, removed
