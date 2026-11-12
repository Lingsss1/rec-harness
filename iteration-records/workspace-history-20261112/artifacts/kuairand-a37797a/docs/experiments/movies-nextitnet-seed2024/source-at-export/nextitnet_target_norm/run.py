"""Supplementary four-coefficient experiment for D's target-normalization plan.

Four fresh trainings are accounted separately from the historical 44-call search.
All four comparisons are fixed before fitting. The preferred model is selected on
validation HR@10 before the fixed comparison checkpoints are tested.
"""
import argparse
from pathlib import Path

import numpy as np

from mle_star_codex.run import Controller, read, save, sha, source_parts
from mle_star_codex.worker import CHECK_VERSION

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = WORKSPACE / 'artifacts/staged-d-target-norm-2024-v2'
COEFFICIENTS = [('beta-0', 0.), ('beta-1', 1.), ('beta-half', .5), ('beta-minus-half', -.5)]


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    c = read(WORKSPACE / 'artifacts/mle-star-movies-2024/config.json')
    c.update(run_dir=str(root), method='Staged-D target-item normalization follow-up',
        check_version=CHECK_VERSION, supplementary_training_calls=4,
        primary_comparison='beta=1 versus beta=0',
        fixed_test_comparisons=[key for key, _ in COEFFICIENTS])
    save(root / 'config.json', c)
    controller = Controller(c)
    template_path = Path(__file__).with_name('model.py')
    template = template_path.read_text()
    parent = WORKSPACE / 'artifacts/staged-movies-2024-aligned44/study/trajectories/movies_nextitnet/D/1/artifacts/t023/model.py'
    specs, sources = [], {}
    for key, beta in COEFFICIENTS:
        assert template.count('target_norm_beta: float=1.0') == 1
        source = template.replace('target_norm_beta: float=1.0', 'target_norm_beta: float=' + repr(beta))
        source_parts(source)
        path = controller.code / (key + '.py')
        path.write_text(source)
        sources[key] = path
        specs.append({'key': key, 'model': path})
    save(root / 'frozen.json', {'config': c, 'parent_source': str(parent), 'parent_sha256': sha(parent),
        'coefficients': dict(COEFFICIENTS), 'seed': 2024, 'check_version': CHECK_VERSION,
        'sources': {key: sha(path) for key, path in sources.items()},
        'implementation': {str(p): sha(p) for p in [Path(__file__), template_path,
            WORKSPACE / 'mle_star_codex/run.py', WORKSPACE / 'mle_star_codex/worker.py']},
        'historical_search_records_modified': False,
        'selection_rule': 'Highest validation HR@10; retain beta=0 on ties. All four fixed comparisons are tested independently of the selection.'})
    validation = controller.jobs(specs)
    for record, (_, beta) in zip(validation, COEFFICIENTS):
        record['beta'] = beta
    save(root / 'validation-results.json', validation)
    finish(root, controller, validation, sources)


def compare_control(historical, current):
    if historical.shape != current.shape or not np.isfinite(current).all():
        raise RuntimeError('Invalid zero-exponent control score matrix')
    historical_ranks = (historical[:, 1:] > historical[:, :1]).sum(1)
    current_ranks = (current[:, 1:] > current[:, :1]).sum(1)
    audit = {'max_absolute_score_difference': float(np.abs(historical - current).max()),
        'mean_absolute_score_difference': float(np.abs(historical - current).mean()),
        'all_target_ranks_identical': bool(np.array_equal(historical_ranks, current_ranks)),
        'float_atol': 1e-4, 'float_rtol': 1e-5,
        'scores_within_float_tolerance': bool(np.allclose(historical, current, atol=1e-4, rtol=1e-5))}
    if not audit['all_target_ranks_identical'] or not audit['scores_within_float_tolerance']:
        raise RuntimeError('Zero-exponent control changed rankings or exceeded float tolerance')
    return audit


def finish(root, controller, validation, sources):
    c = controller.c
    if any(record['status'] != 'success' for record in validation):
        raise RuntimeError('Normalization training failed; inspect retained worker logs')
    control = validation[0]
    if control['metrics']['HR@10'] != .685:
        raise RuntimeError('Zero-exponent control did not reproduce D trial 23; inspect before testing')
    historical = np.load(WORKSPACE / 'artifacts/staged-movies-2024-aligned44/study/trajectories/movies_nextitnet/D/1/validation/t023/adapter_scratch/training/scores.npy')
    current = np.load(root / 'trials/beta-0/scores.npy')
    control_audit = compare_control(historical, current)
    save(root / 'control-audit.json', control_audit)
    winner = max(validation, key=lambda record: record['metrics']['HR@10'])
    save(root / 'selection.json', {'selected_id': winner['id'], 'beta': winner['beta'],
        'validation_hr10': winner['metrics']['HR@10'], 'selected_before_test': True})
    test_specs = [{'key': 'test-' + key, 'model': sources[key], 'mode': 'test',
                   'checkpoint': root / 'trials' / key / 'checkpoint.pt'} for key, _ in COEFFICIENTS]
    tests = controller.jobs(test_specs, final=True)
    save(root / 'test-results.json', tests)
    if any(record['status'] != 'success' for record in tests):
        raise RuntimeError('Final comparison test failed; retain the recorded outcomes')
    comparisons = [{'id': key, 'beta': beta, 'validation': val['metrics'], 'test': test['metrics'],
        'selected': key == winner['id']} for (key, beta), val, test in zip(COEFFICIENTS, validation, tests)]
    summary = {'status': 'complete', 'method': c['method'], 'seed': 2024, 'check_version': CHECK_VERSION,
        'training_calls_added': 4, 'test_calls_added': 4, 'comparisons': comparisons,
        'selected_id': winner['id'], 'allocated_gpu_seconds': controller.spent(),
        'control_reproduces_d_trial_23_rankings': True, 'control_audit': control_audit,
        'limitations': ['Supplementary experiment, outside the historical 44-candidate budget.',
            'One fixed training seed and previously used validation/test split.',
            'Normalization changes both geometry and BPR score scale; performance alone does not establish the mechanism.']}
    save(root / 'summary.json', summary)
    lines = ['# D target-item normalization follow-up', '',
        '| beta | Validation HR@10 | Test HR@10 | Selected by validation |', '|---:|---:|---:|---|']
    lines += ['| %g | %.4f | %.4f | %s |' % (r['beta'], r['validation']['HR@10'], r['test']['HR@10'],
        'yes' if r['selected'] else '') for r in comparisons]
    lines += ['', 'Four additional trainings are separate from the original study. Primary comparison: full normalization versus no normalization. No test scores were used for selection.', '']
    (root / 'report.md').write_text('\n'.join(lines))
    print('D_TARGET_NORMALIZATION_COMPLETE', flush=True)


def finish_existing(root):
    if (root / 'summary.json').exists():
        raise RuntimeError('Supplementary experiment already completed')
    frozen = read(root / 'frozen.json')
    config = read(root / 'config.json')
    if config != frozen['config']:
        raise RuntimeError('Frozen configuration changed')
    for name, expected in frozen['implementation'].items():
        if Path(name).resolve() != Path(__file__).resolve() and sha(name) != expected:
            raise RuntimeError('Training/model/evaluator implementation changed: ' + name)
    expected_runner = frozen['implementation'][str(Path(__file__))]
    if sha(__file__) != expected_runner:
        backup = root / 'run-before-audit-repair.py'
        if not backup.exists() or sha(backup) != expected_runner:
            raise RuntimeError('Original follow-up runner was not archived')
        save(root / 'audit-repair.json', {'reason': 'Cross-run FP32 score tolerance was too strict; exact per-user target ranks and all metrics match. Model/training/evaluator are unchanged.',
            'original_runner_sha256': expected_runner, 'corrected_runner_sha256': sha(__file__),
            'additional_training_calls': 0, 'coefficients_changed': False, 'selection_rule_changed': False})
    validation = read(root / 'validation-results.json')
    if [r['id'] for r in validation] != [key for key, _ in COEFFICIENTS]:
        raise RuntimeError('Validation results do not match the fixed comparisons')
    sources = {key: root / 'models' / (key + '.py') for key, _ in COEFFICIENTS}
    for record in validation:
        key = record['id']
        if sha(sources[key]) != frozen['sources'][key] or sha(root / 'trials' / key / 'checkpoint.pt') != record['checkpoint_sha256']:
            raise RuntimeError('Frozen source/checkpoint changed')
    finish(root, Controller(config), validation, sources)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--finish', action='store_true', help='Finish saved successful trainings without fitting again')
    args = parser.parse_args()
    if args.finish:
        finish_existing(args.root.resolve())
    else:
        run(args.root.resolve())


if __name__ == '__main__':
    main()
