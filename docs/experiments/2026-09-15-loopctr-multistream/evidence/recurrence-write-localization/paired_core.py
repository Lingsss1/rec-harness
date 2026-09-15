"""ten-model, shared-user-draw metric core; no file IO or execution on import.

The caller supplies the accepted AUC and BCE helpers and already aligned arrays.
This module is not a quality-admission gate or a model winner selector.
"""
import hashlib
import itertools

LABELS = ('is_click', 'cart', 'fav', 'buy')
NEW_MODELS = ('first_only', 'second_only', 'collapsed_write', 'tied_write', 'no_static_routes', 'first_only_no_routes', 'second_only_no_routes', 'collapsed_no_routes')
MODEL_ORDER = ('cross8', 'dynamic_write', 'first_only', 'second_only', 'collapsed_write', 'tied_write', 'no_static_routes', 'first_only_no_routes', 'second_only_no_routes', 'collapsed_no_routes')
EXPECTED_EFFECTS = (('dynamic_write__minus__cross8', {'dynamic_write': 1, 'cross8': -1}), ('first_only__minus__cross8', {'first_only': 1, 'cross8': -1}), ('second_only__minus__cross8', {'second_only': 1, 'cross8': -1}), ('collapsed_write__minus__cross8', {'collapsed_write': 1, 'cross8': -1}), ('tied_write__minus__cross8', {'tied_write': 1, 'cross8': -1}), ('no_static_routes__minus__cross8', {'no_static_routes': 1, 'cross8': -1}), ('first_only_no_routes__minus__cross8', {'first_only_no_routes': 1, 'cross8': -1}), ('second_only_no_routes__minus__cross8', {'second_only_no_routes': 1, 'cross8': -1}), ('collapsed_no_routes__minus__cross8', {'collapsed_no_routes': 1, 'cross8': -1}), ('first_only__minus__dynamic_write', {'first_only': 1, 'dynamic_write': -1}), ('second_only__minus__dynamic_write', {'second_only': 1, 'dynamic_write': -1}), ('collapsed_write__minus__dynamic_write', {'collapsed_write': 1, 'dynamic_write': -1}), ('tied_write__minus__dynamic_write', {'tied_write': 1, 'dynamic_write': -1}), ('no_static_routes__minus__dynamic_write', {'no_static_routes': 1, 'dynamic_write': -1}), ('first_only_no_routes__minus__dynamic_write', {'first_only_no_routes': 1, 'dynamic_write': -1}), ('second_only_no_routes__minus__dynamic_write', {'second_only_no_routes': 1, 'dynamic_write': -1}), ('collapsed_no_routes__minus__dynamic_write', {'collapsed_no_routes': 1, 'dynamic_write': -1}), ('second_only__minus__first_only', {'second_only': 1, 'first_only': -1}), ('collapsed_write__minus__first_only', {'collapsed_write': 1, 'first_only': -1}), ('tied_write__minus__first_only', {'tied_write': 1, 'first_only': -1}), ('no_static_routes__minus__first_only', {'no_static_routes': 1, 'first_only': -1}), ('first_only_no_routes__minus__first_only', {'first_only_no_routes': 1, 'first_only': -1}), ('second_only_no_routes__minus__first_only', {'second_only_no_routes': 1, 'first_only': -1}), ('collapsed_no_routes__minus__first_only', {'collapsed_no_routes': 1, 'first_only': -1}), ('collapsed_write__minus__second_only', {'collapsed_write': 1, 'second_only': -1}), ('tied_write__minus__second_only', {'tied_write': 1, 'second_only': -1}), ('no_static_routes__minus__second_only', {'no_static_routes': 1, 'second_only': -1}), ('first_only_no_routes__minus__second_only', {'first_only_no_routes': 1, 'second_only': -1}), ('second_only_no_routes__minus__second_only', {'second_only_no_routes': 1, 'second_only': -1}), ('collapsed_no_routes__minus__second_only', {'collapsed_no_routes': 1, 'second_only': -1}), ('tied_write__minus__collapsed_write', {'tied_write': 1, 'collapsed_write': -1}), ('no_static_routes__minus__collapsed_write', {'no_static_routes': 1, 'collapsed_write': -1}), ('first_only_no_routes__minus__collapsed_write', {'first_only_no_routes': 1, 'collapsed_write': -1}), ('second_only_no_routes__minus__collapsed_write', {'second_only_no_routes': 1, 'collapsed_write': -1}), ('collapsed_no_routes__minus__collapsed_write', {'collapsed_no_routes': 1, 'collapsed_write': -1}), ('no_static_routes__minus__tied_write', {'no_static_routes': 1, 'tied_write': -1}), ('first_only_no_routes__minus__tied_write', {'first_only_no_routes': 1, 'tied_write': -1}), ('second_only_no_routes__minus__tied_write', {'second_only_no_routes': 1, 'tied_write': -1}), ('collapsed_no_routes__minus__tied_write', {'collapsed_no_routes': 1, 'tied_write': -1}), ('first_only_no_routes__minus__no_static_routes', {'first_only_no_routes': 1, 'no_static_routes': -1}), ('second_only_no_routes__minus__no_static_routes', {'second_only_no_routes': 1, 'no_static_routes': -1}), ('collapsed_no_routes__minus__no_static_routes', {'collapsed_no_routes': 1, 'no_static_routes': -1}), ('second_only_no_routes__minus__first_only_no_routes', {'second_only_no_routes': 1, 'first_only_no_routes': -1}), ('collapsed_no_routes__minus__first_only_no_routes', {'collapsed_no_routes': 1, 'first_only_no_routes': -1}), ('collapsed_no_routes__minus__second_only_no_routes', {'collapsed_no_routes': 1, 'second_only_no_routes': -1}))
REPETITIONS = 500
SEED = 20260906
TASK_PAIRS = tuple(itertools.combinations(range(4), 2))
METRIC_ORDER = ('macro_auc', *(f'auc.{x}' for x in LABELS),
                *(f'bce.{x}' for x in LABELS),
                *(f'auc.{LABELS[i]}_minus_{LABELS[j]}' for i,j in TASK_PAIRS),
                *(f'bce.{LABELS[i]}_minus_{LABELS[j]}' for i,j in TASK_PAIRS))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def comparison_spec(protocol):
    statistics=protocol['statistics']
    require(tuple(statistics['model_order'])==MODEL_ORDER,'Model order differs')
    require(tuple(statistics['metrics_per_effect'])==METRIC_ORDER,'Metric order differs')
    actual=tuple((e['name'],e['coefficients']) for e in statistics['comparisons'])
    require(actual==EXPECTED_EFFECTS,'Precommitted contrasts or interaction changed')
    names=[n for n,c in actual]
    vectors=[[c.get(n,0) for n in MODEL_ORDER] for name,c in actual]
    require(len(names)==len(set(names))==45 and all(sum(v)==0 for v in vectors),'Invalid contrasts')
    return names,vectors


def effect_metrics(auc, bce, coefficients):
    """Linear effects applied before task contrasts and confidence intervals.

    Input last dimensions are [model, task]; optional leading draw dimensions
    are retained. Output last dimensions are [effect, 21 metrics].
    """
    import numpy as np
    require(auc.shape == bce.shape and auc.shape[-2:] == (coefficients.shape[1], 4), 'Metric tensor shape')
    da = np.einsum('em,...mt->...et', coefficients, auc)
    db = np.einsum('em,...mt->...et', coefficients, bce)
    return np.concatenate((da.mean(axis=-1, keepdims=True), da, db,
        np.stack([da[...,i]-da[...,j] for i,j in TASK_PAIRS], axis=-1),
        np.stack([db[...,i]-db[...,j] for i,j in TASK_PAIRS], axis=-1)), axis=-1)


def shared_bootstrap(arrays, users, groups, auc_plan, weighted_auc, bce_rows,
                     progress=lambda value: None, repetitions=REPETITIONS, seed=SEED):
    """Compute each model's metrics once in each common user draw.

    Nondefault repetitions/seed exist only for bounded synthetic verification;
    the execution wrapper must enforce the production constants. An undefined
    metric fails the run; no replacement draw or class-conditioned resampling.
    """
    import numpy as np
    model_count = len(MODEL_ORDER)
    require(len(arrays) == model_count and len(users) > 1 and len(groups) > 0, 'Aligned ten-model inputs required')
    n, u = len(groups), len(users)
    plans = [[auc_plan(d['prediction_'+label], d['label_'+label], groups)
              for label in LABELS] for d in arrays]
    counts = np.bincount(groups, minlength=u)
    sums = np.empty((u, model_count, 4), dtype=np.float64)
    for m, d in enumerate(arrays):
        for t, label in enumerate(LABELS):
            sums[:, m, t] = np.bincount(groups,
                weights=bce_rows(d['prediction_'+label], d['label_'+label]), minlength=u)
    ones = np.ones(u, dtype=np.int64)
    point_auc = np.asarray([[weighted_auc(plan, ones) for plan in row] for row in plans])
    point_bce = sums.sum(axis=0) / n
    require(np.isfinite(point_auc).all() and np.isfinite(point_bce).all(), 'Undefined observed metrics')
    # The caller checks source metrics at this point, before any draw is used.
    progress(dict(stage='observed', observed_auc=point_auc, observed_bce=point_bce))
    draws_auc = np.empty((repetitions, model_count, 4), dtype=np.float64)
    draws_bce = np.empty_like(draws_auc)
    row_counts = np.empty(repetitions, dtype=np.int64)
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256(f'seed={seed};repetitions={repetitions};users={u}'.encode())
    flat_sums = sums.reshape(u, model_count * 4)
    for index in range(repetitions):
        indices = rng.integers(len(users), size=len(users))
        digest.update(indices.tobytes())
        weights = np.bincount(indices, minlength=len(users))
        row_counts[index] = weights @ counts
        require(row_counts[index] > 0, 'Empty resampled rows')
        draws_auc[index] = [[weighted_auc(plan, weights) for plan in row] for row in plans]
        draws_bce[index] = (weights @ flat_sums).reshape(model_count,4) / row_counts[index]
        require(np.isfinite(draws_auc[index]).all() and np.isfinite(draws_bce[index]).all(), 'Undefined draw metric')
        if (index + 1) % 25 == 0:
            progress(dict(stage='draws', draws_completed=index+1))
    return dict(point_auc=point_auc, point_bce=point_bce, draw_auc=draws_auc,
                draw_bce=draws_bce, draw_row_count=row_counts,
                resample_indices_sha256=digest.hexdigest())
