"""Batch-prevalence positive-label source at a fixed forward architecture.

Cross native versus batch-mean auxiliary labels in the positive-label-only
source VJP with both external orientations. The source objective is
BCE(y_source)-BCE(0), differentiated through recipient probabilities.
Only its detached cotangents determine the inherited one-bit orientation;
true-label joint BCE, private learning, native transfer line/radius/support,
forward computation and initialization stay fixed. Native, cycled, mean
and zero source graphs execute in every cell. This is a first-order
surrogate VJP, not the ordinary gradient of the reported scalar BCE.
No effectiveness or novelty claim is established by these modules.
"""
import torch
from torch import nn
from unirank.pytorch.models import MultiTaskModel
from unirank.pytorch.layers import FeatureEmbeddingDict
from unirank.pytorch.layers.activations import Dice


CONFIG = {'embedding_dim': 16, 'attention_hidden_units': [64, 32], 'tower_hidden_units': [128, 64], 'attention_normalization': 'none', 'compression_routes': 'uf', 'block_aux_scorer_grad': True, 'block_aux_context_grad': False, 'block_aux_item_grad': True, 'residual_input': 'match_detail', 'residual_routing': 'task', 'residual_hidden': 16, 'temporal_detail': 'position', 'block_detail': 'ordinal', 'curvature_detail': 'quadratic', 'joint_detail': 'duplicate_match', 'match_support': 'available', 'match_scaling': 'norm', 'coarse_extension': 'duplicate', 'primary_partition': 'uniform', 'normalization_support': 'rotated_duplicate', 'support_gradient': 'attached', 'inference_support': 'fully_pruned', 'training_support': 'folded', 'residual_execution': 'direct', 'auxiliary_train_population': 'primary', 'auxiliary_eval_population': 'primary', 'block_aux_tower_context_grad': True, 'block_aux_adapter_context_grad': False, 'context_fusion': 'product', 'interaction_rank': 16, 'block_aux_residual_context_grad': True, 'block_aux_product_context_grad': False, 'product_context_assignment': 'aligned', 'product_context_component': 'centered', 'product_context_magnitude': 'native', 'product_context_orientation': 'aligned', 'parallel_orientation': 'aligned', 'orthogonal_orientation': 'reversed', 'centering_scope': 'aggregate', 'click_context_collection': 'joint', 'auxiliary_reduction_precision': 'float64', 'click_relative_transfer': 'orthogonal_native', 'orthogonal_assignment': 'nuisance_line_fixed_angle_decomposed_label_bit', 'source_component': 'positive_label', 'source_label_assignment': 'batch_mean'}

# Preserve true-label joint private learning; only the sign-source VJP
# receives the positive-label-force versus full-source intervention.
AUXILIARY_OBJECTIVE = 'joint'


def _auxiliary_context_components(gradients, occupied):
    """FP64 components before the single cast at the collection boundary.

    The ordered task sum is G=(cart+fav)+buy, restricted to occupied rows.
    M repeats the occupied-row mean of G; C=G-M on those rows. Empty rows
    stay zero. These are row-space components BEFORE repeated-ID embedding
    scatter, not orthogonal parameter-space gradients. All components are
    computed in every cell; no extra modules, state or RNG are introduced.
    """
    if len(gradients) != 3:
        raise ValueError('Expected cart/fav/buy cotangents')
    if occupied.shape != (gradients[0].shape[0], 1) or occupied.dtype != torch.bool:
        raise ValueError('Expected history occupancy bool[B,1]')
    if any(g.shape != gradients[0].shape for g in gradients):
        raise ValueError('Auxiliary cotangent shape mismatch')
    with torch.autocast(device_type=gradients[0].device.type, enabled=False):
        active = [torch.where(occupied, g.to(dtype=torch.float64), 0.0)
                  for g in gradients]
        count = occupied.to(dtype=torch.float64).sum().clamp_min(1.0)
        total = (active[0] + active[1]) + active[2]
        row_mean = total.sum(dim=0, keepdim=True) / count
        centered = torch.where(occupied, total - row_mean, 0.0)
        mean = torch.where(occupied, row_mean, 0.0)
        blocked = torch.zeros_like(total)
    return {'full': total, 'centered': centered, 'mean': mean, 'blocked': blocked}


def _click_relative_components(click_gradient, gradients, occupied):
    """Global FP64 orthogonal split before sparse embedding accumulation.

    K is the occupied-row-centered *click-only* context cotangent. P is
    proj_K(C), O=C-P. Centering K keeps both components in C's zero-mean
    occupied subspace. Scale K by its maximum absolute entry before the
    projection to handle small nonzero reference magnitudes. If K=0,
    define P=0 and O=C. This is not a per-row or parameter-space split.
    """
    components = _auxiliary_context_components(gradients, occupied)
    centered = components['centered']
    if click_gradient.shape != centered.shape:
        raise ValueError('Click and auxiliary context cotangents must align')
    with torch.autocast(device_type=click_gradient.device.type, enabled=False):
        click = torch.where(occupied, click_gradient.to(dtype=torch.float64), 0.0)
        count = occupied.to(dtype=torch.float64).sum().clamp_min(1.0)
        reference = torch.where(occupied, click - click.sum(dim=0, keepdim=True) / count, 0.0)
        tiny = torch.finfo(torch.float64).tiny
        scale = reference.abs().amax().clamp_min(tiny)
        reference = reference / scale
        reference_norm2 = reference.square().sum()
        coefficient = (centered * reference).sum() / reference_norm2.clamp_min(tiny)
        parallel = coefficient * reference
        orthogonal = centered - parallel
    return centered, parallel, orthogonal


def _norm_matched_projections(centered, parallel, orthogonal):
    """Normalize both nonzero components to ||C||_F, entirely in FP64.

    Scaling by each maximum before normalization avoids division by a tiny
    squared norm. Exact zero maps to zero; no epsilon floor, fallback C,
    arbitrary direction, RNG or new state is introduced. Near-zero P can
    expose the sign of a small projection: record norm ratios on the host.
    This helper runs for every arm, including native and blocked controls.
    """
    with torch.autocast(device_type=centered.device.type, enabled=False):
        target_scale = centered.abs().amax()
        safe_target_scale = torch.where(
            target_scale > 0, target_scale, torch.ones_like(target_scale))
        target_unit = centered / safe_target_scale
        target_norm = target_scale * target_unit.square().sum().sqrt()
        matched = []
        for component in (parallel, orthogonal):
            scale = component.abs().amax()
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            unit = component / safe_scale
            unit_norm = unit.square().sum().sqrt()
            safe_norm = torch.where(
                unit_norm > 0, unit_norm, torch.ones_like(unit_norm))
            matched.append((unit / safe_norm) * target_norm)
    return matched[0], matched[1]


def _reassign_orthogonal(click_gradient, orthogonal, occupied):
    """Pair-exchange O, restore row geometry, then match native ||O||.

    The fixed half-batch involution exchanges rows only when BOTH members
    have a history. Empty rows remain zero. Restoring zero mean and global
    K orthogonality after permutation changes more than assignment alone.
    All-empty, zero-K and zero-O inputs are defined. An exactly zero
    reprojected candidate remains zero; the host must report its frequency
    and the norm-restoration scale before interpreting the contrast.
    """
    with torch.autocast(device_type=click_gradient.device.type, enabled=False):
        half = orthogonal.shape[0] // 2
        eligible = occupied & torch.roll(occupied, shifts=half, dims=0)
        swapped = torch.where(
            eligible, torch.roll(orthogonal, shifts=half, dims=0), orthogonal)
        count = occupied.to(dtype=torch.float64).sum().clamp_min(1.0)
        swapped = torch.where(
            occupied, swapped - swapped.sum(dim=0, keepdim=True) / count, 0.0)
        click = torch.where(occupied, click_gradient.to(dtype=torch.float64), 0.0)
        reference = torch.where(
            occupied, click - click.sum(dim=0, keepdim=True) / count, 0.0)
        tiny = torch.finfo(torch.float64).tiny
        reference = reference / reference.abs().amax().clamp_min(tiny)
        coefficient = ((swapped * reference).sum()
                       / reference.square().sum().clamp_min(tiny))
        restored = swapped - coefficient * reference
        # Robust Frobenius normalization without a squared-norm epsilon.
        target_scale = orthogonal.abs().amax()
        safe_target_scale = torch.where(
            target_scale > 0, target_scale, torch.ones_like(target_scale))
        target_norm = target_scale * (orthogonal / safe_target_scale).square().sum().sqrt()
        scale = restored.abs().amax()
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        unit = restored / safe_scale
        unit_norm = unit.square().sum().sqrt()
        safe_norm = torch.where(unit_norm > 0, unit_norm, torch.ones_like(unit_norm))
        replacement = (unit / safe_norm) * target_norm
    return replacement


def _tangent_orthogonal(click_gradient, orthogonal, occupied, replacement):
    """Remove native-direction overlap from the reassigned replacement.

    All tensors below are FP64. Build an orthonormal basis for centered K
    and native O using two modified Gram-Schmidt passes. Project the paired
    replacement out of that basis, twice, then restore ||O||_F. The result
    is centered on occupied rows and orthogonal to both K and O up to FP64
    rounding, BEFORE the single inherited cast/scatter/optimizer boundary.
    Zero basis/replacement maps to zero. If the pre-normalization tangent
    norm is less than 1e-6 of ||O||, use zero instead of amplifying a nearly
    degenerate residual. This is a declared validity gate, not a fallback
    direction. Host diagnostics must record its rate, norm ratios and
    post-cast residuals.
    This is a constrained surrogate direction, not a pure row permutation.
    """
    with torch.autocast(device_type=click_gradient.device.type, enabled=False):
        count = occupied.to(dtype=torch.float64).sum().clamp_min(1.0)

        def center(value):
            value = torch.where(occupied, value.to(dtype=torch.float64), 0.0)
            return torch.where(occupied, value - value.sum(dim=0, keepdim=True) / count, 0.0)

        def unit_and_norm(value):
            scale = value.abs().amax()
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            scaled = value / safe_scale
            length = scaled.square().sum().sqrt()
            safe_length = torch.where(length > 0, length, torch.ones_like(length))
            return scaled / safe_length, scale * length

        click_basis, _ = unit_and_norm(center(click_gradient))
        native = center(orthogonal)
        for _ in range(2):
            native = native - (native * click_basis).sum() * click_basis
        native_basis, _ = unit_and_norm(native)
        tangent = center(replacement)
        for _ in range(2):
            tangent = tangent - (tangent * click_basis).sum() * click_basis
            tangent = tangent - (tangent * native_basis).sum() * native_basis
        tangent_unit, tangent_norm = unit_and_norm(tangent)
        _, native_norm = unit_and_norm(orthogonal)
        valid = (native_norm > 0) & (tangent_norm >= 1e-6 * native_norm)
        return torch.where(valid, tangent_unit * native_norm, torch.zeros_like(tangent))


def _canonical_tangent(tangent):
    """Fix only the batch-global sign of Q, retaining exactly its line.

    Use the first maximal-absolute flattened entry (torch.argmax tie rule).
    A negative pivot reverses Q; zero Q uses +1 and stays zero. This has no
    CPU scalar extraction, RNG, new tensor state or extra model evaluation.
    Global Q reversal leaves the selected index and canonical U invariant,
    including equal-magnitude ties; signed zero carries no orientation.
    The coordinate rule is arbitrary and order dependent, not a semantic
    positive direction. Host must report pivot ties/near ties, sign balance,
    parity/norm/private-gradient checks and inherited tangent degeneracy.
    """
    with torch.autocast(device_type=tangent.device.type, enabled=False):
        flat = tangent.reshape(-1)
        index = flat.abs().argmax().reshape(1)
        pivot = torch.gather(flat, dim=0, index=index).squeeze(0)
        sign = torch.where(pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot))
        return tangent * sign


def _fixed_coordinate_carrier(orthogonal, occupied):
    """A state-free modular-coordinate carrier, centered and norm matched.

    Two Park-Miller modular multiplications map flattened row/field
    coordinates to fixed +/-1 entries. This is an arbitrary deterministic
    carrier, not isotropic noise or a sample-ID hash. Values stay within
    int64 range. No generator, input IDs, labels or persistent state enter.
    Its radius is native ||O||, so this control retains auxiliary magnitude.
    All-empty, zero-O and constant occupied carriers map to zero.
    """
    with torch.autocast(device_type=orthogonal.device.type, enabled=False):
        index = torch.arange(1, orthogonal.numel() + 1,
                             device=orthogonal.device, dtype=torch.int64)
        hashed = (index * 48271).remainder(2147483647)
        hashed = (hashed * 48271).remainder(2147483647)
        carrier = (2 * hashed.remainder(2) - 1).to(dtype=torch.float64)
        carrier = carrier.reshape_as(orthogonal)
        carrier = torch.where(occupied, carrier, 0.0)
        count = occupied.to(dtype=torch.float64).sum().clamp_min(1.0)
        carrier = torch.where(
            occupied, carrier - carrier.sum(dim=0, keepdim=True) / count, 0.0)
        target_scale = orthogonal.abs().amax()
        safe_target_scale = torch.where(
            target_scale > 0, target_scale, torch.ones_like(target_scale))
        target_norm = target_scale * (orthogonal / safe_target_scale).square().sum().sqrt()
        scale = carrier.abs().amax()
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        unit = carrier / safe_scale
        length = unit.square().sum().sqrt()
        safe_length = torch.where(length > 0, length, torch.ones_like(length))
        return (unit / safe_length) * target_norm


def _fixed_line_with_orientation(click_gradient, orthogonal, occupied, tangent):
    """Replace Q's line while transporting its coordinate orientation bit.

    Project the fixed carrier out of centered K/native O using precisely
    the inherited tangent helper, including its 1e-6 degeneracy gate.
    Canonicalize that line V and return s(Q)*V. s uses the same first-max
    pivot convention as _canonical_tangent. For Q=0 return zero, even if V
    exists. If V degenerates while Q does not, return zero and require a
    host validity report; do not silently use another direction.

    At fixed click/state, V is even under simultaneous auxiliary-gradient
    reversal and s(Q) is odd for nonzero Q, hence R is odd away from
    finite-precision ties/gates. Auxiliary information remains in ||O||,
    the subspace excluding O and the one transported bit. This does not
    remove all labels, preserve covariance or define isotropic noise.
    """
    carrier = _fixed_coordinate_carrier(orthogonal, occupied)
    fixed_tangent = _tangent_orthogonal(
        click_gradient, orthogonal, occupied, carrier)
    canonical_fixed = _canonical_tangent(fixed_tangent)
    with torch.autocast(device_type=tangent.device.type, enabled=False):
        flat = tangent.reshape(-1)
        pivot_index = flat.abs().argmax().reshape(1)
        pivot = torch.gather(flat, dim=0, index=pivot_index).squeeze(0)
        sign = torch.where(pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot))
        return torch.where(pivot != 0, canonical_fixed * sign,
                           torch.zeros_like(canonical_fixed))


def _click_only_fixed_line(click_gradient, orthogonal, occupied, tangent, fixed_oriented):
    """Remove native-O exclusion, retaining radius, pivot bit and zero gate.

    Center the same coordinate carrier and project out centered click K
    twice in FP64. Unlike the inherited fixed line, do not project out O.
    Match ||O|| and canonicalize with the same first-max pivot convention.
    Transfer s(Q), the inherited paired-tangent sign bit. Require the old
    oriented line to be nonzero so its validity gate is shared exactly.
    A new near-degeneracy yields zero and must invalidate a matched-radius
    interpretation unless the host demonstrates no support mismatch.

    This removes only O's projection constraint. Auxiliary norm, s(Q),
    validity gates and their training trajectories remain. Geometry here
    precedes cast, embedding scatter and adaptive optimization. This is
    a surrogate VJP, not the derivative of the reported joint scalar loss.
    """
    carrier = _fixed_coordinate_carrier(orthogonal, occupied)
    with torch.autocast(device_type=click_gradient.device.type, enabled=False):
        count = occupied.to(dtype=torch.float64).sum().clamp_min(1.0)

        def center(value):
            value = torch.where(occupied, value.to(dtype=torch.float64), 0.0)
            return torch.where(occupied, value - value.sum(dim=0, keepdim=True) / count, 0.0)

        def unit_and_norm(value):
            scale = value.abs().amax()
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            scaled = value / safe_scale
            length = scaled.square().sum().sqrt()
            safe_length = torch.where(length > 0, length, torch.ones_like(length))
            return scaled / safe_length, scale * length

        click_basis, _ = unit_and_norm(center(click_gradient))
        direction = center(carrier)
        for _ in range(2):
            direction = direction - (direction * click_basis).sum() * click_basis
        unit, length = unit_and_norm(direction)
        _, radius = unit_and_norm(orthogonal)
        valid = (radius > 0) & (length >= 1e-6 * radius)
        matched = torch.where(valid, unit * radius, torch.zeros_like(direction))
        canonical = _canonical_tangent(matched)
        flat = tangent.reshape(-1)
        pivot_index = flat.abs().argmax().reshape(1)
        pivot = torch.gather(flat, dim=0, index=pivot_index).squeeze(0)
        sign = torch.where(pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot))
        # The inherited line already includes the Q and K/O validity gates.
        old_active = fixed_oriented.abs().amax() > 0
        return torch.where(old_active & (pivot != 0), canonical * sign,
                           torch.zeros_like(canonical))


def _fixed_nuisance_carrier(orthogonal, occupied):
    """Second deterministic coordinate carrier, with the same native radius.

    Use the inherited two modular multiplications after the prespecified
    offset 104729. This offset is fixed before evaluation, not searched.
    The direction before radius matching uses only coordinates/occupancy.
    It is an arbitrary nuisance line, not independent isotropic noise.
    The auxiliary radius and support are intentionally retained.
    """
    with torch.autocast(device_type=orthogonal.device.type, enabled=False):
        index = torch.arange(1, orthogonal.numel() + 1,
                             device=orthogonal.device, dtype=torch.int64)
        index = index + 104729
        hashed = (index * 48271).remainder(2147483647)
        hashed = (hashed * 48271).remainder(2147483647)
        carrier = (2 * hashed.remainder(2) - 1).to(dtype=torch.float64)
        carrier = carrier.reshape_as(orthogonal)
        carrier = torch.where(occupied, carrier, 0.0)
        count = occupied.to(dtype=torch.float64).sum().clamp_min(1.0)
        carrier = torch.where(
            occupied, carrier - carrier.sum(dim=0, keepdim=True) / count, 0.0)
        target_scale = orthogonal.abs().amax()
        safe_target_scale = torch.where(
            target_scale > 0, target_scale, torch.ones_like(target_scale))
        target_norm = target_scale * (orthogonal / safe_target_scale).square().sum().sqrt()
        scale = carrier.abs().amax()
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        unit = carrier / safe_scale
        length = unit.square().sum().sqrt()
        safe_length = torch.where(length > 0, length, torch.ones_like(length))
        return (unit / safe_length) * target_norm


def _matched_nuisance_exclusion(click_gradient, orthogonal, occupied,
                                  carrier, nuisance):
    """Match the excluded line's norm and absolute overlap with F after K.

    U is a fixed-nuisance direction projected out of K and F with the
    inherited tangent helper. Let E be unit(F after removing K), a=<O,E>
    and b=||O-aE||. Set H=|a|E+bU_unit. Then ||H||=||O|| and
    |<H,E>|=|<O,E>| up to FP64 roundoff. This preserves an unoriented
    line's absolute angle; replacing O by -O leaves H invariant.
    Directional information beyond these scalar constraints is replaced.
    H still uses auxiliary magnitude, overlap and validity information.
    Failed F/U geometry returns zero with no alternate line or offset.
    """
    nuisance_tangent = _tangent_orthogonal(
        click_gradient, carrier, occupied, nuisance)
    with torch.autocast(device_type=click_gradient.device.type, enabled=False):
        count = occupied.to(dtype=torch.float64).sum().clamp_min(1.0)

        def center(value):
            value = torch.where(occupied, value.to(dtype=torch.float64), 0.0)
            return torch.where(occupied, value - value.sum(dim=0, keepdim=True) / count, 0.0)

        def unit_and_norm(value):
            scale = value.abs().amax()
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            scaled = value / safe_scale
            length = scaled.square().sum().sqrt()
            safe_length = torch.where(length > 0, length, torch.ones_like(length))
            return scaled / safe_length, scale * length

        click_basis, _ = unit_and_norm(center(click_gradient))
        reference = center(carrier)
        for _ in range(2):
            reference = reference - (reference * click_basis).sum() * click_basis
        reference_unit, reference_norm = unit_and_norm(reference)
        nuisance_unit, nuisance_norm = unit_and_norm(nuisance_tangent)
        signed_overlap = (orthogonal * reference_unit).sum()
        tail = orthogonal - signed_overlap * reference_unit
        _, tail_norm = unit_and_norm(tail)
        _, radius = unit_and_norm(orthogonal)
        valid = ((radius > 0) & (reference_norm >= 1e-6 * radius)
                 & (nuisance_norm > 0))
        matched = signed_overlap.abs() * reference_unit + tail_norm * nuisance_unit
        return torch.where(valid, matched, torch.zeros_like(matched)), valid


def _nuisance_line_with_orientation(click_gradient, orthogonal, occupied,
                                    tangent, fixed_oriented):
    """Replace O exclusion by H exclusion; retain F, radius, bit, old gate.

    H uses the new matched-nuisance helper, retaining O norm and absolute
    overlap with the K-residual of F. The inherited two-pass tangent
    helper projects F out of K/H and restores ||H|| = ||O|| in FP64.
    A separate rank gate
    requires H's residual after K projection to be at least 1e-6 ||H||;
    this prevents calling a collapsed basis a two-direction control.
    The old oriented line must be active, preserving its zero support.
    New rank/tangent degeneracy maps to zero, never a fallback direction.
    Host must report any support/norm mismatch before interpretation.
    Auxiliary norm, Q bit and native validity still enter; this does not
    remove all auxiliary-label information or preserve covariance.
    """
    carrier = _fixed_coordinate_carrier(orthogonal, occupied)
    raw_nuisance = _fixed_nuisance_carrier(orthogonal, occupied)
    nuisance, match_valid = _matched_nuisance_exclusion(
        click_gradient, orthogonal, occupied, carrier, raw_nuisance)
    nuisance_tangent = _tangent_orthogonal(
        click_gradient, nuisance, occupied, carrier)
    canonical = _canonical_tangent(nuisance_tangent)
    with torch.autocast(device_type=click_gradient.device.type, enabled=False):
        count = occupied.to(dtype=torch.float64).sum().clamp_min(1.0)

        def center(value):
            value = torch.where(occupied, value.to(dtype=torch.float64), 0.0)
            return torch.where(occupied, value - value.sum(dim=0, keepdim=True) / count, 0.0)

        def unit_and_norm(value):
            scale = value.abs().amax()
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            scaled = value / safe_scale
            length = scaled.square().sum().sqrt()
            safe_length = torch.where(length > 0, length, torch.ones_like(length))
            return scaled / safe_length, scale * length

        click_basis, _ = unit_and_norm(center(click_gradient))
        residual = center(nuisance)
        for _ in range(2):
            residual = residual - (residual * click_basis).sum() * click_basis
        _, residual_norm = unit_and_norm(residual)
        _, nuisance_norm = unit_and_norm(nuisance)
        rank_valid = (nuisance_norm > 0) & (residual_norm >= 1e-6 * nuisance_norm)
        flat = tangent.reshape(-1)
        pivot_index = flat.abs().argmax().reshape(1)
        pivot = torch.gather(flat, dim=0, index=pivot_index).squeeze(0)
        sign = torch.where(pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot))
        old_active = fixed_oriented.abs().amax() > 0
        return torch.where(old_active & match_valid & rank_valid & (pivot != 0), canonical * sign,
                           torch.zeros_like(canonical))


def _fixed_angle_nuisance_line(click_gradient, orthogonal, occupied,
                                tangent, nuisance_oriented):
    """Replace only the nuisance exclusion's auxiliary angle by pi/4.

    E is unit(F after K); U is the same unit nuisance after removing K/F.
    Native H=abs(<O,E>)*E+||O-<O,E>E||*U. Here H45=||O||*(E+U)/sqrt(2).
    Thus the exclusion radius, carrier, click constraint, Q orientation bit
    and inherited support remain; only H's scalar angle is replaced. This
    is not removal of all auxiliary information or isotropic noise.

    All geometry is FP64 before the inherited single cast/scatter boundary.
    Reuse the inherited tangent projection and 1e-6 validity convention.
    New degeneracies map to zero and require a host support-mismatch report.
    Normalizing the final direction again restores exactly native ||O|| up
    to FP64 roundoff. No RNG, parameters, persistent state or forward change.
    """
    carrier = _fixed_coordinate_carrier(orthogonal, occupied)
    nuisance = _fixed_nuisance_carrier(orthogonal, occupied)
    nuisance_tangent = _tangent_orthogonal(
        click_gradient, carrier, occupied, nuisance)
    with torch.autocast(device_type=click_gradient.device.type, enabled=False):
        count = occupied.to(dtype=torch.float64).sum().clamp_min(1.0)

        def center(value):
            value = torch.where(occupied, value.to(dtype=torch.float64), 0.0)
            return torch.where(occupied, value - value.sum(dim=0, keepdim=True) / count, 0.0)

        def unit_and_norm(value):
            scale = value.abs().amax()
            safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            scaled = value / safe_scale
            length = scaled.square().sum().sqrt()
            safe_length = torch.where(length > 0, length, torch.ones_like(length))
            return scaled / safe_length, scale * length

        click_basis, _ = unit_and_norm(center(click_gradient))
        residual = center(carrier)
        for _ in range(2):
            residual = residual - (residual * click_basis).sum() * click_basis
        reference_unit, reference_norm = unit_and_norm(residual)
        nuisance_unit, nuisance_norm = unit_and_norm(nuisance_tangent)
        _, radius = unit_and_norm(orthogonal)
        exclusion = (reference_unit + nuisance_unit) * (radius / (2.0 ** 0.5))
        direction = _tangent_orthogonal(
            click_gradient, exclusion, occupied, carrier)
        direction_unit, direction_norm = unit_and_norm(direction)
        canonical = _canonical_tangent(direction_unit * radius)
        flat = tangent.reshape(-1)
        pivot_index = flat.abs().argmax().reshape(1)
        pivot = torch.gather(flat, dim=0, index=pivot_index).squeeze(0)
        sign = torch.where(pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot))
        inherited_active = nuisance_oriented.abs().amax() > 0
        valid = (inherited_active & (radius > 0)
                 & (reference_norm >= 1e-6 * radius)
                 & (nuisance_norm > 0) & (direction_norm > 0) & (pivot != 0))
        return torch.where(valid, canonical * sign, torch.zeros_like(canonical))


def _centered_click_orientation_bit(click_gradient, occupied):
    """A single sign from occupied-row-centered click K, with zero -> +1.

    Match the inherited FP64 centering and first maximal-absolute flattened
    pivot rule. The bit uses no auxiliary cotangent at a fixed click/state.
    Zero K uses +1 without an extra gate, preserving the old output support.
    This coordinate sign is arbitrary and order dependent, not a semantic
    positive direction. Its balance/agreement with s(Q) require host checks.
    The replacement still retains auxiliary radius and inherited validity;
    later states also depend on earlier updates, so it is not label free.
    """
    with torch.autocast(device_type=click_gradient.device.type, enabled=False):
        active = torch.where(occupied, click_gradient.to(dtype=torch.float64), 0.0)
        count = occupied.to(dtype=torch.float64).sum().clamp_min(1.0)
        centered = torch.where(
            occupied, active - active.sum(dim=0, keepdim=True) / count, 0.0)
        flat = centered.reshape(-1)
        index = flat.abs().argmax().reshape(1)
        pivot = torch.gather(flat, dim=0, index=index).squeeze(0)
        return torch.where(pivot < 0, -torch.ones_like(pivot), torch.ones_like(pivot))


def _leave_one_out_orientation_bits(click_gradient, gradients, occupied):
    """Three Q pivot bits; omit one task ONLY in this helper's local inputs.

    Preserve the inherited ordered (cart+fav)+buy reduction by replacing
    one cotangent with a shape/dtype-matched zero, never by subtraction of
    an already-rounded aggregate. Recompute C/O, pair reassignment and Q
    using the exact inherited FP64 helpers. Select the first max-absolute
    flattened pivot sign; exact zero is +1 and adds no support gate.

    These reduced Q norms and validity gates determine only their bits.
    The actual transfer line/radius/zero support still come from all tasks.
    The removed task remains in those paths and private-task learning, so
    this is not a task-label ablation or a label-free replacement. At fixed
    state no direct parameter gradient is added by these diagnostics.
    All three constructions run in every variant, without state or RNG.
    """
    if len(gradients) != 3:
        raise ValueError('Expected ordered cart/fav/buy cotangents')
    bits = []
    for omitted in range(3):
        reduced = tuple(torch.zeros_like(gradient) if index == omitted else gradient
                        for index, gradient in enumerate(gradients))
        _, _, reduced_orthogonal = _click_relative_components(
            click_gradient, reduced, occupied)
        reduced_replacement = _reassign_orthogonal(
            click_gradient, reduced_orthogonal, occupied)
        reduced_tangent = _tangent_orthogonal(
            click_gradient, reduced_orthogonal, occupied, reduced_replacement)
        with torch.autocast(device_type=click_gradient.device.type, enabled=False):
            flat = reduced_tangent.reshape(-1)
            index = flat.abs().argmax().reshape(1)
            pivot = torch.gather(flat, dim=0, index=index).squeeze(0)
            bits.append(torch.where(pivot < 0, -torch.ones_like(pivot),
                                    torch.ones_like(pivot)))
    return tuple(bits)


def _fav_only_orientation_bit(click_gradient, gradients, occupied):
    """Retain only fav's cotangent when constructing the transported Q bit.

    At a fixed state, cart/buy are replaced by exact zeros only in this
    helper. Use the inherited ordered FP64 reduction, click projection,
    pair reassignment, tangent gate and first max-absolute pivot rule.
    Exact-zero Q uses +1 without changing the full-task support gate.
    Full-task radius, canonical line and support remain outside this
    helper. This is bit-source replacement, not removal of other labels.
    The helper runs in all four cells without state, RNG or extra forward.
    """
    if len(gradients) != 3:
        raise ValueError('Expected ordered cart/fav/buy cotangents')
    reduced = (torch.zeros_like(gradients[0]), gradients[1],
               torch.zeros_like(gradients[2]))
    _, _, orthogonal = _click_relative_components(
        click_gradient, reduced, occupied)
    replacement = _reassign_orthogonal(click_gradient, orthogonal, occupied)
    tangent = _tangent_orthogonal(
        click_gradient, orthogonal, occupied, replacement)
    with torch.autocast(device_type=click_gradient.device.type, enabled=False):
        flat = tangent.reshape(-1)
        index = flat.abs().argmax().reshape(1)
        pivot = torch.gather(flat, dim=0, index=index).squeeze(0)
        return torch.where(pivot < 0, -torch.ones_like(pivot),
                           torch.ones_like(pivot))


def _paired_source_orientation_bit(click_gradient, gradients, occupied):
    """Permute complete auxiliary triples only for the transported Q bit.

    Use the fixed half-batch involution, exchanging rows only when BOTH
    have histories. The same eligible permutation acts on cart/fav/buy,
    preserving their joint row multiset on occupied rows exactly. Click K
    and occupancy stay fixed. Recompute the inherited FP64 C/O, paired
    reassignment and tangent from these surrogate cotangents, and return
    its first-max-absolute pivot sign. Exact-zero tangent uses +1 and
    introduces no new gate in the actual full-task transfer.

    No labels, probabilities, private losses or real gradient arrays are
    reassigned or changed in place. Donor features/probabilities influence
    these cotangents; this is a correspondence replacement, not a clean
    label permutation or proof of independence. Empty rows cannot donate.
    Full-task radius, canonical line and validity retain auxiliary data.
    Host checks must measure eligible fraction, source-bit disagreement,
    balance, pivot degeneracy/ties and fixed-state norm/support parity.
    This helper runs in all cells without RNG or persistent model state.
    """
    if len(gradients) != 3:
        raise ValueError('Expected ordered cart/fav/buy cotangents')
    with torch.autocast(device_type=click_gradient.device.type, enabled=False):
        half = occupied.shape[0] // 2
        eligible = occupied & torch.roll(occupied, shifts=half, dims=0)
        donors = tuple(
            torch.where(eligible, torch.roll(g.to(dtype=torch.float64),
                                             shifts=half, dims=0),
                        g.to(dtype=torch.float64))
            for g in gradients)
    _, _, orthogonal = _click_relative_components(
        click_gradient, donors, occupied)
    replacement = _reassign_orthogonal(click_gradient, orthogonal, occupied)
    tangent = _tangent_orthogonal(
        click_gradient, orthogonal, occupied, replacement)
    with torch.autocast(device_type=click_gradient.device.type, enabled=False):
        flat = tangent.reshape(-1)
        index = flat.abs().argmax().reshape(1)
        pivot = torch.gather(flat, dim=0, index=index).squeeze(0)
        return torch.where(pivot < 0, -torch.ones_like(pivot),
                           torch.ones_like(pivot))


def _occupied_cycle_donor_rows(occupied):
    """Rotate the ascending occupied-row list by max(1, floor(n/4)).

    Empty rows map to themselves. Occupied rows form one ordered list;
    rotation preserves its multiset exactly, including when occupancy is
    noncontiguous. For n>=3 this shift is neither identity nor involution.
    n=2 necessarily exchanges two rows, n<=1 is identity. No data-dependent
    host scalar, RNG, model state, sample ID or label enters this mapping.
    The inherited half-batch reassignment inside Q remains unchanged.
    """
    if occupied.ndim != 2 or occupied.shape[1] != 1 or occupied.dtype != torch.bool:
        raise ValueError('Expected history occupancy bool[B,1]')
    rows = torch.arange(occupied.shape[0], device=occupied.device, dtype=torch.int64)
    active = occupied[:, 0]
    count = active.to(dtype=torch.int64).sum()
    # Unique sort keys put occupied indices first, in their original order.
    order = torch.argsort(torch.where(active, rows, rows + occupied.shape[0]))
    rank = active.to(dtype=torch.int64).cumsum(dim=0) - 1
    shift = torch.div(count, 4, rounding_mode='floor').clamp_min(1)
    donor_rank = (rank + shift).remainder(count.clamp_min(1))
    donors = torch.gather(order, dim=0, index=donor_rank)
    return torch.where(active, donors, rows)


def _occupied_cycle_source_orientation_bit(click_gradient, gradients, occupied):
    """Change only the auxiliary row source used to calculate the Q bit.

    Apply one common occupied-cycle permutation to all three cotangents.
    Keep click K, occupancy, the native Q's half-batch operator and all
    FP64 projection/centering/pivot conventions fixed. Exact-zero Q maps
    to +1 and adds no gate to the actual transfer. The native full-task
    line, radius and support are computed independently outside this helper.

    This deterministic donor control is not an independent-label sample:
    each cotangent retains donor features/probabilities, batch dependencies
    and private-task learning. A changed source need not change its one-bit
    summary. Host must measure fixed-state bit disagreement before making
    an inference from a null result. n<=2 is not a noninvolution contrast.
    All arms compute this helper; no parameters, state or RNG are added.
    """
    if len(gradients) != 3:
        raise ValueError('Expected ordered cart/fav/buy cotangents')
    donor_rows = _occupied_cycle_donor_rows(occupied)
    with torch.autocast(device_type=click_gradient.device.type, enabled=False):
        donors = tuple(g.to(dtype=torch.float64).index_select(0, donor_rows)
                       for g in gradients)
    _, _, orthogonal = _click_relative_components(
        click_gradient, donors, occupied)
    replacement = _reassign_orthogonal(click_gradient, orthogonal, occupied)
    tangent = _tangent_orthogonal(
        click_gradient, orthogonal, occupied, replacement)
    with torch.autocast(device_type=click_gradient.device.type, enabled=False):
        flat = tangent.reshape(-1)
        index = flat.abs().argmax().reshape(1)
        pivot = torch.gather(flat, dim=0, index=index).squeeze(0)
        return torch.where(pivot < 0, -torch.ones_like(pivot),
                           torch.ones_like(pivot))


def _zero_source_targets(probabilities):
    """Training-only zeros independent of all current auxiliary labels.

    Zero EVERY row, including empty histories. Training-time Dice can couple
    rows, so retaining true labels on empty rows would leave an explicit
    label path into occupied source cotangents. No extra forward, statistics,
    state or RNG. The real objective continues to use true targets everywhere.
    """
    return torch.zeros_like(probabilities.detach(), dtype=torch.float32)


def _cycle_source_targets(truth):
    """Move whole label tuples by a fixed cycle across ALL training rows.

    For the frozen B=8192 use shift=4097, coprime to B. Thus there is one
    cycle, no fixed rows and no pair involution. B=2 uses shift=1; other
    even batches use shift=1 as a defined check-only path. The shift depends
    only on batch size, never labels/features. All rows including empty
    histories participate: Dice can couple their source derivatives into
    occupied rows. Click's permuted column is never used in a source loss.
    Preserve exact per-task positive counts and joint auxiliary tuple counts.
    This deterministic donor control is not an independent-label sample;
    repeated users, ordering and prevalence remain possible explanations.
    """
    if truth.ndim != 2 or truth.shape[0] < 2 or truth.shape[0] % 2:
        raise ValueError('Expected an even training label matrix [B,4]')
    shift = 4097 if truth.shape[0] == 8192 else 1
    return torch.roll(truth.detach(), shifts=shift, dims=0)


def _mean_source_targets(truth):
    """Use each task's batch prevalence as a soft label on EVERY row.

    Training only. Reduce detached FP32 labels over all rows, including
    empty histories because Dice couples examples. For the frozen binary
    B=8192 batch, positive counts divided by 8192 are exactly representable
    in FP32. This removes example assignment and joint label tuples from
    this source while retaining per-task prevalence and recipient graphs.
    It does not remove true labels from private updates or native geometry.
    No persistent statistics, sample IDs, RNG or additional forward pass.
    """
    if truth.ndim != 2 or truth.shape[0] < 2:
        raise ValueError('Expected training label matrix [B,4] with B>=2')
    labels = truth.detach().to(dtype=torch.float32)
    return labels.mean(dim=0, keepdim=True).expand_as(labels)


def _label_source_orientation_bit(click_gradient, gradients, occupied):
    """Q's sign from counterfactual labels at fixed recipient computation.

Unlike moving a precomputed cotangent, the caller differentiates a source
objective through the original recipient probability graph. Its source is
a declared full-source or positive-label-only BCE difference. Only this bit uses that VJP;
native true-label geometry is kept separate.
Exact-zero Q maps to +1 without an additional validity gate. The helper
runs in native controls too. Bit disagreement must be measured on the host.
    """
    _, _, orthogonal = _click_relative_components(
        click_gradient, gradients, occupied)
    replacement = _reassign_orthogonal(click_gradient, orthogonal, occupied)
    tangent = _tangent_orthogonal(
        click_gradient, orthogonal, occupied, replacement)
    with torch.autocast(device_type=click_gradient.device.type, enabled=False):
        flat = tangent.reshape(-1)
        index = flat.abs().argmax().reshape(1)
        pivot = torch.gather(flat, dim=0, index=index).squeeze(0)
        return torch.where(pivot < 0, -torch.ones_like(pivot),
                           torch.ones_like(pivot))


def _reduce_context_transfer(click_gradient, gradients, occupied,
                             parallel_orientation, orthogonal_orientation,
                             click_relative_transfer, orthogonal_assignment,
                             label_source_gradients=None):
    """Cross native versus label-replaced Q bit with external sign.

    All old and new helpers execute in every cell. Counterfactual VJPs enter
    only the new bit. Native bridges retain the inherited transfer expression.
    The optional default supports direct fixed-state helper comparisons.
    """
    if parallel_orientation != 'aligned' or click_relative_transfer != 'orthogonal_native':
        raise ValueError('Assignment study freezes native orthogonal-only transfer')
    if orthogonal_orientation not in ('aligned', 'reversed'):
        raise ValueError('Unknown orthogonal orientation')
    if orthogonal_assignment not in ('native', 'paired_reprojected', 'paired_tangent', 'canonical_tangent', 'fixed_line_oriented', 'fixed_line_canonical', 'click_line_oriented', 'nuisance_line_oriented', 'nuisance_line_fixed_angle', 'nuisance_line_fixed_angle_canonical', 'nuisance_line_fixed_angle_click_bit', 'nuisance_line_fixed_angle_without_cart_bit', 'nuisance_line_fixed_angle_without_fav_bit', 'nuisance_line_fixed_angle_without_buy_bit', 'nuisance_line_fixed_angle_fav_only_bit', 'nuisance_line_fixed_angle_paired_source_bit', 'nuisance_line_fixed_angle_occupied_cycle_source_bit', 'nuisance_line_fixed_angle_label_mean_bit', 'nuisance_line_fixed_angle_decomposed_label_bit'):
        raise ValueError('Unknown orthogonal assignment')
    centered, parallel, orthogonal = _click_relative_components(
        click_gradient, gradients, occupied)
    _matched_parallel, _matched_orthogonal = _norm_matched_projections(
        centered, parallel, orthogonal)
    replacement = _reassign_orthogonal(click_gradient, orthogonal, occupied)
    tangent = _tangent_orthogonal(click_gradient, orthogonal, occupied, replacement)
    canonical = _canonical_tangent(tangent)
    fixed_oriented = _fixed_line_with_orientation(
        click_gradient, orthogonal, occupied, tangent)
    fixed_canonical = _canonical_tangent(fixed_oriented)
    click_oriented = _click_only_fixed_line(
        click_gradient, orthogonal, occupied, tangent, fixed_oriented)
    nuisance_oriented = _nuisance_line_with_orientation(
        click_gradient, orthogonal, occupied, tangent, fixed_oriented)
    fixed_angle_oriented = _fixed_angle_nuisance_line(
        click_gradient, orthogonal, occupied, tangent, nuisance_oriented)
    # Same line, radius and zero gate; remove only transported s(Q).
    fixed_angle_canonical = _canonical_tangent(fixed_angle_oriented)
    click_bit = _centered_click_orientation_bit(click_gradient, occupied)
    fixed_angle_click_oriented = fixed_angle_canonical * click_bit
    reduced_bits = _leave_one_out_orientation_bits(
        click_gradient, gradients, occupied)
    fav_only_bit = _fav_only_orientation_bit(click_gradient, gradients, occupied)
    paired_source_bit = _paired_source_orientation_bit(
        click_gradient, gradients, occupied)
    cycle_source_bit = _occupied_cycle_source_orientation_bit(
        click_gradient, gradients, occupied)
    label_source_bit = _label_source_orientation_bit(
        click_gradient, (gradients if label_source_gradients is None
                         else label_source_gradients), occupied)
    with torch.autocast(device_type=click_gradient.device.type, enabled=False):
        label_source_oriented = fixed_angle_canonical * label_source_bit
        label_source_negative = -label_source_oriented
        cycle_source_oriented = fixed_angle_canonical * cycle_source_bit
        cycle_source_negative = -cycle_source_oriented
        paired_source_oriented = fixed_angle_canonical * paired_source_bit
        paired_source_negative = -paired_source_oriented
        fav_only_oriented = fixed_angle_canonical * fav_only_bit
        fav_only_negative = -fav_only_oriented
        task_bit_outputs = {
            'nuisance_line_fixed_angle_without_' + label + '_bit':
                fixed_angle_canonical * bit
            for label, bit in zip(('cart', 'fav', 'buy'), reduced_bits)}
        task_bit_negative = {name: -value for name, value in task_bit_outputs.items()}
        native_negative = -orthogonal
        replacement_negative = -replacement
        tangent_negative = -tangent
        canonical_negative = -canonical
        fixed_negative = -fixed_oriented
        fixed_canonical_negative = -fixed_canonical
        click_negative = -click_oriented
        nuisance_negative = -nuisance_oriented
        fixed_angle_negative = -fixed_angle_oriented
        fixed_angle_canonical_negative = -fixed_angle_canonical
        fixed_angle_click_negative = -fixed_angle_click_oriented
        if orthogonal_assignment == 'native':
            transfer = (orthogonal if orthogonal_orientation == 'aligned'
                        else native_negative)
        elif orthogonal_assignment == 'paired_reprojected':
            transfer = (replacement if orthogonal_orientation == 'aligned'
                        else replacement_negative)
        elif orthogonal_assignment == 'paired_tangent':
            transfer = tangent if orthogonal_orientation == 'aligned' else tangent_negative
        elif orthogonal_assignment == 'canonical_tangent':
            transfer = canonical if orthogonal_orientation == 'aligned' else canonical_negative
        elif orthogonal_assignment == 'fixed_line_oriented':
            transfer = fixed_oriented if orthogonal_orientation == 'aligned' else fixed_negative
        elif orthogonal_assignment == 'fixed_line_canonical':
            transfer = (fixed_canonical if orthogonal_orientation == 'aligned'
                        else fixed_canonical_negative)
        elif orthogonal_assignment == 'click_line_oriented':
            transfer = click_oriented if orthogonal_orientation == 'aligned' else click_negative
        elif orthogonal_assignment == 'nuisance_line_oriented':
            transfer = (nuisance_oriented if orthogonal_orientation == 'aligned'
                        else nuisance_negative)
        elif orthogonal_assignment == 'nuisance_line_fixed_angle':
            transfer = (fixed_angle_oriented if orthogonal_orientation == 'aligned'
                        else fixed_angle_negative)
        elif orthogonal_assignment == 'nuisance_line_fixed_angle_canonical':
            transfer = (fixed_angle_canonical if orthogonal_orientation == 'aligned'
                        else fixed_angle_canonical_negative)
        elif orthogonal_assignment == 'nuisance_line_fixed_angle_click_bit':
            transfer = (fixed_angle_click_oriented if orthogonal_orientation == 'aligned'
                        else fixed_angle_click_negative)
        elif orthogonal_assignment in ('nuisance_line_fixed_angle_label_mean_bit',
                                       'nuisance_line_fixed_angle_decomposed_label_bit'):
            transfer = (label_source_oriented if orthogonal_orientation == 'aligned'
                        else label_source_negative)
        elif orthogonal_assignment == 'nuisance_line_fixed_angle_occupied_cycle_source_bit':
            transfer = (cycle_source_oriented if orthogonal_orientation == 'aligned'
                        else cycle_source_negative)
        elif orthogonal_assignment == 'nuisance_line_fixed_angle_paired_source_bit':
            transfer = (paired_source_oriented if orthogonal_orientation == 'aligned'
                        else paired_source_negative)
        elif orthogonal_assignment == 'nuisance_line_fixed_angle_fav_only_bit':
            transfer = (fav_only_oriented if orthogonal_orientation == 'aligned'
                        else fav_only_negative)
        else:
            transfer = (task_bit_outputs[orthogonal_assignment]
                        if orthogonal_orientation == 'aligned'
                        else task_bit_negative[orthogonal_assignment])
        return click_gradient + transfer.to(dtype=click_gradient.dtype)


def source_bit_diagnostics(click_gradient, true_gradients, source_gradients, occupied):
    """Optional host fixed-state probe; no logging, state or graph mutation.

    Call with detached cotangents from the same real training batch/state.
    This is not automatically collected by the host's documented API.
    A null AUC contrast without bit disagreement is an ineffective bit
    intervention, not evidence that correct labels are unnecessary.
    """
    with torch.no_grad():
        native_bit = _label_source_orientation_bit(
            click_gradient, true_gradients, occupied)
        source_bit = _label_source_orientation_bit(
            click_gradient, source_gradients, occupied)
        _, _, native_orthogonal = _click_relative_components(
            click_gradient, true_gradients, occupied)
        _, _, source_orthogonal = _click_relative_components(
            click_gradient, source_gradients, occupied)
        replacement = _reassign_orthogonal(click_gradient, native_orthogonal, occupied)
        tangent = _tangent_orthogonal(
            click_gradient, native_orthogonal, occupied, replacement)
        fixed = _fixed_line_with_orientation(
            click_gradient, native_orthogonal, occupied, tangent)
        nuisance = _nuisance_line_with_orientation(
            click_gradient, native_orthogonal, occupied, tangent, fixed)
        native_transfer = _fixed_angle_nuisance_line(
            click_gradient, native_orthogonal, occupied, tangent, nuisance)
        native_total = _reduce_context_transfer(
            click_gradient, true_gradients, occupied, 'aligned', 'reversed',
            'orthogonal_native', 'nuisance_line_fixed_angle', source_gradients)
        replacement_total = _reduce_context_transfer(
            click_gradient, true_gradients, occupied, 'aligned', 'reversed',
            'orthogonal_native', 'nuisance_line_fixed_angle_label_mean_bit',
            source_gradients)
        return {
            'native_bit': native_bit,
            'source_bit': source_bit,
            'bit_disagreement': native_bit != source_bit,
            'native_transfer_active': native_transfer.abs().amax() > 0,
            'native_transfer_radius': native_transfer.square().sum().sqrt(),
            'occupied_rows': occupied.to(dtype=torch.int64).sum(),
            'native_orthogonal_radius': native_orthogonal.square().sum().sqrt(),
            'source_orthogonal_radius_for_bit_only': source_orthogonal.square().sum().sqrt(),
            'actual_context_vjp_l2_difference': (
                native_total.to(dtype=torch.float64)
                - replacement_total.to(dtype=torch.float64)).square().sum().sqrt(),
        }


class _CollectedContextTransfer(torch.autograd.Function):
    """Seven identity edges: click, true auxiliary, alternate-label auxiliary.

    Only the first output supplies click's context uses; the other outputs
    supply cart/fav/buy product contexts. Other auxiliary context edges stay
    blocked. Private parameters receive the ordinary true-label BCE graph.
    The last three edges receive a zero-value linear surrogate in the loss
    hook, transporting detached counterfactual cotangents without training
    any private module on counterfactual labels. No extra forward is run.
    Unused cotangents are materialized as zeros by the autograd default.
    """
    @staticmethod
    def forward(ctx, context, occupied, parallel_orientation, orthogonal_orientation,
                click_relative_transfer, orthogonal_assignment):
        if context.ndim != 2 or context.shape[0] < 2 or context.shape[0] % 2:
            raise ValueError('Frozen training requires even context batches [B,D]')
        if occupied.shape != (context.shape[0], 1) or occupied.dtype != torch.bool:
            raise ValueError('Expected history occupancy bool[B,1]')
        if parallel_orientation not in ('aligned', 'reversed'):
            raise ValueError('Unknown parallel orientation')
        if orthogonal_orientation not in ('aligned', 'reversed'):
            raise ValueError('Unknown orthogonal orientation')
        if parallel_orientation != 'aligned':
            raise ValueError('Direction factorial fixes the parallel component sign')
        if click_relative_transfer not in (
                'centered', 'parallel_native', 'orthogonal_native', 'blocked'):
            raise ValueError('Unknown click-relative transfer selection')
        if click_relative_transfer != 'orthogonal_native':
            raise ValueError('Assignment study requires orthogonal-only transfer')
        if orthogonal_assignment not in ('native', 'paired_reprojected', 'paired_tangent', 'canonical_tangent', 'fixed_line_oriented', 'fixed_line_canonical', 'click_line_oriented', 'nuisance_line_oriented', 'nuisance_line_fixed_angle', 'nuisance_line_fixed_angle_canonical', 'nuisance_line_fixed_angle_click_bit', 'nuisance_line_fixed_angle_without_cart_bit', 'nuisance_line_fixed_angle_without_fav_bit', 'nuisance_line_fixed_angle_without_buy_bit', 'nuisance_line_fixed_angle_fav_only_bit', 'nuisance_line_fixed_angle_paired_source_bit', 'nuisance_line_fixed_angle_occupied_cycle_source_bit', 'nuisance_line_fixed_angle_label_mean_bit', 'nuisance_line_fixed_angle_decomposed_label_bit'):
            raise ValueError('Unknown orthogonal assignment')
        ctx.orthogonal_assignment = orthogonal_assignment
        ctx.click_relative_transfer = click_relative_transfer
        ctx.parallel_orientation = parallel_orientation
        ctx.orthogonal_orientation = orthogonal_orientation
        ctx.save_for_backward(occupied)
        return tuple(context.view_as(context) for _ in range(7))

    @staticmethod
    def backward(ctx, click_gradient, cart_gradient, fav_gradient, buy_gradient,
                 source_cart_gradient, source_fav_gradient, source_buy_gradient):
        (occupied,) = ctx.saved_tensors
        total = _reduce_context_transfer(
            click_gradient, (cart_gradient, fav_gradient, buy_gradient), occupied,
            ctx.parallel_orientation, ctx.orthogonal_orientation,
            ctx.click_relative_transfer, ctx.orthogonal_assignment,
            (source_cart_gradient, source_fav_gradient, source_buy_gradient))
        return total, None, None, None, None, None


class Model(MultiTaskModel):
    def __init__(self, feature_map, model_id='TargetMatchOrderMoment', gpu=-1,
                 embedding_dim=16, attention_hidden_units=(64, 32),
                 tower_hidden_units=(128, 64), attention_normalization='none',
                 compression_routes='uf', block_aux_scorer_grad=True,
                 block_aux_context_grad=False, block_aux_item_grad=True, num_tasks=None,
                 residual_input='match_detail', residual_routing='task', residual_hidden=16,
                 temporal_detail='position', block_detail='ordinal',
                 curvature_detail='quadratic', joint_detail='duplicate_match',
                 match_support='available', match_scaling='norm', coarse_extension='duplicate',
                 primary_partition='uniform', normalization_support='rotated_duplicate',
                 support_gradient='attached', inference_support='fully_pruned',
                 training_support='folded', residual_execution='direct',
                 auxiliary_train_population='primary', auxiliary_eval_population='primary',
                 block_aux_tower_context_grad=True,
                 block_aux_adapter_context_grad=False,
                 context_fusion='product', interaction_rank=16,
                 block_aux_residual_context_grad=True,
                 block_aux_product_context_grad=False,
                 product_context_assignment='aligned',
                 product_context_component='centered',
                 product_context_magnitude='native',
                 product_context_orientation='aligned',
                 parallel_orientation='aligned',
                 orthogonal_orientation='reversed',
                 centering_scope='aggregate',
                 click_context_collection='joint',
                 auxiliary_reduction_precision='float64',
                 click_relative_transfer='orthogonal_native',
                 orthogonal_assignment='nuisance_line_fixed_angle_decomposed_label_bit',
                 source_component='positive_label',
                 source_label_assignment='batch_mean',
                 task='binary_classification', **kwargs):
        if num_tasks is not None and num_tasks != len(feature_map.labels):
            raise ValueError('Task count mismatch')
        if set(feature_map.labels) != {'is_click', 'cart', 'fav', 'buy'}:
            raise ValueError('This study requires the frozen four-task schema')
        if attention_normalization != 'none':
            raise ValueError('Unnormalized DIN activation is required')
        if compression_routes != 'uf':
            raise ValueError('This source-path study freezes click/auxiliary routes to uf')
        super().__init__(feature_map, model_id=model_id, gpu=gpu,
                         num_tasks=len(feature_map.labels), task=task, **kwargs)
        self.compression_routes = compression_routes
        if block_aux_scorer_grad is not True:
            raise ValueError('Auxiliary scorer parameter gradients must stay blocked')
        if not isinstance(block_aux_context_grad, bool) or not isinstance(block_aux_item_grad, bool):
            raise ValueError('Source-path switches must be booleans')
        self.block_aux_scorer_grad = block_aux_scorer_grad
        self.block_aux_context_grad = block_aux_context_grad
        self.block_aux_item_grad = block_aux_item_grad
        if block_aux_context_grad is not False or block_aux_item_grad is not True:
            raise ValueError('Use the separate context-path gates; item/action gradients stay blocked')
        if (not isinstance(block_aux_tower_context_grad, bool)
                or not isinstance(block_aux_adapter_context_grad, bool)):
            raise ValueError('Context-path gates must be booleans')
        self.block_aux_tower_context_grad = block_aux_tower_context_grad
        self.block_aux_adapter_context_grad = block_aux_adapter_context_grad
        if (not isinstance(block_aux_residual_context_grad, bool)
                or not isinstance(block_aux_product_context_grad, bool)):
            raise ValueError('Adapter-specific context gates must be booleans')
        # Plain metadata only: no modules, buffers, RNG or parameter changes.
        self.block_aux_residual_context_grad = block_aux_residual_context_grad
        self.block_aux_product_context_grad = block_aux_product_context_grad
        if product_context_assignment not in ('aligned', 'half_swap'):
            raise ValueError('Unknown collected auxiliary assignment')
        if product_context_orientation not in ('aligned', 'reversed'):
            raise ValueError('Unknown centered-component orientation')
        if product_context_assignment != 'aligned':
            raise ValueError('Direction factorial freezes example assignment')
        if parallel_orientation not in ('aligned', 'reversed'):
            raise ValueError('Unknown parallel orientation')
        if orthogonal_orientation not in ('aligned', 'reversed'):
            raise ValueError('Unknown orthogonal orientation')
        if product_context_orientation != 'aligned':
            raise ValueError('Use the two click-relative orientation switches')
        if centering_scope != 'aggregate':
            raise ValueError('Component factorial freezes aggregate reduction')
        if click_context_collection != 'joint':
            raise ValueError('Click-relative study requires joint cotangent collection')
        if auxiliary_reduction_precision != 'float64':
            raise ValueError('Component factorial freezes FP64 reduction')
        self.auxiliary_reduction_precision = auxiliary_reduction_precision
        self.centering_scope = centering_scope
        self.click_context_collection = click_context_collection
        if parallel_orientation != 'aligned':
            raise ValueError('Direction factorial fixes the parallel component sign')
        if click_relative_transfer not in (
                'centered', 'parallel_native', 'orthogonal_native', 'blocked'):
            raise ValueError('Unknown click-relative transfer selection')
        if click_relative_transfer != 'orthogonal_native':
            raise ValueError('Assignment study requires orthogonal-only transfer')
        if orthogonal_assignment not in ('native', 'paired_reprojected', 'paired_tangent', 'canonical_tangent', 'fixed_line_oriented', 'fixed_line_canonical', 'click_line_oriented', 'nuisance_line_oriented', 'nuisance_line_fixed_angle', 'nuisance_line_fixed_angle_canonical', 'nuisance_line_fixed_angle_click_bit', 'nuisance_line_fixed_angle_without_cart_bit', 'nuisance_line_fixed_angle_without_fav_bit', 'nuisance_line_fixed_angle_without_buy_bit', 'nuisance_line_fixed_angle_fav_only_bit', 'nuisance_line_fixed_angle_paired_source_bit', 'nuisance_line_fixed_angle_occupied_cycle_source_bit', 'nuisance_line_fixed_angle_label_mean_bit', 'nuisance_line_fixed_angle_decomposed_label_bit'):
            raise ValueError('Unknown orthogonal assignment')
        if source_component not in ('positive_label', 'full'):
            raise ValueError('Expected positive_label or full auxiliary bit source')
        if source_component != 'positive_label':
            raise ValueError('Assignment study freezes the positive-label-only source')
        if source_label_assignment not in ('native', 'batch_cycle', 'batch_mean'):
            raise ValueError('Expected native, batch_cycle or batch_mean source label assignment')
        self.source_label_assignment = source_label_assignment
        self.source_component = source_component
        self.orthogonal_assignment = orthogonal_assignment
        self.click_relative_transfer = click_relative_transfer
        self.parallel_orientation = parallel_orientation
        self.orthogonal_orientation = orthogonal_orientation
        self.product_context_orientation = product_context_orientation
        if product_context_component != 'centered':
            raise ValueError('Click-relative study freezes centered transfer')
        if block_aux_residual_context_grad is not True:
            raise ValueError('Component study isolates product-only context transfer')
        self.product_context_component = product_context_component
        if product_context_magnitude != 'native':
            raise ValueError('Direction study freezes native component magnitude')
        self.product_context_magnitude = product_context_magnitude
        if (block_aux_tower_context_grad is not True
                or block_aux_adapter_context_grad is not False
                or block_aux_product_context_grad is not False):
            raise ValueError('Freeze other context gates; auxiliary product edge must be open')
        if context_fusion != 'product':
            raise ValueError('Direction attribution freezes product fusion')
        self.product_context_assignment = product_context_assignment
        if context_fusion not in ('product', 'sum'):
            raise ValueError('Unknown context/history fusion operator')
        if (not isinstance(interaction_rank, int) or isinstance(interaction_rank, bool)
                or interaction_rank < 1):
            raise ValueError('Interaction rank must be a positive integer')
        self.context_fusion = context_fusion
        self.interaction_rank = interaction_rank
        if residual_input != 'match_detail':
            raise ValueError('The original match-detail channel must be retained')
        if temporal_detail != 'position':
            raise ValueError('The incumbent within-block ordinal moment is frozen')
        if block_detail != 'ordinal':
            raise ValueError('The incumbent coarse-linear summary is frozen')
        if curvature_detail != 'quadratic':
            raise ValueError('The incumbent quadratic fourth summary is frozen')
        if joint_detail != 'duplicate_match':
            raise ValueError('This factorial retains the fifth match direction')
        if match_support not in ('available', 'joint'):
            raise ValueError('Unknown fifth match-support control')
        if match_scaling not in ('norm', 'raw'):
            raise ValueError('Unknown fifth match-scaling control')
        if residual_routing != 'task':
            raise ValueError('This replacement study freezes task-specific routing')
        if residual_routing not in ('task', 'shared'):
            raise ValueError('Unknown residual parameter-sharing control')
        if not isinstance(residual_hidden, int) or residual_hidden < 1:
            raise ValueError('Residual width must be a positive integer')
        self.residual_input = residual_input
        self.residual_routing = residual_routing
        self.temporal_detail = temporal_detail
        self.block_detail = block_detail
        self.curvature_detail = curvature_detail
        self.joint_detail = joint_detail
        self.match_support = match_support
        self.match_scaling = match_scaling
        if match_support != 'available' or match_scaling != 'norm':
            raise ValueError('This study freezes the incumbent fifth support and scale')
        if coarse_extension not in ('cubic', 'scrambled', 'duplicate', 'bridge'):
            raise ValueError('Unknown coarse ordinal replacement')
        self.coarse_extension = coarse_extension
        if primary_partition not in ('aligned', 'rotated', 'uniform'):
            raise ValueError('Unknown primary partition readout')
        if primary_partition != 'uniform':
            raise ValueError('This study fixes the primary uniform readout')
        self.primary_partition = primary_partition
        if normalization_support not in ('aligned_rotated', 'rotated_duplicate'):
            raise ValueError('Unknown Dice support population')
        if support_gradient not in ('attached', 'blocked'):
            raise ValueError('Unknown support preactivation gradient')
        self.normalization_support = normalization_support
        self.support_gradient = support_gradient
        if normalization_support != 'rotated_duplicate' or support_gradient != 'attached':
            raise ValueError('This study retains native attached rotated support during training')
        if inference_support not in ('full', 'score_pruned', 'fully_pruned'):
            raise ValueError('Unknown inference support execution mode')
        self.inference_support = inference_support
        if training_support not in ('folded', 'compression_only', 'full', 'unweighted'):
            raise ValueError('Unknown support computation control')
        if normalization_support != 'rotated_duplicate' or support_gradient != 'attached':
            raise ValueError('Folding requires identical attached support pairs')
        if inference_support != 'fully_pruned':
            raise ValueError('This suite shares the validated pruned eval path')
        self.training_support = training_support
        if residual_execution not in ('direct', 'all'):
            raise ValueError('Unknown task-adapter execution mode')
        self.residual_execution = residual_execution
        if auxiliary_train_population not in ('feedback', 'primary'):
            raise ValueError('Unknown auxiliary training normalization population')
        if auxiliary_eval_population not in ('primary', 'feedback'):
            raise ValueError('Unknown auxiliary evaluation normalization population')
        self.auxiliary_train_population = auxiliary_train_population
        self.auxiliary_eval_population = auxiliary_eval_population
        self.embedding_layer = FeatureEmbeddingDict(feature_map, embedding_dim)
        self.item_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') == 'item' and spec['type'] != 'meta']
        self.user_fields = [name for name, spec in feature_map.features.items()
                            if spec.get('source') not in ('item', 'action')
                            and spec['type'] != 'meta']
        self.action_field = next((name for name, spec in feature_map.features.items()
                                  if spec.get('source') == 'action'), None)
        if self.action_field is None:
            raise ValueError('Historical action field is required')
        self.auxiliary_indices = [list(feature_map.labels).index(label)
                                  for label in ('cart', 'fav', 'buy')]
        self.key_width = len(self.item_fields) * embedding_dim
        self.value_width = self.key_width + embedding_dim
        # Only supplied encoded fields are compared; no external lookups.
        # Zero and declared special IDs never constitute positive evidence.
        self.match_fields = ('item_id', 'cate_id', 'brand')
        self.match_excluded_ids = {}
        for name in self.match_fields:
            if name not in self.item_fields:
                raise ValueError('Required target-match field is absent: ' + name)
            spec = feature_map.features[name]
            excluded = {0}
            for special in ('padding_idx', 'oov_idx'):
                if spec.get(special) is not None:
                    excluded.add(int(spec[special]))
            self.match_excluded_ids[name] = tuple(sorted(excluded))

        # Inherit the action-aware scorer and towers from the previous factorial.
        # All cells instantiate exactly the same modules in exactly this order.
        layers = []
        width = 4 * self.key_width + embedding_dim
        for hidden in attention_hidden_units:
            layers.extend([nn.Linear(width, hidden), Dice(hidden)])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.local_activation = nn.Sequential(*layers)
        input_width = len(self.user_fields) * embedding_dim + self.key_width + self.value_width
        self.towers = nn.ModuleList()
        for _ in feature_map.labels:
            tower = []
            width = input_width
            for hidden in tower_hidden_units:
                tower.extend([nn.Linear(width, hidden), Dice(hidden)])
                width = hidden
            tower.append(nn.Linear(width, 1))
            self.towers.append(nn.Sequential(*tower))
        self.reset_parameters()
        # No change to inherited modules, their initialization or host RNG.
        # Zero final layers give source-equivalent predictions initially.
        with torch.random.fork_rng(devices=[]):
            self.residual_bank = nn.ModuleList([
                nn.Sequential(nn.Linear(input_width + len(self.match_fields) + 5 * self.value_width,
                                        residual_hidden), nn.ReLU(),
                              nn.Linear(residual_hidden, self.value_width))
                for _ in feature_map.labels
            ])
            for adapter in self.residual_bank:
                nn.init.zeros_(adapter[-1].weight)
                nn.init.zeros_(adapter[-1].bias)
            # Appended after every inherited learned module is initialized.
            # This same fork preserves the host RNG after construction.
            context_width = len(self.user_fields) * embedding_dim
            history_width = 6 * self.value_width + len(self.match_fields)
            self.context_queries = nn.ModuleList([
                nn.Linear(context_width, interaction_rank, bias=False)
                for _ in feature_map.labels
            ])
            self.history_projections = nn.ModuleList([
                nn.Linear(history_width, interaction_rank, bias=False)
                for _ in feature_map.labels
            ])
            self.interaction_outputs = nn.ModuleList([
                nn.Linear(interaction_rank, self.value_width, bias=False)
                for _ in feature_map.labels
            ])
            for output in self.interaction_outputs:
                nn.init.zeros_(output.weight)
        # Add only nonlearned state, AFTER every inherited learned module and
        # initialization. Native Dice has running mean/variance and counters.
        # Registered clones move with the model and round-trip in checkpoints;
        # they never alias the primary running buffers. No RNG is consumed.
        self._auxiliary_state_names = {}
        for index, (name, buffer) in enumerate(self.local_activation.named_buffers()):
            storage_name = '_auxiliary_moment_' + str(index)
            self.register_buffer(storage_name, buffer.detach().clone())
            self._auxiliary_state_names[name] = storage_name
        self.model_to_device()

    def _represent(self, inputs):
        # Do not allow the framework fallback that treats every event as
        # positive for every task when historical masks are missing.
        if len(inputs) != 4:
            raise ValueError('Released historical task masks are required')
        supplied = inputs[3]
        if isinstance(supplied, (list, tuple)):
            if len(supplied) != self.num_tasks:
                raise ValueError('Wrong number of historical task masks')
        elif torch.is_tensor(supplied):
            if supplied.ndim != 3 or supplied.shape[0] != self.num_tasks:
                raise ValueError('Expected historical masks with shape [4,B,100]')
        else:
            raise ValueError('Released historical task masks are required')
        user, item, mask, task_masks = self.get_inputs(inputs, return_multi_masks=True)
        if mask.ndim != 2 or mask.shape[1] != 100:
            raise ValueError('Frozen protocol requires 100 history positions')
        auxiliary = torch.zeros_like(mask, dtype=torch.bool)
        for index in self.auxiliary_indices:
            if task_masks[index].shape != mask.shape:
                raise ValueError('Historical task mask shape mismatch')
            auxiliary = auxiliary | (task_masks[index] > 0)
        auxiliary = auxiliary & mask.bool()
        user_emb = self.embedding_layer(user)
        item_emb = self.embedding_layer(item)
        target = torch.cat([item_emb[name][:, -1] for name in self.item_fields], dim=-1)
        keys = torch.cat([item_emb[name][:, :-1] for name in self.item_fields], dim=-1)
        # Current action is unknown and never enters either summary or scorer.
        values = torch.cat([keys, item_emb[self.action_field][:, :-1]], dim=-1)
        context = torch.cat([user_emb[name] for name in self.user_fields], dim=-1)
        detail, position_detail, block_detail, curvature_detail, joint_detail, match_counts, match_union = self._target_match_summary(
            item, values, mask)
        return context, target, values, mask, auxiliary, detail, position_detail, block_detail, curvature_detail, joint_detail, match_counts, match_union

    def _target_match_summary(self, item, values, mask):
        """Five masked value summaries; only the fifth direction/gate varies.

        Quadratic/scrambled coefficients are orthogonal to valid constants
        and coarse linear ranks under event-count weighting. A common gate
        excludes fewer than three occupied blocks and degenerate residuals.
        All fourth coefficients match the inherited coarse coefficient norm.
        Duplicate coefficients copy that coarse vector and add no new moment.
        FP32 coefficient construction precedes the inherited bf16 contraction.
        """
        matches = []
        for name in self.match_fields:
            ids = item[name]
            if ids.ndim != 2 or ids.shape[1] != 101 or ids.shape[0] != mask.shape[0]:
                raise ValueError('Expected target-match IDs with shape [B,101]')
            history_ids, target_id = ids[:, :-1], ids[:, -1:]
            valid = mask.bool() & (history_ids > 0) & (target_id > 0)
            for special in self.match_excluded_ids[name]:
                valid = valid & (history_ids != special) & (target_id != special)
            matches.append(valid & (history_ids == target_id))
        match_matrix = torch.stack(matches, dim=-1).to(torch.float32)
        match_counts = torch.log1p(match_matrix.sum(dim=1))
        batch, length, width = values.shape
        if length != 100:
            raise ValueError('Frozen target-match compression requires 100 history positions')
        valid = mask.bool().reshape(batch, 10, 10).to(torch.float32)
        counts = valid.sum(dim=-1, keepdim=True)
        affinity = match_matrix.mean(dim=-1).reshape(batch, 10, 10)
        centered = affinity - valid * (affinity.sum(dim=-1, keepdim=True)
                                      / counts.clamp_min(1))
        # Retain the incumbent within-block ordinal moment without changes.
        position = torch.arange(10, device=values.device, dtype=torch.float32).view(1, 1, 10) / 9.0
        position_sum = (position * valid).sum(dim=-1, keepdim=True)
        centered_position = valid * (position - position_sum / counts.clamp_min(1))
        # Coarse ordinals add only information missing under whole-block
        # permutations. Center using event counts, including partial blocks.
        total = counts.sum(dim=1).clamp_min(1)
        block_rank = torch.arange(10, device=values.device, dtype=torch.float32).view(1, 10, 1) / 9.0
        block_mean = (block_rank * valid).sum(dim=(1, 2), keepdim=True) / total.unsqueeze(-1)
        centered_block = valid * (block_rank - block_mean)
        local_norm2 = centered_position.square().sum(dim=(1, 2), keepdim=True)
        block_norm2 = centered_block.square().sum(dim=(1, 2), keepdim=True)
        centered_block = (centered_block * (local_norm2 / block_norm2.clamp_min(1e-12)).sqrt()
                          * (block_norm2 > 1e-12).to(valid.dtype))
        # Retain the inherited coarse coefficient construction, including its
        # unused local duplicate, identically in all fourth-channel cells.
        duplicated_position = centered_position * (block_norm2 > 1e-12).to(valid.dtype)
        block_coefficients = (centered_block if self.block_detail == 'ordinal'
                              else duplicated_position)
        # Work with the unscaled centered block rank. Its constant and
        # linear span is independent of the incumbent norm matching above.
        linear = valid * (block_rank - block_mean)
        linear_norm2 = linear.square().sum(dim=(1, 2), keepdim=True)
        quadratic_raw = (block_rank - block_mean).square()
        # One fixed, predeclared permutation; no sample identities or RNG.
        scrambled_rank = torch.tensor(
            [0, 7, 2, 9, 4, 1, 8, 3, 6, 5],
            device=values.device, dtype=torch.float32).view(1, 10, 1) / 9.0
        scrambled_raw = scrambled_rank.square()
        quadratic, quadratic_norm2 = self._orthogonal_block_residual(
            quadratic_raw, linear, linear_norm2, valid, total)
        scrambled, scrambled_norm2 = self._orthogonal_block_residual(
            scrambled_raw, linear, linear_norm2, valid, total)
        occupied = (counts > 0).sum(dim=1, keepdim=True)
        support = ((occupied >= 3) & (quadratic_norm2 > 1e-10)
                   & (scrambled_norm2 > 1e-10)).to(valid.dtype)
        target_norm2 = centered_block.square().sum(dim=(1, 2), keepdim=True)
        quadratic = (quadratic * (target_norm2 / quadratic_norm2.clamp_min(1e-10)).sqrt()
                     * support)
        scrambled = (scrambled * (target_norm2 / scrambled_norm2.clamp_min(1e-10)).sqrt()
                     * support)
        duplicate = centered_block * support
        if self.curvature_detail == 'quadratic':
            fourth = quadratic
        elif self.curvature_detail == 'scrambled':
            fourth = scrambled
        else:
            fourth = duplicate
        incumbent_fifth = self._joint_coefficients(
            centered, centered_position, fourth, valid, counts,
            block_rank, scrambled_rank)
        fifth = self._coarse_extension_coefficients(
            incumbent_fifth, centered_position, fourth, linear,
            block_rank, scrambled_rank, valid, counts, total)
        coefficients = torch.stack(
            (centered, centered_position, block_coefficients, fourth, fifth), dim=-2)
        summaries = torch.bmm(
            coefficients.to(values.dtype).reshape(batch * 10, 5, 10),
            values.reshape(batch * 10, 10, width))
        summaries = summaries.reshape(batch, 10, 5, width).sum(dim=1)
        summaries = summaries / total.sqrt().unsqueeze(-1).to(summaries.dtype)
        return (summaries[:, 0, :], summaries[:, 1, :], summaries[:, 2, :],
                summaries[:, 3, :], summaries[:, 4, :], match_counts,
                match_matrix.bool().any(dim=-1))

    def _coarse_extension_coefficients(self, incumbent, position, quadratic,
                                       linear, rank, scrambled_rank,
                                       valid, counts, total):
        """Replace the redundant fifth direction at fixed learned capacity.

        Event-weighted projection uses the actual validity mask, including
        partial/noncontiguous blocks. Every candidate coefficient is block
        constant on valid events, hence orthogonal to the first target-match
        and second local-position coefficients as well. A common gate and
        norm make information content the main replacement contrast. The
        gate can itself carry information; the duplicate isolates that.
        """
        cubic = self._coarse_residual(
            rank * rank * rank, linear, quadratic, valid, total)
        scrambled = self._coarse_residual(
            scrambled_rank * scrambled_rank * scrambled_rank,
            linear, quadratic, valid, total)
        cubic_norm2 = cubic.square().sum(dim=(1, 2), keepdim=True)
        scrambled_norm2 = scrambled.square().sum(dim=(1, 2), keepdim=True)
        incumbent_norm2 = incumbent.square().sum(dim=(1, 2), keepdim=True)
        target_norm2 = position.square().sum(dim=(1, 2), keepdim=True)
        occupied = (counts > 0).sum(dim=1, keepdim=True)
        support = ((occupied >= 4) & (cubic_norm2 > 1e-10)
                   & (scrambled_norm2 > 1e-10)
                   & (incumbent_norm2 > 1e-10)
                   & (target_norm2 > 1e-10)).to(valid.dtype)
        cubic = (cubic * (target_norm2 / cubic_norm2.clamp_min(1e-10)).sqrt()
                 * support)
        scrambled = (scrambled
                     * (target_norm2 / scrambled_norm2.clamp_min(1e-10)).sqrt()
                     * support)
        duplicate = incumbent * support
        # All alternatives above execute in all cells; no modules or RNG added.
        if self.coarse_extension == 'cubic':
            return cubic
        if self.coarse_extension == 'scrambled':
            return scrambled
        if self.coarse_extension == 'duplicate':
            return duplicate
        return incumbent

    @staticmethod
    def _coarse_residual(raw, linear, quadratic, valid, total):
        """Two projection sweeps in FP32; zero bases are harmless."""
        residual = valid * raw
        linear_norm2 = linear.square().sum(dim=(1, 2), keepdim=True)
        quadratic_norm2 = quadratic.square().sum(dim=(1, 2), keepdim=True)
        for _ in range(2):
            mean = residual.sum(dim=(1, 2), keepdim=True) / total.unsqueeze(-1)
            residual = valid * (residual - mean)
            slope = ((residual * linear).sum(dim=(1, 2), keepdim=True)
                     / linear_norm2.clamp_min(1e-12))
            residual = valid * (residual - slope * linear)
            slope = ((residual * quadratic).sum(dim=(1, 2), keepdim=True)
                     / quadratic_norm2.clamp_min(1e-12))
            residual = valid * (residual - slope * quadratic)
        mean = residual.sum(dim=(1, 2), keepdim=True) / total.unsqueeze(-1)
        return valid * (residual - mean)

    def _joint_coefficients(self, affinity, position, fourth, valid, counts,
                            rank, scrambled_rank):
        """Retain the incumbent gate; vary only duplicate-match support/scale.

        All basis construction is FP32 and independent of learned values.
        Within-block centering gives orthogonality to every block-constant
        coefficient, including the inherited linear and quadratic moments.
        Two projection sweeps reduce finite-precision leakage into the
        remaining local-ordinal and target-match directions. The common
        support gate can carry availability information even in a duplicate
        direction. Build every inherited coefficient in all factorial cells;
        basic availability additionally admits matches lacking joint moments.
        """
        position_norm2 = position.square().sum(dim=(1, 2), keepdim=True)
        match_norm2 = affinity.square().sum(dim=(1, 2), keepdim=True)
        fourth_norm2 = fourth.square().sum(dim=(1, 2), keepdim=True)
        # Orthogonalize affinity against position first. A zero basis is
        # harmless; no inverse of a near-singular two-by-two matrix is used.
        basis = affinity
        for _ in range(2):
            slope = ((basis * position).sum(dim=(1, 2), keepdim=True)
                     / position_norm2.clamp_min(1e-10))
            basis = valid * (basis - slope * position)
            basis = valid * (basis - basis.sum(dim=-1, keepdim=True)
                             / counts.clamp_min(1))
        basis_norm2 = basis.square().sum(dim=(1, 2), keepdim=True)
        basis = basis * (basis_norm2 > 1e-10).to(valid.dtype)
        basis_norm2 = basis.square().sum(dim=(1, 2), keepdim=True)

        ordered = self._joint_residual(
            affinity * rank, position, position_norm2,
            basis, basis_norm2, valid, counts)
        scrambled = self._joint_residual(
            affinity * scrambled_rank, position, position_norm2,
            basis, basis_norm2, valid, counts)
        ordered_norm2 = ordered.square().sum(dim=(1, 2), keepdim=True)
        scrambled_norm2 = scrambled.square().sum(dim=(1, 2), keepdim=True)
        support = ((ordered_norm2 > 1e-10) & (scrambled_norm2 > 1e-10)
                   & (position_norm2 > 1e-10) & (match_norm2 > 1e-10)
                   & (fourth_norm2 > 1e-10)).to(valid.dtype)
        ordered = ordered * (position_norm2 / ordered_norm2.clamp_min(1e-10)).sqrt() * support
        scrambled = scrambled * (position_norm2 / scrambled_norm2.clamp_min(1e-10)).sqrt() * support
        duplicate_match = affinity * (position_norm2 / match_norm2.clamp_min(1e-10)).sqrt() * support
        duplicate_curvature = fourth * (position_norm2 / fourth_norm2.clamp_min(1e-10)).sqrt() * support
        # Gate and rescaling depend only on the authorized masked input IDs
        # and positions. No labels, learned values, new state or RNG enter.
        available = ((position_norm2 > 1e-10)
                     & (match_norm2 > 1e-10)).to(valid.dtype)
        available_norm = affinity * (position_norm2 / match_norm2.clamp_min(1e-10)).sqrt() * available
        available_raw = affinity * available
        joint_raw = affinity * support
        # All four cells build all four alternatives. The joint/norm cell
        # returns exactly the old expression for a same-seed parity bridge.
        if self.match_support == 'available':
            return available_norm if self.match_scaling == 'norm' else available_raw
        return duplicate_match if self.match_scaling == 'norm' else joint_raw

    @staticmethod
    def _joint_residual(raw, position, position_norm2, basis, basis_norm2, valid, counts):
        residual = valid * (raw - (raw * valid).sum(dim=-1, keepdim=True)
                            / counts.clamp_min(1))
        for _ in range(2):
            slope = ((residual * position).sum(dim=(1, 2), keepdim=True)
                     / position_norm2.clamp_min(1e-10))
            residual = valid * (residual - slope * position)
            slope = ((residual * basis).sum(dim=(1, 2), keepdim=True)
                     / basis_norm2.clamp_min(1e-10))
            residual = valid * (residual - slope * basis)
            residual = valid * (residual - residual.sum(dim=-1, keepdim=True)
                                / counts.clamp_min(1))
        return residual

    @staticmethod
    def _orthogonal_block_residual(raw, linear, linear_norm2, valid, total):
        mean = (raw * valid).sum(dim=(1, 2), keepdim=True) / total.unsqueeze(-1)
        residual = valid * (raw - mean)
        slope = ((residual * linear).sum(dim=(1, 2), keepdim=True)
                 / linear_norm2.clamp_min(1e-12))
        residual = valid * (residual - slope * linear)
        # Recenter once more to reduce finite-precision constant leakage.
        residual_mean = residual.sum(dim=(1, 2), keepdim=True) / total.unsqueeze(-1)
        residual = valid * (residual - residual_mean)
        return residual, residual.square().sum(dim=(1, 2), keepdim=True)

    def _compress_both(self, values, mask, auxiliary):
        batch, length, width = values.shape
        if length != 100:
            raise ValueError('Frozen protocol requires 100 history positions')
        groups, block_size = 10, 10
        valid = mask.bool().reshape(batch, groups, block_size)
        membership = auxiliary.reshape(batch, groups, block_size) & valid
        # Per temporal block: one uniform slot and two feedback strata.
        # Each branch independently covers every valid event exactly once.
        assignment = torch.stack((valid, valid & ~membership, membership), dim=-2)
        assignment = assignment.to(values.dtype).reshape(batch * groups, 3, block_size)
        counts = assignment.sum(-1)
        grouped = values.reshape(batch * groups, block_size, width)
        sums = torch.bmm(assignment, grouped)
        means = sums / counts.clamp_min(1).unsqueeze(-1)
        return means.reshape(batch, groups * 3, width), counts.reshape(batch, groups * 3)

    def _compress_with_match(self, values, mask, auxiliary, match_union):
        """Seven rows with population-matched counts and support emptiness.

        Valid-rank rotation operates inside a block, including noncontiguous
        validity masks, and preserves target-match cardinality. Special IDs
        have already been excluded by _target_match_summary. No labels,
        external identities, timestamps, state or RNG enter this assignment.
        """
        batch, length, width = values.shape
        if length != 100:
            raise ValueError('Frozen protocol requires 100 history positions')
        valid = mask.bool().reshape(batch, 10, 10)
        feedback = auxiliary.bool().reshape(batch, 10, 10) & valid
        matched = match_union.bool().reshape(batch, 10, 10) & valid
        rotated = self._rotate_valid_membership(matched, valid)
        # Both support pairs have identical per-block cardinalities and
        # empty-slot patterns. Duplicate control removes only aligned value
        # grouping; the target-match counts and rotated pair remain present.
        first_support = (matched if self.normalization_support == 'aligned_rotated'
                         else rotated)
        assignment = torch.stack(
            (valid, valid & ~feedback, feedback,
             valid & ~first_support, first_support,
             valid & ~rotated, rotated), dim=-2)
        assignment = assignment.to(values.dtype).reshape(batch * 10, 7, 10)
        counts = assignment.sum(dim=-1)
        sums = torch.bmm(assignment, values.reshape(batch * 10, 10, width))
        means = sums / counts.clamp_min(1).unsqueeze(-1)
        return means.reshape(batch, 70, width), counts.reshape(batch, 70)

    def _compress_unique_support(self, values, mask, auxiliary, match_union):
        """Five representatives of the inherited seven assignment rows.

        Empty slots remain zero rows and retain their normalization mass.
        Counts are used only by readout, never as Dice multiplicities.
        """
        batch, length, width = values.shape
        if length != 100:
            raise ValueError('Frozen protocol requires 100 history positions')
        valid = mask.bool().reshape(batch, 10, 10)
        feedback = auxiliary.bool().reshape(batch, 10, 10) & valid
        matched = match_union.bool().reshape(batch, 10, 10) & valid
        rotated = self._rotate_valid_membership(matched, valid)
        assignment = torch.stack(
            (valid, valid & ~feedback, feedback, valid & ~rotated, rotated), dim=-2)
        assignment = assignment.to(values.dtype).reshape(batch * 10, 5, 10)
        counts = assignment.sum(dim=-1)
        sums = torch.bmm(assignment, values.reshape(batch * 10, 10, width))
        means = sums / counts.clamp_min(1).unsqueeze(-1)
        return means.reshape(batch, 50, width), counts.reshape(batch, 50)

    @staticmethod
    def _expand_support_rows(rows):
        """[B,50,D] -> [B,70,D], preserving block and original row order.

        index_select is differentiable: two copies accumulate gradients back
        into one representative. No detach, state, RNG or mask changes.
        """
        batch, slots, width = rows.shape
        if slots != 50:
            raise ValueError('Expected five representatives per temporal block')
        index = torch.tensor([0, 1, 2, 3, 4, 3, 4], dtype=torch.long,
                             device=rows.device)
        return rows.reshape(batch, 10, 5, width).index_select(2, index).reshape(
            batch, 70, width)

    def _folded_support_activation(self, interactions, batch):
        """Affine maps on 50 rows; the SAME Dice sees all original 70 rows.

        After Dice, duplicate support outputs are identical in real
        arithmetic. Retain the first five representatives and accumulate
        their downstream multiplicities at the next expansion. Autograd
        through both gather and selection preserves the derivative of the
        tied computation. This relies on the audited Linear/Dice scorer;
        stochastic or cross-row layers would require a new derivation.
        Native BatchNorm owns its epsilon, biased batch variance, unbiased
        running variance, momentum, counter updates and coupled derivatives.
        """
        hidden = interactions
        for layer in self.local_activation:
            if isinstance(layer, Dice):
                expanded = self._expand_support_rows(hidden.reshape(batch, 50, -1))
                expanded = layer(expanded.reshape(batch * 70, -1))
                hidden = expanded.reshape(batch, 10, 7, -1)[:, :, :5, :].reshape(
                    batch * 50, -1)
            elif isinstance(layer, nn.Linear):
                hidden = layer(hidden)
            else:
                raise ValueError('Support folding only supports the audited Linear/Dice scorer')
        return hidden

    @staticmethod
    def _rotate_valid_membership(matched, valid):
        # Pack membership by valid-event rank, rotate by half the valid
        # count, then unpack to the original valid positions. Invalid
        # entries contribute zero even when scatter indices coincide.
        matched = matched & valid
        ranks = (valid.long().cumsum(dim=-1) - 1).clamp_min(0)
        counts = valid.long().sum(dim=-1, keepdim=True)
        packed = torch.zeros_like(ranks).scatter_add(-1, ranks, matched.long())
        source = (ranks + counts // 2).remainder(counts.clamp_min(1))
        return packed.gather(-1, source).bool() & valid

    def _pool_seven(self, weights, values, counts):
        batch = values.shape[0]
        weights = weights.reshape(batch, 70) * counts.to(weights.dtype)
        weighted = (weights.unsqueeze(-1) * values).reshape(
            batch, 10, 7, self.value_width)
        return weighted.sum(dim=1)

    def _interactions(self, target, values):
        keys = values[:, :, :self.key_width]
        action = values[:, :, self.key_width:]
        batch, slots, _ = keys.shape
        query = target[:, None, :].expand(-1, slots, -1)
        return torch.cat([query, keys, query - keys, query * keys, action], dim=-1).reshape(batch * slots, -1)

    def _primary_activation(self, interactions):
        """Forward-identical stop-gradient on nonreadout support rows.

        Flattened order is batch, block, seven strata. The first three strata
        retain the incumbent gradient paths. Before EACH Dice, rows 3..6
        become constants for moment estimation in the blocked cell. This
        removes their backward contribution to the preceding affine layer
        and embedding sources, while preserving their forward activations,
        batch moments and running-buffer updates at a fixed parameter state.
        Selected rows still differentiate through their own batch moments;
        shared scorer parameters and later updates can consequently differ.
        No extra scorer pass, parameters, persistent state or RNG is added.
        """
        if not self.training or self.support_gradient == 'attached':
            return self.local_activation(interactions)
        retained = (torch.arange(interactions.shape[0], device=interactions.device)
                    .remainder(7) < 3).unsqueeze(-1)
        hidden = interactions
        for layer in self.local_activation:
            if isinstance(layer, Dice):
                hidden = torch.where(retained, hidden, hidden.detach())
            hidden = layer(hidden)
        return hidden

    def _auxiliary_activation(self, interactions):
        # Functional evaluation uses the SAME learned scorer with parameter
        # stop-gradient fixed on. Do not use no_grad: gradients into each
        # allowed embedding source must survive the shared scoring operator.
        parameters = {
            name: (parameter.detach() if self.block_aux_scorer_grad else parameter)
            for name, parameter in self.local_activation.named_parameters()
        }
        # Independent registered state is updated by native BatchNorm during
        # training and read during eval. Fixed momentum0.01 makes train output
        # independent of previous running moments/counter values. The original
        # primary buffers are never supplied to this functional call.
        buffers = {name: getattr(self, storage_name)
                   for name, storage_name in self._auxiliary_state_names.items()}
        return torch.func.functional_call(
            self.local_activation, (parameters, buffers), (interactions,), strict=True)

    def _pool(self, weights, values, counts):
        batch, slots, _ = values.shape
        weights = weights.reshape(batch, slots) * counts.to(weights.dtype)
        weighted = (weights.unsqueeze(-1) * values).reshape(batch, 10, 3, self.value_width)
        return weighted.sum(dim=1)

    def _context_history_correction(self, index, context, history):
        # Only supplied context and already-masked compressed history enter.
        # FP32 bounded projections and a shared RMS denominator reduce gross
        # scale confounding between product and sum. Their geometry and
        # optimization still differ; equal norms do not imply equal capacity.
        query = torch.tanh(self.context_queries[index](context).float())
        summary = torch.tanh(self.history_projections[index](history).float())
        if self.context_fusion == 'product':
            fused = query * summary
        else:
            fused = (query + summary) * (2.0 ** -0.5)
        fused = fused * torch.rsqrt(fused.square().mean(dim=-1, keepdim=True) + 1e-4)
        return self.interaction_outputs[index](fused.to(history.dtype))

    def forward(self, inputs):
        # A transient graph reference only, never a buffer or checkpoint field.
        # Clear before representing the next batch, including eval batches.
        self._training_source_bit_context = None
        context, target, values, mask, auxiliary, detail, position_detail, block_detail, curvature_detail, joint_detail, match_counts, match_union = self._represent(inputs)
        batch = values.shape[0]
        if self.training:
            occupied = mask.bool().any(dim=1, keepdim=True)
            collected_contexts = _CollectedContextTransfer.apply(
                context, occupied, self.parallel_orientation,
                self.orthogonal_orientation, self.click_relative_transfer,
                self.orthogonal_assignment)
            self._training_source_bit_context = (collected_contexts, occupied)
            product_context_by_label = dict(zip(('cart', 'fav', 'buy'), collected_contexts[1:4]))
            # Other auxiliary context uses are blocked below. Preserve their
            # forward values, while click uses only the first collection edge.
            transfer_context = context
            context = collected_contexts[0]
        if (self.training and self.training_support == 'full') or self.inference_support == 'full':
            # This training path is exactly the incumbent's operator sequence:
            # all 70 rows enter Dice and retain native support gradients.
            all_values, all_counts = self._compress_with_match(values, mask, auxiliary, match_union)
            primary_weights = self._primary_activation(self._interactions(target, all_values))
            all_strata = self._pool_seven(primary_weights, all_values, all_counts)
            uniform = all_strata[:, 0, :]
            values = all_values.reshape(batch, 10, 7, self.value_width)[:, :, :3, :].reshape(
                batch, 30, self.value_width)
            counts = all_counts.reshape(batch, 10, 7)[:, :, :3].reshape(batch, 30)
            primary_strata = all_strata[:, :3, :]
        elif self.training:
            # Folded and unweighted use five unique assignment rows. The
            # compression-only control restores seven BEFORE every scorer
            # operation, isolating compression arithmetic from affine work.
            unique_values, unique_counts = self._compress_unique_support(
                values, mask, auxiliary, match_union)
            if self.training_support == 'compression_only':
                all_values = self._expand_support_rows(unique_values)
                all_counts = self._expand_support_rows(unique_counts.unsqueeze(-1)).squeeze(-1)
                primary_weights = self._primary_activation(self._interactions(target, all_values))
                all_strata = self._pool_seven(primary_weights, all_values, all_counts)
                primary_strata = all_strata[:, :3, :]
            else:
                interactions = self._interactions(target, unique_values)
                if self.training_support == 'folded':
                    primary_weights = self._folded_support_activation(interactions, batch)
                else:
                    # Ablation removes two support copies from every Dice
                    # population. It is not an equivalence-preserving path.
                    primary_weights = self.local_activation(interactions)
                weights = primary_weights.reshape(batch, 10, 5)[:, :, :3].reshape(batch, 30)
                read_values = unique_values.reshape(batch, 10, 5, self.value_width)[:, :, :3, :].reshape(
                    batch, 30, self.value_width)
                read_counts = unique_counts.reshape(batch, 10, 5)[:, :, :3].reshape(batch, 30)
                primary_strata = self._pool(weights, read_values, read_counts)
            uniform = primary_strata[:, 0, :]
            values = unique_values.reshape(batch, 10, 5, self.value_width)[:, :, :3, :].reshape(
                batch, 30, self.value_width)
            counts = unique_counts.reshape(batch, 10, 5)[:, :, :3].reshape(batch, 30)
        else:
            # Pruning is valid only in eval mode. Never change train/eval
            # flags, running buffers, parameters, or the host batch policy.
            if self.inference_support == 'score_pruned':
                # Retain the original compression arithmetic to isolate the
                # effect of reducing scorer GEMM rows from 70 to 30.
                all_values, all_counts = self._compress_with_match(values, mask, auxiliary, match_union)
                values = all_values.reshape(batch, 10, 7, self.value_width)[:, :, :3, :].reshape(
                    batch, 30, self.value_width)
                counts = all_counts.reshape(batch, 10, 7)[:, :, :3].reshape(batch, 30)
            else:
                # The first three assignment rows of seven-row compression
                # are exactly the inherited uniform/feedback assignment.
                values, counts = self._compress_both(values, mask, auxiliary)
            primary_weights = self._primary_activation(self._interactions(target, values))
            primary_strata = self._pool(primary_weights, values, counts)
            uniform = primary_strata[:, 0, :]

        if self.training:
            # Item fields share tables across current target and past events:
            # gate both uses together. Historical action joins this group.
            # Context includes only supplied non-item/non-action features.
            aux_context = transfer_context.detach() if self.block_aux_context_grad else transfer_context
            aux_target = target.detach() if self.block_aux_item_grad else target
            aux_values = values.detach() if self.block_aux_item_grad else values
            aux_detail = detail.detach() if self.block_aux_item_grad else detail
            aux_position_detail = position_detail.detach() if self.block_aux_item_grad else position_detail
            aux_block_detail = block_detail.detach() if self.block_aux_item_grad else block_detail
            aux_curvature_detail = curvature_detail.detach() if self.block_aux_item_grad else curvature_detail
            aux_joint_detail = joint_detail.detach() if self.block_aux_item_grad else joint_detail
            aux_weights = self._auxiliary_activation(self._interactions(aux_target, aux_values))
            feedback_population_strata = self._pool(aux_weights, aux_values, counts)
            # Execute feedback scoring in every training cell to match work
            # and collect a complete independent running-state trajectory.
            # Primary strata have all support rows in their native moments;
            # detach the entire result to preserve auxiliary source ownership.
            if self.auxiliary_train_population == 'primary':
                aux_strata = primary_strata.detach()
            else:
                aux_strata = feedback_population_strata
        else:
            # The eval factor never changes training operations. Each choice
            # uses only frozen running moments; eval cannot update either set.
            aux_context, aux_target = context, target
            aux_detail = detail
            aux_position_detail = position_detail
            aux_block_detail = block_detail
            aux_curvature_detail = curvature_detail
            aux_joint_detail = joint_detail
            if self.auxiliary_eval_population == 'feedback':
                aux_weights = self._auxiliary_activation(self._interactions(target, values))
                aux_strata = self._pool(aux_weights, values, counts)
            else:
                aux_strata = primary_strata
        feedback = aux_strata[:, 1, :] + aux_strata[:, 2, :]
        grouped_counts = counts.reshape(counts.shape[0], 10, 3)
        total_count = grouped_counts[:, :, 0].sum(dim=1, keepdim=True)
        has_history = (total_count > 0).to(uniform.dtype)
        predictions = {}
        for index, (label, tower) in enumerate(zip(self.feature_map.labels, self.towers)):
            if label == 'is_click':
                task_context, task_target, base = context, target, uniform
                task_detail = detail
                task_position_detail = position_detail
                task_block_detail = block_detail
                task_curvature_detail = curvature_detail
                task_joint_detail = joint_detail
            else:
                task_context, task_target, base = aux_context, aux_target, feedback
                task_detail = aux_detail
                task_position_detail = aux_position_detail
                task_block_detail = aux_block_detail
                task_curvature_detail = aux_curvature_detail
                task_joint_detail = aux_joint_detail
            # Keep the two uses numerically identical. Detach only the
            # selected context edge; every auxiliary adapter/tower parameter
            # remains in its task loss graph. Eval has no gradient gating.
            adapter_context = task_context
            tower_context = task_context
            if self.training and label != 'is_click':
                if self.block_aux_adapter_context_grad:
                    adapter_context = adapter_context.detach()
                if self.block_aux_tower_context_grad:
                    tower_context = tower_context.detach()
            # Split only the two context input edges. The master adapter gate
            # above remains effective; private adapter parameters never detach.
            residual_context = adapter_context
            product_context = adapter_context
            if self.training and label != 'is_click':
                if self.block_aux_residual_context_grad:
                    residual_context = residual_context.detach()
                product_context = product_context_by_label[label]
                if self.block_aux_product_context_grad:
                    product_context = product_context.detach()
                # Only input context cotangents change. Private product-module
                # gradients and all true-label task losses stay ordinary.
            signal = torch.cat([task_detail, task_position_detail, task_block_detail,
                                task_curvature_detail, task_joint_detail], dim=-1)
            residual_features = torch.cat([residual_context, task_target, base,
                                           match_counts.to(base.dtype), signal], dim=-1)
            # Task routing is required by __init__. Each adapter still
            # receives its own task's features once; every parameter remains
            # in the loss graph. Unselected adapters are stateless and their
            # outputs have zero loss derivative in the all-bank control.
            if self.residual_execution == 'direct':
                correction = self.residual_bank[index](residual_features)
            else:
                bank = torch.stack([adapter(residual_features)
                                    for adapter in self.residual_bank], dim=1)
                correction = bank[:, index, :] if self.residual_routing == 'task' else bank.mean(dim=1)
            history_features = torch.cat([base, match_counts.to(base.dtype), signal], dim=-1)
            correction = correction + self._context_history_correction(
                index, product_context, history_features)
            pooled = base + has_history * correction
            features = torch.cat([tower_context, task_target, pooled], dim=-1)
            predictions[label + '_pred'] = torch.sigmoid(tower(features))
        return predictions

    def training_loss(self, inputs, probabilities, train_labels):
        """True joint BCE plus a zero-valued source-bit surrogate.

        The source is BCE(y_assignment)-BCE(0). Native assignment uses the
        true labels; batch_mean broadcasts each task's batch prevalence.
        Native, cycle, mean and zero source BCE graphs run in all cells.
        Away from inherited probability-BCE saturation/clamps, their logit
        derivatives are exactly -y_assignment. No new probability/logit
        transform, extra forward, module, optimizer state or RNG is added.
        Only the detached source cotangents feed the global orientation bit;
        private learning and native transfer geometry retain true labels.
        """
        if (probabilities.ndim != 2 or train_labels.ndim != 2
                or probabilities.shape != train_labels.shape
                or probabilities.shape[1] != len(self.feature_map.labels)):
            raise ValueError('Expected aligned prediction and training-label matrices [B,4]')
        if AUXILIARY_OBJECTIVE != 'joint':
            raise ValueError('Bit-source study freezes true-label joint private learning')
        if probabilities.shape[0] < 2:
            raise ValueError('This frozen full-batch training study requires at least two examples')
        cached = self._training_source_bit_context
        if not self.training or cached is None:
            raise ValueError('training_loss requires a fresh gradient-enabled training forward')
        edges, occupied = cached
        if probabilities.shape[0] != occupied.shape[0]:
            raise ValueError('Training context and prediction batch mismatch')
        # BCE is unsafe inside CUDA autocast. Keep the inherited FP32 path.
        with torch.autocast(device_type=probabilities.device.type, enabled=False):
            truth = train_labels.detach().to(dtype=torch.float32)
            source_targets = _zero_source_targets(probabilities)
            cycle_targets = _cycle_source_targets(truth)
            mean_targets = _mean_source_targets(truth)
            losses = []
            zero_source_losses = []
            source_true_losses = []
            cycle_source_losses = []
            mean_source_losses = []
            for index, label in enumerate(self.feature_map.labels):
                task_loss = nn.functional.binary_cross_entropy(
                    probabilities[:, index].float(), truth[:, index], reduction='mean')
                losses.append(1.0 * task_loss)
                if label != 'is_click':
                    source_true_losses.append(task_loss)
                    cycle_source_losses.append(nn.functional.binary_cross_entropy(
                        probabilities[:, index].float(), cycle_targets[:, index], reduction='mean'))
                    mean_source_losses.append(nn.functional.binary_cross_entropy(
                        probabilities[:, index].float(), mean_targets[:, index], reduction='mean'))
                    zero_source_losses.append(nn.functional.binary_cross_entropy(
                        probabilities[:, index].float(), source_targets[:, index], reduction='mean'))
            true_auxiliary_loss = torch.stack(source_true_losses).sum()
            zero_auxiliary_loss = torch.stack(zero_source_losses).sum()
            cycle_auxiliary_loss = torch.stack(cycle_source_losses).sum()
            mean_auxiliary_loss = torch.stack(mean_source_losses).sum()
            # Match graph work, including zero-coefficient graphs. These
            # labels never train private parameters or alter native geometry.
            native_weight, cycle_weight, mean_weight = {
                'native': (1.0, 0.0, 0.0),
                'batch_cycle': (0.0, 1.0, 0.0),
                'batch_mean': (0.0, 0.0, 1.0),
            }[self.source_label_assignment]
            source_loss = (native_weight * true_auxiliary_loss
                           + cycle_weight * cycle_auxiliary_loss
                           + mean_weight * mean_auxiliary_loss
                           - zero_auxiliary_loss)
            source_gradients = torch.autograd.grad(
                source_loss, edges[1:4], retain_graph=True,
                create_graph=False, allow_unused=False)
            # Subtraction before multiplication keeps the surrogate value
            # exactly zero for finite values, without a higher-order graph.
            surrogate = torch.stack([
                ((edge.float() - edge.detach().float()) * gradient.detach().float()).sum()
                for edge, gradient in zip(edges[4:7], source_gradients)
            ]).sum()
            result = torch.stack(losses).sum() + surrogate
        self._training_source_bit_context = None
        return result
