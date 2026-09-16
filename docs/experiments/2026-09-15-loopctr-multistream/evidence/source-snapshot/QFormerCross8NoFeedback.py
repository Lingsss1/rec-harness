"""Generalize the near-best two-state recurrence without prediction feedback."""
from functools import partial
import math
from .QFormerCross8 import QFormerCross8, RecursiveSequenceCrossValueStage


class NoFeedbackStage(RecursiveSequenceCrossValueStage):
    def __init__(self, *args, work_steps=2, entry_weight=.5, **kwargs):
        super().__init__(*args, **kwargs)
        if len(self.layers) != 3:
            raise ValueError('Require three backbone layers')
        self.work_steps = work_steps
        self.entry_weight = entry_weight

    def forward_components(self, queries, sequence_features, sequence_mask=None):
        entry = self.layers[0](queries, sequence_features, sequence_mask)
        working = prediction = entry
        for step in range(self.work_steps):
            if step == 0:
                update_input = entry
            elif self.entry_weight == .5:
                # Preserve the original candidate's exact arithmetic.
                update_input = (entry + working) / 2
            else:
                update_input = self.entry_weight * entry + (1-self.entry_weight) * working
            working = self.layers[1](update_input, sequence_features, sequence_mask)
            prediction = self.layers[2](prediction, working, None)
        mixed = self.post_self_attn(prediction, prediction, prediction)
        prediction = self.post_self_norm(prediction + self.dropout(mixed))
        return prediction, self.dropout(self.post_ffn(prediction))


class QFormerCross8NoFeedback(QFormerCross8):
    def __init__(self, feature_map, model_id='QFormerCross8NoFeedback',
                 work_steps=2, entry_weight=.5, **kwargs):
        if type(work_steps) is not int or not 1 <= work_steps <= 4:
            raise ValueError('work_steps must be integer 1..4')
        if isinstance(entry_weight, bool) or not math.isfinite(entry_weight) or not 0 <= entry_weight <= 1:
            raise ValueError('entry_weight must be finite 0..1')
        for k in ('task_unified_post_ffn_scales','task_unified_post_ffn_eval_scales'):
            if kwargs.get(k) is not None:
                raise ValueError('Keep original shared readout')
        if kwargs.get('unified_post_ffn_extra_layers',0):
            raise ValueError('Keep original post FFN')
        super().__init__(feature_map, model_id=model_id,
            _unified_qformer_cls=partial(NoFeedbackStage,work_steps=work_steps,
                entry_weight=entry_weight), **kwargs)
