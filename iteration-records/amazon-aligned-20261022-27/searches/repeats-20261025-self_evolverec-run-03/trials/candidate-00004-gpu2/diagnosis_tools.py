import torch


class DiagnosisProbe:
    def __init__(self, view, device):
        self.view = view
        self.device = device

    def run_full_diagnosis(self, train_loader):
        model = self.view.model
        was_training = model.training
        model.eval()
        count = 0
        neural_margin = fused_margin = residual_margin = 0.0
        try:
            with torch.no_grad():
                for _, sequence, positive, negative in train_loader:
                    sequence = sequence.to(self.device, dtype=torch.long)
                    positive = positive.to(self.device, dtype=torch.long)
                    negative = negative.to(self.device, dtype=torch.long)
                    valid = (sequence != 0) & (positive != 0) & (negative != 0)
                    features = model.log2feats(sequence)
                    neural_pos = (features * model.item_emb(positive)).sum(-1)
                    neural_neg = (features * model.item_emb(negative)).sum(-1)
                    # Omitted table IDs explicitly select the full training table zero.
                    rp = model.residual(sequence, positive)
                    rn = model.residual(sequence, negative)
                    boost = model.transition_gate_logit.sigmoid() * (rp - rn)
                    margin = neural_pos - neural_neg
                    count += int(valid.sum().item())
                    neural_margin += margin[valid].sum().item()
                    fused_margin += (margin + boost)[valid].sum().item()
                    residual_margin += boost[valid].sum().item()
        finally:
            model.train(was_training)
        definitions = {
            "valid_positions": "Number of nonpadding diagnostic training positions scored.",
            "neural_margin": "Mean positive-minus-negative neural logit on diagnostic training positions; in-sample.",
            "fused_margin": "Mean positive-minus-negative combined logit using the full training-only transition table; in-sample, not an unbiased generalization estimate.",
            "residual_margin": "Mean gated residual positive-minus-negative contribution on the same diagnostic positions; in-sample.",
            "gate": "Sigmoid of the learned scalar residual gate, bounded between zero and one.",
        }
        metrics = {
            "valid_positions": count,
            "neural_margin": neural_margin / count if count else None,
            "fused_margin": fused_margin / count if count else None,
            "residual_margin": residual_margin / count if count else None,
            "gate": float(model.transition_gate_logit.sigmoid().detach().cpu()),
        }
        return {"metrics": metrics, "metric_definitions": definitions}
