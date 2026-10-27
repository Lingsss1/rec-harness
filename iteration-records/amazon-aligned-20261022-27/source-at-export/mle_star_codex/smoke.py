"""Exercise four GPU workers on real data before launching the full search."""
import argparse
from pathlib import Path

from mle_star_codex.run import Controller, baseline_source, read, save, validate_source


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    c = read(a.config)
    c.update(run_dir=str(Path(c['run_dir']) / 'smoke'), epochs=1, batches=2,
             eval_users=20, budget_gpu_seconds=600, final_reserve=0,
             trial_timeout=120, minimum_trial_seconds=30)
    controller = Controller(c)
    baseline = baseline_source(Path(c['repo']) / 'gagc/templates/nextitnet_perdataset/train.py')
    tail = baseline.replace('user_emb = out[0, lengths[0]]', 'user_emb = out[0, -1]')
    validate_source(tail, baseline, 'NextItNetModel')
    activation = baseline.replace('y = self.dropout(y)', 'y = self.dropout(self.activation(y))')
    validate_source(activation, baseline, 'NextItNetBlock')
    specs = []
    for i, source in enumerate([baseline, tail, activation, baseline]):
        path = controller.code / f'smoke-t{i}.py'
        path.write_text(source)
        specs.append({'key': f'smoke-t{i}', 'model': path})
    results = controller.jobs(specs)
    save(controller.root / 'summary.json', results)
    if any(r['status'] != 'success' for r in results):
        raise RuntimeError('GPU smoke failed; inspect smoke/trials/*/worker.log')
    import numpy as np
    first = np.load(Path(results[0]['output']) / 'scores.npy')
    repeat = np.load(Path(results[3]['output']) / 'scores.npy')
    if not np.allclose(first, repeat, atol=1e-5, rtol=1e-5):
        raise RuntimeError('seed=2024 predictions differ across GPUs 0 and 3')
    print('FOUR_GPU_SMOKE_PASSED', flush=True)


if __name__ == '__main__':
    main()
