"""Recovery: evaluate frozen selected checkpoint; no training or search."""
import fcntl
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
sys.path.insert(0, str(WORKSPACE))
from mle_star_codex.run import Controller, read, save, sha


def main():
    with (ROOT / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = read(ROOT / 'config.json')
        frozen = read(ROOT / 'frozen.json')
        assert sha(ROOT / 'config.json') == frozen['config_sha256']
        for name in ('run.py', 'worker.py'):
            path = WORKSPACE / 'mle_star_codex' / name
            assert sha(path) == frozen['files'][str(path)], 'Frozen worker/controller changed'
        progress, selection = read(ROOT / 'progress.json'), read(ROOT / 'selection.json')
        assert progress['completed_candidates'] == progress['candidate_budget'] == 44
        assert not progress['inflight']
        assert selection['id'] == progress['best_id']
        specs = []
        for label, candidate in [('baseline', 'baseline'), ('selected', selection['id'])]:
            old = read(ROOT / 'trials' / ('final-' + label) / 'receipt.json')
            assert old['status'] == 'crash' and "invalid choice: 'predict'" in old['error']
            directory = ROOT / 'trials' / candidate
            original = read(directory / 'receipt.json')
            source = ROOT / 'models' / (candidate + '.py')
            checkpoint = directory / 'checkpoint.pt'
            assert original['status'] == 'success'
            assert sha(source) == original['source_sha256']
            assert sha(checkpoint) == original['checkpoint_sha256']
            specs.append({'key': 'retest-gpu-final-' + label, 'model': source,
                'mode': 'test', 'checkpoint': checkpoint})
        manifest = ROOT / 'final-test-recovery-gpu.json'
        if not manifest.exists():
            save(manifest, {'reason': 'Original unsupported predict mode failed before evaluation; first recovery also stopped before evaluation because sandbox CUDA initialization failed',
                'correction': 'Use supported test mode with identical frozen source/checkpoints/data/seed',
                'training_calls_added': 0, 'search_calls_added': 0,
                'script_sha256': sha(__file__), 'time': time.time(),
                'original_failed_trials': ['final-baseline', 'final-selected'],
                'new_trial_keys': [s['key'] for s in specs]})
        else:
            assert read(manifest)['script_sha256'] == sha(__file__)
        controller = Controller(config)
        tests = controller.jobs(specs, final=True)
        assert all(t['status'] == 'success' for t in tests), 'Inspect preserved retest receipts'
        reference = read(WORKSPACE / 'artifacts/autoresearch-movies-2024/trials/final-selected/receipt.json') if (WORKSPACE / 'artifacts/autoresearch-movies-2024/trials/final-selected/receipt.json').exists() else None
        assert tests[0]['rows_sha256'] == tests[1]['rows_sha256']
        if reference:
            assert tests[1]['rows_sha256'] == reference['rows_sha256']
        assert tests[0]['metrics']['HR@10'] == 0.3650548033526757
        baseline = read(ROOT / 'trials/baseline/receipt.json')
        training_calls = sum(read(p)['mode'] == 'train' for p in (ROOT / 'trials').glob('*/request.json') if p.parent.name != 'baseline')
        selected_training = read(ROOT / 'trials' / selection['id'] / 'receipt.json')
        summary = {'status': 'complete', 'method': config['method'], 'candidate_budget': 44,
            'completed_candidates': 44, 'successful_candidates': sum(not n['buggy'] for n in progress['nodes']),
            'failed_candidates': sum(bool(n['buggy']) for n in progress['nodes']),
            'training_calls': training_calls, 'seed': config['seed'], 'selected_id': selection['id'],
            'baseline_validation': baseline['metrics'], 'selected_validation': selected_training['metrics'],
            'selected_validation_hr10': selection['validation_hr10'],
            'baseline_test': tests[0]['metrics'], 'selected_test': tests[1]['metrics'],
            'test_rows_sha256': tests[1]['rows_sha256'], 'allocated_gpu_seconds': controller.spent(),
            'model': config['model'], 'effort': config['effort'],
            'final_test_recovery': str(manifest), 'final_test_keys': [s['key'] for s in specs],
            'limitations': ['single training seed', 'candidate-attempt matched, not successful-training/GPU-time/token matched',
                'task-specific frozen NextItNet adaptation', 'source and scoring-constraint failures consumed candidate slots',
                'test split used in earlier studies', 'prior MLE-STAR reasoning settings unchanged', 'no ensemble']}
        target = ROOT / 'summary.json'
        if target.exists():
            assert read(target) == summary, 'Refuse to overwrite a differing summary'
        else:
            save(target, summary)
        controller.event('final_test_recovery_complete', selected_test=summary['selected_test'], summary_path=str(target))


if __name__ == '__main__':
    main()
