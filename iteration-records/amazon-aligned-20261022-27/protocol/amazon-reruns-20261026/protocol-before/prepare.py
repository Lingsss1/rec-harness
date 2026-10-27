"""Prepare the exact four-category RecHarness default splits, outside GPU budget."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DATASETS = ['Movies_and_TV', 'Industrial_and_Scientific', 'Electronics', 'CDs_and_Vinyl']
WORKSPACE = Path(__file__).resolve().parents[1]
ROOT = WORKSPACE / 'artifacts/amazon-aligned-20261022'
REPO = WORKSPACE / 'artifacts/RecHarness-6lyc'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(root=ROOT):
    data = root / 'data'
    data.mkdir(parents=True, exist_ok=True)
    start = time.time()
    raw = WORKSPACE / 'artifacts/amazon-reviews-2023'
    verification = json.loads((raw / 'verification.json').read_text())
    assert verification['all_verified'] and len(verification['files']) == 16

    def category(name):
        receipt = data / (name + '.prepared.json')
        if receipt.exists():
            return json.loads(receipt.read_text())
        command = [sys.executable, str(REPO / 'gagc/data_preprocess.py'),
                   '--dataset', name, '--source', '2023', '--local-dir', str(raw),
                   '--data_dir', str(data / 'trainval'), '--test_dir', str(data / 'test'),
                   '--seed', '0']
        with (data / (name + '.prepare.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4'})
        paths = [data / directory / (name + suffix) for directory, suffix in [
            ('trainval', '_train.txt'), ('trainval', '_valid.txt'), ('test', '_test.txt'),
            ('trainval', '_text_name_dict.json.gz')]]
        result = {'dataset': name, 'files': {str(p.relative_to(data)): sha(p) for p in paths}}
        receipt.write_text(json.dumps(result, indent=2) + '\n')
        print('PREPARED ' + name, flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(category, DATASETS))
    with (data / 'stats.log').open('w') as log:
        subprocess.run([sys.executable, str(REPO / 'gagc/compute_dataset_stats.py'),
                        '--data_dir', str(data / 'trainval'), '--test_dir', str(data / 'test'),
                        '--output_dir', str(data / 'trainval')], stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    result = {'status': 'ready', 'preprocessing_seed': 0, 'training_seed': 42,
              'source': 'RecHarness run2.sh defaults', 'datasets': DATASETS,
              'gpu_budget_charge': 0, 'wall_seconds': time.time() - start,
              'preprocessor_sha256': sha(REPO / 'gagc/data_preprocess.py'),
              'raw_verification_sha256': sha(raw / 'verification.json'), 'receipts': receipts,
              'stats_sha256': sha(data / 'trainval/dataset_stats.json')}
    (data / 'prepared.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    prepare(parser.parse_args().root.resolve())
