"""Download official Amazon 2023 inputs and invoke upstream preprocessing."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--seed', type=int, default=2024)
    a = p.parse_args()
    raw = a.root.resolve() / 'raw'
    files = [f'benchmark/5core/last_out/Movies_and_TV.{s}.csv' for s in ('train', 'valid', 'test')]
    files += ['raw/meta_categories/meta_Movies_and_TV.jsonl']

    def download(rel):
        dest = raw / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            temporary = dest.with_suffix(dest.suffix + '.partial')
            subprocess.run(['curl', '-fL', '--retry', '5', '--connect-timeout', '30',
                            '--max-time', '3600', '-o', str(temporary),
                            'https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/' + rel], check=True)
            temporary.replace(dest)
        h = hashlib.sha256()
        with dest.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return {'file': rel, 'bytes': dest.stat().st_size, 'sha256': h.hexdigest()}

    cached = Path('/tmp/Movies_and_TV.valid.csv')
    target = raw / files[1]
    if cached.exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(download, files))
    subprocess.run([sys.executable, str(a.repo.resolve() / 'gagc/data_preprocess.py'),
                    '--dataset', 'Movies_and_TV', '--source', '2023', '--local-dir', str(raw),
                    '--data_dir', str(a.root.resolve() / 'trainval'),
                    '--test_dir', str(a.root.resolve() / 'test'), '--seed', str(a.seed)], check=True)
    # Only the vocabulary size is exposed to training; test labels stay out of
    # training workers and all Codex prompts.
    itemnum = 0
    for split in ('train', 'valid', 'test'):
        directory = 'test' if split == 'test' else 'trainval'
        with (a.root / directory / f'Movies_and_TV_{split}.txt').open() as f:
            for line in f:
                _, item = map(int, line.split())
                itemnum = max(itemnum, item)
    (a.root / 'trainval/dataset_stats.json').write_text(json.dumps({'Movies_and_TV': {'itemnum': itemnum}}))
    (a.root / 'provenance.json').write_text(json.dumps({'source': 'Amazon Reviews 2023',
        'sampling_ratio': 0.05, 'preprocessing_seed': a.seed, 'inputs': receipts}, indent=2))


if __name__ == '__main__':
    main()
