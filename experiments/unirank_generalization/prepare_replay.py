"""Create a fresh runnable workspace from a published evidence bundle; no training/download."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def prepare(bundle, workspace, upstream):
    bundle, workspace, upstream = bundle.resolve(), workspace.resolve(), upstream.resolve()
    if workspace.exists():
        raise ValueError('Use a new replay workspace')
    expected = json.loads((bundle / 'evidence/setup.json').read_text())['upstream_commit']
    actual = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != expected:
        raise ValueError('Check out the pinned UniRank commit first: ' + expected)
    dependencies = json.loads((bundle / 'evidence/upstream-source-identity.json').read_text())['files_sha256']
    for name, expected_hash in dependencies.items():
        if hashlib.sha256((upstream / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError('Pinned framework dependency differs: ' + name)
    workspace.mkdir(parents=True)
    shutil.copytree(bundle / 'source', workspace / 'source')
    setup = json.loads((bundle / 'evidence/setup.json').read_text())
    setup['upstream_path'] = str(upstream)
    (workspace / 'setup.json').write_text(json.dumps(setup, indent=2) + '\n')
    for source in sorted((bundle / 'datasets').iterdir()):
        target = workspace / 'datasets' / source.name
        shutil.copytree(source, target)
        params = json.loads((target / 'parameters.json').read_text())
        for split in ['train', 'valid', 'test']:
            params[split + '_data'] = str(target / 'data' / split / 'data')
            for kind in ['user_info', 'item_info']:
                params[split + '_' + kind] = str(target / 'data' / split / kind)
        params['data_root'] = str(target / 'data')
        (target / 'parameters.json').write_text(json.dumps(params, indent=2) + '\n')
        # The archived receipt must never mark absent local data as ready.
        (target / 'acquisition-status.json').write_text(json.dumps({'phase': 'awaiting_local_download_and_verification'}) + '\n')
    print(workspace)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.bundle, args.workspace, args.upstream)
