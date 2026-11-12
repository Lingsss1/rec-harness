"""Stage the workspace's code and historical evidence, excluding datasets/weights.

This writes an isolated publication tree, never the user's Git index or branch.
Network publication is a separate step. Large diagnostic text is losslessly gzip
encoded to stay below GitHub's per-file limit; every source/destination is hashed.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

TEXT = {'.py', '.json', '.jsonl', '.txt', '.log', '.md', '.yaml', '.yml', '.toml', '.csv',
        '.patch', '.diff', '.sh', '.ts', '.tsx', '.js', '.mjs', '.css', '.sql', '.ini', '.cfg', '.xml', '.cff'}
NAMES = {'LICENSE', 'NOTICE', '.gitignore', 'Makefile', 'Dockerfile', 'MANIFEST.in'}
BLOCKED_DIRS = {'.git', 'node_modules', '__pycache__', '.pytest_cache', '.cache', '.venv', 'venv',
                'site-packages', 'dist-packages', 'runtime-packages', 'reference-cache', '.codex', '.aws', '.ssh', '.agents'}
DATA_DIRS = {'data', 'raw', 'prepared', 'processed', 'downloads', 'raw_data', 'dataset', 'datasets', 'wheels'}
DATA_METADATA = {'feature_map.json', 'meta_data.json', 'dataset-source-manifest.json', 'acquisition-status.json',
                 'block_manifest.json', 'protocol.json', 'parameters.json', 'candidate-config.json',
                 'joint1-config.json', 'joint3-config.json', 'din-config.json', 'reference-reuse.json'}
ARTIFACT_EXCLUDE = {'exports', 'publications', 'unirank-env', 'kuairand-env', 'amazon-reviews-2023'}
SECRET_NAMES = {'auth.json', 'credentials.json', 'credentials', 'id_rsa', 'id_ed25519', 'feature_vocab.json',
                'feature_processor.pkl', '.netrc', '.npmrc', '.pypirc'}
SECRET_PATTERNS = [
    re.compile(rb'(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{24,}'),
    re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|hf_[A-Za-z0-9]{25,})'),
    re.compile(rb'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    re.compile(rb'https?://[A-Za-z0-9._~%-]+:[A-Za-z0-9._~%+!$*(),;=-]+@'),
]
# Reviewed upstream tests contain deliberately synthetic token/header fixtures.
# Digest-based exceptions never exempt an edited file containing a real secret.
BENIGN_FIXTURE_SHA256 = {
    '012c86258c0ef986de70380538ed818d2895994337f64d4201dba284438743c9',
    '6a4e17c725dc672395e6ef7951e33a470143b80e3e14a73f0c90fb4346c74bfb',
    '152dcfc3cc670138d073d607e84754d64fa8cedd7d45920b2cc547ef366a9a52',
}


def git(*args, cwd=None):
    return subprocess.check_output(['git', *args], cwd=cwd)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def export(project, publication, parent):
    project, publication = project.resolve(), publication.resolve()
    assert not publication.exists(), 'Use a fresh publication folder'
    stage = publication / 'stage'
    stage.mkdir(parents=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d')
    history_rel = Path('iteration-records') / ('workspace-history-' + stamp)
    history = stage / history_rel
    history.mkdir(parents=True)
    inventory = {}
    excluded, redactions, nested = [], [], []
    aggregates, counts = Counter(), Counter()
    remote_tree = {}
    for record in git('ls-tree', '-r', '-z', parent).split(b'\0'):
        if record:
            meta, path = record.split(b'\t', 1)
            remote_tree[path.decode()] = meta.split()[2].decode()
    progress = publication / 'export-progress.json'
    source_manifest = gzip.open(history / 'copy-sources.jsonl.gz', 'wt', encoding='utf-8', compresslevel=1)

    def safe(path):
        parts = path.relative_to(project).parts
        return (not BLOCKED_DIRS.intersection(parts) and path.name not in SECRET_NAMES
                and not path.name.startswith('.env') and not path.is_symlink())

    def copy(path, relative, category, respect_frozen_archive=False):
        if not path.is_file() or not safe(path):
            return
        if relative.as_posix() in inventory:
            return
        data = path.read_bytes()
        # Recognized credential examples are still redacted in publication;
        # the exact original file digest and redaction count remain auditable.
        original_sha = sha(data)
        substitutions = 0
        if original_sha not in BENIGN_FIXTURE_SHA256:
            for pattern in SECRET_PATTERNS:
                data, n = pattern.subn(b'REDACTED_CREDENTIAL', data)
                substitutions += n
        if substitutions:
            redactions.append({'source': str(path.relative_to(project)), 'count': substitutions})
        oid = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        remote_oid = remote_tree.get(relative.as_posix())
        if respect_frozen_archive and remote_oid:
            if remote_oid == oid:
                return
            relative = history_rel / 'local-iteration-records' / relative.relative_to('iteration-records')
        encoding = 'identity'
        if len(data) > 24 * 2**20:
            data = gzip.compress(data, compresslevel=1, mtime=0)
            relative = Path(str(relative) + '.gz')
            encoding = 'gzip'
        if len(data) > 90 * 2**20:
            excluded.append({'source': str(path.relative_to(project)), 'reason': 'file still exceeds safe Git size after compression'})
            return
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        destination.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
        entry = {'source': path.relative_to(project).as_posix(), 'path': relative.as_posix(), 'category': category,
                 'original_sha256': original_sha, 'sha256': sha(data), 'bytes': len(data),
                 'encoding': encoding, 'credential_redactions': substitutions,
                 'git_blob_oid': hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()}
        inventory[relative.as_posix()] = entry
        source_manifest.write(json.dumps(entry, ensure_ascii=False) + '\n')
        counts[category] += 1
        aggregates[category] += len(data)
        if len(inventory) % 3000 == 0:
            write(progress, {'phase': 'staging', 'files': len(inventory), 'bytes': sum(aggregates.values()),
                             'category': category, 'updated_unix': time.time()})

    # Existing modified implementation files are authorized by the full-workspace request.
    modified = git('diff', '--name-only', '--diff-filter=ACMRT', 'HEAD').decode().splitlines()
    overlaps = git('diff', '--name-only', 'HEAD', parent, '--', *modified).decode().splitlines() if modified else []
    if overlaps:
        raise ValueError('Remote changes overlap current implementation edits: ' + str(overlaps))
    for name in modified:
        copy(project / name, Path(name), 'modified-project')
    candidates = [x.decode() for x in git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0') if x]
    for name in candidates:
        path = project / name
        parts = Path(name).parts
        if parts[0] in ['artifacts', 'iteration-records', 'UniRank', 'benchmark_res']:
            continue
        if parts[:2] == ('portable', 'runtime') or not path.is_file():
            continue
        if path.suffix not in TEXT and path.name not in NAMES and path.suffix not in ['.svg', '.png', '.jpg', '.html']:
            continue
        copy(path, Path(name), 'project-source')
    # Embed code, not Git links, for the two local nested checkouts.
    for dirname in ['UniRank', 'benchmark_res']:
        folder = project / dirname
        for args in [('ls-files', '-z'), ('ls-files', '--others', '--exclude-standard', '-z')]:
            for raw in git(*args, cwd=folder).split(b'\0'):
                if not raw:
                    continue
                path = folder / raw.decode()
                if path.is_file() and safe(path) and (path.suffix in TEXT or path.name in NAMES):
                    copy(path, Path(dirname) / raw.decode(), 'nested-project-source')
        nested.append({'directory': dirname, 'commit': git('rev-parse', 'HEAD', cwd=folder).decode().strip(),
                       'remote': git('remote', 'get-url', 'origin', cwd=folder).decode().strip(),
                       'working_changes': git('status', '--short', cwd=folder).decode().splitlines()})
    # Existing publication directories remain immutable; keep a distinct copy if local bytes differ.
    for path in sorted((project / 'iteration-records').rglob('*')):
        if path.is_file() and safe(path) and (path.suffix in TEXT or path.name in NAMES):
            copy(path, path.relative_to(project), 'local-iteration-records', True)
    for folder, directories, names in os.walk(project / 'artifacts', followlinks=False):
        current = Path(folder)
        relative = current.relative_to(project / 'artifacts')
        directories[:] = [d for d in directories if d not in BLOCKED_DIRS
                          and not (relative == Path('.') and d in ARTIFACT_EXCLUDE)
                          and not (current / d).is_symlink()]
        for name in names:
            path = current / name
            if not safe(path) or (path.suffix not in TEXT and path.name not in NAMES):
                continue
            parts = path.relative_to(project / 'artifacts').parts
            if DATA_DIRS.intersection(parts) and path.suffix not in {'.py', '.sh', '.md', '.yaml', '.yml', '.toml'} and name not in DATA_METADATA:
                continue
            if path.stat().st_size == 0 and path.suffix == '.log':
                continue
            copy(path, history_rel / 'artifacts' / path.relative_to(project / 'artifacts'), 'history/' + parts[0])
    source_manifest.close()
    write(history / 'nested-sources.json', nested)
    write(history / 'exclusions.json', {'credential_redactions': redactions, 'other_exclusions': excluded,
          'datasets_weights_environments_archives_excluded': True,
          'large_text_policy': 'text over24MiB is preserved losslessly as .gz; redactions are separately recorded'})
    write(history / 'summary.json', {'snapshot_finished_utc': datetime.now(timezone.utc).isoformat(),
          'parent_commit': parent, 'files_by_category': dict(counts), 'bytes_by_category': dict(aggregates),
          'file_count': len(inventory), 'credential_redactions': sum(x['count'] for x in redactions),
          'full_datasets_included': False, 'weights_included': False, 'environments_included': False})
    (history / 'README.md').write_text('''# 工作区源码与历史实验记录快照

按用户要求直接提交 Git；不生成项目 tar，也不上传数据集、模型权重或运行环境。
项目源码位于仓库原目录，UniRank 与 benchmark_res 保留当前源码及上游版本来源。
artifacts/ 保存历史实验的代码、配置、假设、上下文、结果、失败/重试日志与运行凭据；原始绝对路径用于来源审计。
已发布的迭代目录保持原样，本地不同版本放在 local-iteration-records/。

- summary.json：各历史实验的文件数、体积和范围。
- copy-sources.jsonl.gz：原路径、目标路径、SHA256、原始摘要、编码、Git blob 和必要的凭据脱敏记录。
- nested-sources.json：嵌入源码的两个子仓库及其工作区改动。
- exclusions.json：排除范围与凭据脱敏统计。

超过24MiB的诊断文本以逐文件gzip完整保留，读取时用gzip解压；这不是数据集压缩包。
重建实验需另行获取数据和依赖，并按各协议的固定 revision/切分/评分口径重建本地路径。
历史失败与源码仅作为审计材料，不承诺每个失败候选可运行或有效。
''')
    # The manifest covers the staged artifact, not live source files that may keep receiving telemetry.
    manifest = history / 'manifest.jsonl.gz'
    with gzip.open(manifest, 'wt', encoding='utf-8', compresslevel=1) as stream:
        for path in sorted(stage.rglob('*')):
            if path.is_file() and path != manifest:
                data = path.read_bytes()
                stream.write(json.dumps({'path': path.relative_to(stage).as_posix(), 'sha256': sha(data), 'bytes': len(data),
                    'git_blob_oid': hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()}, ensure_ascii=False) + '\n')
    write(publication / 'export-plan.json', {'stage': str(stage), 'history_directory': history_rel.as_posix(),
          'parent': parent, 'file_count': len(inventory), 'staged_bytes': sum(aggregates.values()),
          'modified_project_paths': modified, 'credential_redactions': redactions,
          'manifest': str(manifest), 'completed_unix': time.time()})
    write(progress, {'phase': 'complete', 'files': len(inventory), 'bytes': sum(aggregates.values()), 'updated_unix': time.time()})
    print(json.dumps({'publication': str(publication), 'files': len(inventory), 'MiB': sum(aggregates.values()) / 2**20,
                      'credential_redaction_files': len(redactions)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--parent', required=True)
    args = parser.parse_args()
    export(args.project, args.publication, args.parent)
