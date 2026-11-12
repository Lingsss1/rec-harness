"""Acquire the missing raw log from a versioned public mirror, with receipts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import zipfile


REF = 'zerotrustai/taobao-ad-display-click'
VERSION = 1
BASE = 'https://www.kaggle.com/api/v1/datasets/'
EXPECTED_ARCHIVE_BYTES = 5165473616
EXPECTED_LOG_BYTES = 23728773580


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(8 << 20), b''):
            value.update(data)
    return value.hexdigest()


def main(root):
    root.mkdir(parents=True, exist_ok=True)
    previous=json.loads((root/'acquisition.json').read_text()) if (root/'acquisition.json').exists() else {}
    state = {'phase': 'acquiring', 'pid': os.getpid(), 'started_unix': previous.get('started_unix',time.time()), 'attempt_started_unix':time.time(),
             'source': REF, 'version': VERSION, 'official_source': 'https://tianchi.aliyun.com/dataset/56',
             'provenance': 'third-party public mirror; no official behavior-log checksum was located'}
    for endpoint, name in [('view/', 'mirror-metadata.json'), ('list/', 'mirror-files.json')]:
        with urllib.request.urlopen(BASE + endpoint + REF, timeout=60) as response:
            value = json.load(response)
        write(root / name, value)
        if endpoint == 'view/' and value['currentVersionNumber'] != VERSION:
            raise ValueError('Unexpected mirror version')
    write(root / 'acquisition.json', state)
    archive = root / 'raw-mirror-v1.zip'
    partial = root / 'raw-mirror-v1.zip.partial'
    if not archive.exists():
        url = BASE + 'download/' + REF + '?datasetVersionNumber=' + str(VERSION)
        start = time.monotonic()
        failures=0
        while not partial.exists() or partial.stat().st_size<EXPECTED_ARCHIVE_BYTES:
            if time.time()>=state['started_unix']+86400:raise TimeoutError('Acquisition hard budget reached')
            offset=partial.stat().st_size if partial.exists() else 0
            end=min(EXPECTED_ARCHIVE_BYTES-1,offset+(256<<20)-1)
            request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-{end}'})
            try:
                with urllib.request.urlopen(request,timeout=120) as response:
                    expected=f'bytes {offset}-{end}/{EXPECTED_ARCHIVE_BYTES}'
                    if response.status!=206 or response.headers.get('Content-Range')!=expected:
                        raise ValueError('Source did not honor the exact inspected byte range')
                    with partial.open('ab') as target:
                        while data:=response.read(8<<20):target.write(data)
                if partial.stat().st_size!=end+1:raise ValueError('Truncated range; retain verified prefix for resume')
                failures=0
            except Exception as error:
                failures+=1
                with (root/'download-failures.jsonl').open('a') as stream:stream.write(json.dumps({'at':time.time(),'offset':offset,'end':end,'error':repr(error)})+'\n')
                if failures>=5:raise
                time.sleep(min(20,failures*3))
            state.update(downloaded_bytes=partial.stat().st_size,expected_bytes=EXPECTED_ARCHIVE_BYTES,elapsed_seconds=time.monotonic()-start)
            write(root/'acquisition.json',state)
            print(json.dumps({'phase':'download','bytes':state['downloaded_bytes'],'elapsed':state['elapsed_seconds']}),flush=True)
        if partial.stat().st_size != EXPECTED_ARCHIVE_BYTES:
            raise ValueError('Truncated archive')
        partial.replace(archive)
    state.update(phase='extracting', archive_sha256=sha(archive))
    write(root / 'acquisition.json', state)
    raw = root / 'raw'
    raw.mkdir(exist_ok=True)
    local = Path('/home/liuzhishan/ss/modformer/raw/taobao')
    existing = json.loads((local / 'source.json').read_text())
    with zipfile.ZipFile(archive) as bundle:
        # Three matching independent mirror hashes bind the log to the same raw release.
        for entry in existing['files']:
            digest=hashlib.sha256()
            with bundle.open(entry['name']+'/'+entry['name']) as source:
                while data:=source.read(8<<20):digest.update(data)
            if digest.hexdigest()!=entry['expected_sha256']:raise ValueError('Mirrors disagree on '+entry['name'])
    for entry in existing['files']:
        source = local / entry['name']
        if source.stat().st_size != entry['expected_bytes'] or sha(source) != entry['expected_sha256']:
            raise ValueError('Local raw-table integrity failure: ' + entry['name'])
        link = raw / entry['name']
        if not link.exists():
            link.symlink_to(source)
    log = raw / 'behavior_log.csv'
    if not log.exists():
        partial_log = raw / 'behavior_log.csv.partial'
        with zipfile.ZipFile(archive) as bundle:
            member = bundle.getinfo('behavior_log.csv/behavior_log.csv')
            if member.file_size != EXPECTED_LOG_BYTES:
                raise ValueError('Unexpected behavior-log size')
            with bundle.open(member) as source, partial_log.open('xb') as target:
                digest = hashlib.sha256()
                while data := source.read(16 << 20):
                    target.write(data)
                    digest.update(data)
        if partial_log.stat().st_size != EXPECTED_LOG_BYTES:
            raise ValueError('Incomplete behavior log')
        partial_log.replace(log)
        log_sha = digest.hexdigest()
    else:
        log_sha = sha(log)
    state.update(phase='complete', completed_at=datetime.now(timezone.utc).isoformat(),
                 behavior_log={'bytes': log.stat().st_size, 'sha256': log_sha},
                 reused_tables=existing, raw_root=str(raw.resolve()))
    write(root / 'acquisition.json', state)
    print(json.dumps({'phase': 'complete', 'raw_root': str(raw.resolve())}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    try:
        main(args.root.resolve())
    except BaseException as error:
        write(args.root.resolve() / 'acquisition-error.json', {'error': repr(error), 'at': time.time()})
        raise
