"""Resumable range download; only the official full-archive MD5 publishes data."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

from .core import ContractError, read_json, write_json
from .kuairand_baseline import lock

URL = 'https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz'
MD5 = '3e3c799a24e2d23a4d2c757fbf9adf59'


def download(root, workers=8, chunk_bytes=32*1024*1024):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    target = root/'KuaiRand-27K.tar.gz'
    with lock(root/'.range-download.lock'):
        if target.exists():
            h = hashlib.md5()
            with target.open('rb') as f:
                for data in iter(lambda: f.read(8*1024*1024), b''): h.update(data)
            if h.hexdigest() != MD5: raise ContractError('existing official archive has wrong MD5')
            return target
        headers = {'User-Agent': 'rec-harness-kuairand/1.0', 'Accept-Encoding': 'identity', 'Range': 'bytes=0-0'}
        with urlopen(Request(URL, headers=headers), timeout=60) as response:
            match = re.fullmatch(r'bytes 0-0/(\d+)', response.headers.get('Content-Range', ''))
            if response.status != 206 or not match:
                raise ContractError('server does not support exact byte ranges; serial official downloader remains available')
            total = int(match[1]); response.read(1)
        chunks = root/'ranges'; chunks.mkdir(exist_ok=True)
        identity = {'url': URL, 'md5': MD5, 'total_bytes': total, 'chunk_bytes': chunk_bytes}
        manifest = chunks/'manifest.json'
        if manifest.exists() and read_json(manifest) != identity: raise ContractError('range manifest changed')
        write_json(manifest, identity)
        def one(index):
            start = index*chunk_bytes; end = min(total, start+chunk_bytes)-1
            out = chunks/('%05d.part' % index); expected = end-start+1
            for attempt in range(5):
                have = out.stat().st_size if out.exists() else 0
                if have == expected: return
                if have > expected: raise ContractError('oversized download segment')
                try:
                    current = start+have
                    request = Request(URL, headers={**headers, 'Range': 'bytes=%d-%d' % (current, end)})
                    with urlopen(request, timeout=60) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != 'bytes %d-%d/%d' % (current, end, total):
                            raise ContractError('server returned mismatched segment range')
                        with out.open('ab') as f:
                            while True:
                                block = response.read(min(1024*1024, expected-have))
                                if not block: break
                                f.write(block); have += len(block)
                                if have == expected: break
                    if have != expected: raise OSError('segment response interrupted')
                    return
                except OSError:
                    if attempt == 4: raise
                    time.sleep(min(2**attempt, 16))
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(one, range((total+chunk_bytes-1)//chunk_bytes)))
        pending = root/'KuaiRand-27K.tar.gz.assembling'
        md5, sha = hashlib.md5(), hashlib.sha256()
        with pending.open('wb') as out:
            for index in range((total+chunk_bytes-1)//chunk_bytes):
                with (chunks/('%05d.part' % index)).open('rb') as f:
                    for block in iter(lambda: f.read(8*1024*1024), b''):
                        out.write(block); md5.update(block); sha.update(block)
        if md5.hexdigest() != MD5:
            pending.rename(root/('checksum-failed-'+str(time.time_ns())+'.tar.gz'))
            raise ContractError('parallel archive failed official MD5; no data published')
        os.replace(pending, target)
        write_json(root/'range-receipt.json', {**identity, 'sha256': sha.hexdigest(), 'verified_md5': MD5})
        return target
