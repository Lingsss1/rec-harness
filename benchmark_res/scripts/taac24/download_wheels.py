"""Accelerate large public PyPI wheels with verified HTTP ranges, then install."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

ROOT = Path(sys.argv[1]).resolve()
REPO = Path(__file__).resolve().parents[2]
WHEELS = ROOT / 'wheels'
WHEELS.mkdir(exist_ok=True)
packages = [('torch','2.6.0')]
for requirement in json.loads((ROOT/'torch-requirements.json').read_text()):
    match = re.match(r'((?:nvidia-[a-z0-9-]+)|triton)\s*\(?==([a-z0-9.]+)', requirement)
    if match:
        packages.append(match.groups())

selected = []
for name, version in packages:
    with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=30) as response:
        info = json.load(response)
    choices = [f for f in info['urls'] if f['filename'].endswith('.whl') and 'x86_64' in f['filename'] and ('manylinux' in f['filename'] or 'linux_x86_64' in f['filename']) and ('cp310' in f['filename'] or 'py3-none' in f['filename'])]
    if not choices:
        raise ValueError('No compatible public wheel: '+name)
    chosen = choices[0]
    # Small dependencies already use the normal verified installer/cache.
    if chosen['size'] >= 30_000_000:
        selected.append(chosen)
(ROOT/'wheel-publisher-manifest.json').write_text(json.dumps(selected,indent=2))


def segment(job):
    entry, fd, first, last = job
    for attempt in range(5):
        try:
            request = urllib.request.Request(entry['url'],headers={'Range':f'bytes={first}-{last}'})
            with urllib.request.urlopen(request,timeout=40) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {first}-{last}/{entry["size"]}':
                    raise ValueError('Server did not honor exact HTTP range')
                position = first
                while True:
                    block = response.read(1<<20)
                    if not block:
                        break
                    if position+len(block)>last+1:
                        raise ValueError('Excess range bytes')
                    offset = 0
                    while offset<len(block):
                        offset += os.pwrite(fd,block[offset:],position+offset)
                    position += len(block)
                if position != last+1:
                    raise ValueError('Incomplete HTTP range')
            return
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def checksum(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4<<20),b''):
            h.update(block)
    return h.hexdigest()


jobs, opened = [], []
for entry in selected:
    path=WHEELS/entry['filename']
    if path.exists() and checksum(path)==entry['digests']['sha256']:
        continue
    temp=path.with_suffix('.partial')
    fd=os.open(temp,os.O_RDWR|os.O_CREAT|os.O_TRUNC,0o600)
    os.ftruncate(fd,entry['size'])
    opened.append((entry,fd,temp,path))
    chunk=8<<20
    jobs.extend((entry,fd,start,min(start+chunk,entry['size'])-1) for start in range(0,entry['size'],chunk))
print(json.dumps({'wheels':len(opened),'bytes':sum(e['size'] for e,_,_,_ in opened),'segments':len(jobs)}),flush=True)
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        done=0
        for future in concurrent.futures.as_completed([pool.submit(segment,j) for j in jobs]):
            future.result()
            done+=1
            if done%20==0 or done==len(jobs):
                print(json.dumps({'segments_done':done,'total':len(jobs)}),flush=True)
finally:
    for _,fd,_,_ in opened:
        os.close(fd)
for entry,_,temp,path in opened:
    if checksum(temp)!=entry['digests']['sha256']:
        raise ValueError('Public PyPI SHA256 mismatch: '+path.name)
    os.replace(temp,path)
    print('SHA256 verified '+path.name,flush=True)
command=['/root/miniconda3/bin/uv','pip','install','--python',str(REPO/'.venv/bin/python'),'--index-url','https://pypi.org/simple','--find-links',str(WHEELS),'-e',str(REPO)+'[taac,taac-semantic]','pytest','torch==2.6.0','transformers==4.57.6']
subprocess.run(command,check=True,cwd=REPO)
subprocess.run(['/root/miniconda3/bin/uv','pip','check','--python',str(REPO/'.venv/bin/python')],check=True,cwd=REPO)
(ROOT/'dependencies-ready.json').write_text(json.dumps({'ready':True,'python':str(REPO/'.venv/bin/python')}))
