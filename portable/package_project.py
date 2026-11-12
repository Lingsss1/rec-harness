"""Create a relocatable task snapshot with parallel, verified gzip members."""
import concurrent.futures
from collections import deque
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

source = Path(__file__).resolve().parents[1]
out = source/'artifacts/exports'
out.mkdir(parents=True, exist_ok=True)
stamp = time.strftime('%Y%m%d-%H%M%S', time.gmtime())
archive = out/f'rec-harness-{stamp}.tar.gz'
partial = archive.with_suffix(archive.suffix+'.part')
status = out/'pack-status.json'
index = archive.with_suffix('.members.txt')
errors = archive.with_suffix('.tar.log')
excluded = ['rec-harness/artifacts/exports', '*/.codex', '*/.aws', '*/.ssh', '*/.agents',
            '*/__pycache__', '*/.pytest_cache', '*/_trial_*', '*.pyc', '*.lock', '*.sock',
            '*/.env', '*/.env.*', '*/auth.json', '*/credentials.json', '*/credentials',
            '*/id_rsa', '*/id_ed25519']
command = ['tar','-cvf','-', '-C',str(source.parent),'--index-file='+str(index)]
command += ['--exclude='+pattern for pattern in excluded]
command += [source.name]
affinity = set(range(32,40)) & os.sched_getaffinity(0)
if affinity:
    os.sched_setaffinity(0, affinity)
started = time.time()
raw_size = compressed_size = members = 0
hash_value = hashlib.sha256()
last_report = 0
tail = b''


def report(state, **extra):
    global last_report
    value = {'status':state,'archive':str(archive),'started_at':started,'updated_at':time.time(),
             'source_bytes_processed':raw_size,'archive_bytes_written':compressed_size,
             'gzip_members':members,'cpu_affinity':sorted(os.sched_getaffinity(0)), **extra}
    temporary = status.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(status)
    print(json.dumps(value),flush=True)
    last_report = time.time()


def compress(data):
    packed = gzip.compress(data, compresslevel=1, mtime=0)
    if gzip.decompress(packed) != data:
        raise RuntimeError('Gzip member verification failed')
    return packed, len(data), data[-1024:]


report('packing')
with errors.open('w') as err, partial.open('xb') as dest:
    process = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=err)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            pending = deque()
            while True:
                data = process.stdout.read(8*1024*1024)
                if data:
                    pending.append(pool.submit(compress,data))
                if len(pending)>=12 or not data:
                    while pending and (len(pending)>=12 or not data):
                        packed, count, last = pending.popleft().result()
                        dest.write(packed)
                        hash_value.update(packed)
                        raw_size += count
                        compressed_size += len(packed)
                        members += 1
                        tail = (tail+last)[-1024:]
                        if time.time()-last_report>=10:
                            report('packing')
                if not data:
                    break
        code = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise
if code not in (0,1):
    report('failed',tar_exit_code=code,log=str(errors))
    raise RuntimeError('Tar failed; partial archive retained for inspection: '+str(errors))
if tail != b'\0'*1024:
    raise RuntimeError('Tar end-of-archive blocks missing')
partial.replace(archive)
checksum = archive.with_suffix(archive.suffix+'.sha256')
checksum.write_text(hash_value.hexdigest()+'  '+archive.name+'\n')
report('complete',sha256=hash_value.hexdigest(),checksum=str(checksum),member_index=str(index),
       tar_log=str(errors),tar_exit_code=code,verified='every gzip member decompressed and compared before writing; tar trailer verified',
       snapshot='live experiment logs/checkpoints are a point-in-time copy; running jobs were not stopped')
