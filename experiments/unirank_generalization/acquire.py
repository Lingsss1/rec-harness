"""Resumable pinned official LFS downloads; no dataset parsing or test feedback."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import threading
import time
from urllib.request import Request,urlopen
import uuid


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def acquire(folder,workers=8):
    manifest=json.loads((folder/'dataset-source-manifest.json').read_text());rows=manifest['files']
    data=folder/'data';data.mkdir(exist_ok=True);lock=threading.Lock();completed=0
    done_bytes=0;started=time.time()
    repo='https://huggingface.co/datasets/'+manifest['repo_id']
    def download(row,url,headers=None):
        nonlocal completed,done_bytes
        target=data/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
        expected=row.get('lfs_sha256')
        if target.exists() and target.stat().st_size==row['bytes'] and (not expected or digest(target)==expected):
            pass
        else:
            partial=target.with_name(target.name+'.partial')
            for attempt in range(6):
                try:
                    offset=partial.stat().st_size if partial.exists() else 0
                    request_headers=dict(headers or {})
                    if offset:request_headers['Range']='bytes='+str(offset)+'-'
                    with urlopen(Request(url,headers=request_headers),timeout=90) as response:
                        append=offset>0 and response.status==206
                        with partial.open('ab' if append else 'wb') as stream:
                            for block in iter(lambda:response.read(1<<20),b''):stream.write(block)
                    if partial.stat().st_size!=row['bytes']:raise ValueError('Incomplete download')
                    if expected and digest(partial)!=expected:
                        partial.unlink();raise ValueError('Checksum mismatch')
                    partial.replace(target);break
                except Exception:
                    if attempt==5:raise
                    time.sleep(min(30,2**(attempt+1)))
        row['local_sha256']=digest(target)
        with lock:
            completed+=1;done_bytes+=row['bytes'] or 0
            write(folder/'acquisition-status.json',{'phase':'downloading','started_unix':started,'at_unix':time.time(),
                'completed_files':completed,'total_files':len(rows),'completed_bytes':done_bytes,
                'total_bytes':manifest['total_bytes'],'revision':manifest['revision']})
    for row in rows:
        if not row['lfs_sha256']:download(row,repo+'/raw/'+manifest['revision']+'/'+row['path'])
    lfs=[row for row in rows if row['lfs_sha256']]
    for offset in range(0,len(lfs),32):
        group=lfs[offset:offset+32]
        body=json.dumps({'operation':'download','transfers':['basic'],
            'objects':[{'oid':r['lfs_sha256'],'size':r['bytes']} for r in group]}).encode()
        objects=None
        for attempt in range(5):
            try:
                with urlopen(Request(repo+'.git/info/lfs/objects/batch',data=body,
                    headers={'Accept':'application/vnd.git-lfs+json','Content-Type':'application/vnd.git-lfs+json'}),timeout=60) as response:
                    objects={obj['oid']:obj for obj in json.loads(response.read())['objects']}
                break
            except Exception:
                if attempt==4:raise
                time.sleep(min(30,2**(attempt+1)))
        def one(row):
            obj=objects[row['lfs_sha256']]
            if 'error' in obj:raise ValueError('Official LFS object unavailable: '+row['path'])
            action=obj['actions']['download'];download(row,action['href'],action.get('header'))
        with ThreadPoolExecutor(workers) as pool:
            for future in as_completed([pool.submit(one,row) for row in group]):future.result()
    manifest['verified_unix']=manifest.get('verified_unix',time.time())
    write(folder/'dataset-source-manifest.json',manifest)
    write(folder/'acquisition-status.json',{'phase':'downloaded_and_verified','at_unix':time.time(),
        'completed_files':len(rows),'total_files':len(rows),'bytes':manifest['total_bytes'],'revision':manifest['revision']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8);args=parser.parse_args()
    try:acquire(args.folder.resolve(),args.workers)
    except BaseException as error:
        # Signed URLs are deliberately excluded from durable public errors.
        write(args.folder/'acquisition-status.json',{'phase':'failed','at_unix':time.time(),
            'error_type':type(error).__name__,'message':'Official release download failed; inspect file progress and rerun to resume.'})
        raise SystemExit('Dataset download failed: '+type(error).__name__)
