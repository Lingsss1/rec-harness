"""Download only the pinned official UniRank Taobao release and verify its files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from huggingface_hub import HfApi


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def run(root):
    data=root/'data/Taobao_Action';data.mkdir(parents=True,exist_ok=True)
    write(root/'acquisition-status.json',{'phase':'listing_official_taobao_release','at':time.time()})
    saved=root/'dataset-source-manifest.json'
    if saved.exists():manifest=json.loads(saved.read_text());files=manifest['files']
    else:
        info=HfApi().dataset_info('salmon1802/Taobao',files_metadata=True)
        files=[{'path':f.rfilename,'bytes':f.size,'lfs_sha256':f.lfs.sha256 if f.lfs else None} for f in info.siblings]
        manifest={'repo_id':'salmon1802/Taobao','revision':info.sha,'files':files,
                  'total_bytes':sum(f['bytes'] or 0 for f in files),'started_unix':time.time()}
        write(saved,manifest)
    base='https://huggingface.co/datasets/salmon1802/Taobao'
    def download(row,url,headers=None):
        path=data/row['path'];path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.stat().st_size==row['bytes']:return
        partial=path.with_suffix(path.suffix+'.partial')
        for attempt in range(5):
            try:
                with requests.get(url,headers=headers or {},timeout=(30,120),stream=True) as response:
                    response.raise_for_status()
                    with partial.open('wb') as stream:
                        for chunk in response.iter_content(1<<20):stream.write(chunk)
                if partial.stat().st_size!=row['bytes']:raise ValueError('Incomplete download: '+row['path'])
                partial.replace(path);return
            except (requests.RequestException,ValueError):
                if attempt==4:raise
                time.sleep(min(30,2**(attempt+1)))
    for row in files:
        if not row['lfs_sha256']:download(row,base+'/raw/'+manifest['revision']+'/'+row['path'])
    lfs=[row for row in files if row['lfs_sha256']]
    for offset in range(0,len(lfs),24):
        group=lfs[offset:offset+24]
        response=requests.post(base+'.git/info/lfs/objects/batch',json={'operation':'download',
            'transfers':['basic'],'objects':[{'oid':r['lfs_sha256'],'size':r['bytes']} for r in group]},
            headers={'Accept':'application/vnd.git-lfs+json'},timeout=60)
        response.raise_for_status();objects={obj['oid']:obj for obj in response.json()['objects']}
        def one(row):
            action=objects[row['lfs_sha256']]['actions']['download']
            download(row,action['href'],action.get('header'))
        with ThreadPoolExecutor(4) as pool:list(pool.map(one,group))
        write(root/'acquisition-status.json',{'phase':'downloading_official_lfs','at':time.time(),
            'completed_lfs_files':min(offset+24,len(lfs)),'total_lfs_files':len(lfs)})
        print('Downloaded',min(offset+24,len(lfs)),'/',len(lfs),'official LFS files',flush=True)
    for row in files:
        path=data/row['path']
        if row['bytes'] is not None and path.stat().st_size!=row['bytes']:raise ValueError('Size mismatch: '+row['path'])
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(16<<20),b''):digest.update(chunk)
        row['local_sha256']=digest.hexdigest()
        if row['lfs_sha256'] and row['local_sha256']!=row['lfs_sha256']:raise ValueError('Digest mismatch: '+row['path'])
    manifest['verified_unix']=time.time();write(root/'dataset-source-manifest.json',manifest)
    write(root/'acquisition-status.json',{'phase':'downloaded_and_verified','at':time.time(),
        'files':len(files),'bytes':manifest['total_bytes'],'revision':manifest['revision']})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    try:run(args.root)
    except BaseException as error:
        write(args.root/'acquisition-status.json',{'phase':'failed','error':repr(error),'at':time.time()})
        raise
