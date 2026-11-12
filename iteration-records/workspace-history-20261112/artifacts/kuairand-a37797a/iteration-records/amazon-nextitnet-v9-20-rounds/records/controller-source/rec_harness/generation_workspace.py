"""Retain stopped agent drafts independently of response/schema admission."""
import os
from pathlib import Path
import shutil
from .core import ContractError,file_hash,read_json,safe_file,write_json


def capture(workspace,directory):
    workspace=Path(workspace);directory=Path(directory)
    output=directory/'returned-workspace'
    if output.exists():raise ContractError('returned workspace already captured')
    output.mkdir()
    copied={};excluded=[]
    def copy(name):
        source=workspace/name
        if source.is_symlink() or not source.is_file():
            excluded.append({'path':name,'reason':'not a regular non-symlink file'});return
        source=safe_file(workspace,name)
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        copied[name]={'sha256':file_hash(target),'size':target.stat().st_size}
    response=workspace/'response.json'
    if response.exists() or response.is_symlink():copy('response.json')
    model=workspace/'model'
    if model.is_symlink():excluded.append({'path':'model','reason':'symlink directory'})
    elif model.exists():
        for parent,dirs,names in os.walk(model,followlinks=False):
            for child in list(dirs):
                path=Path(parent)/child
                if path.is_symlink():
                    excluded.append({'path':str(path.relative_to(workspace)),'reason':'symlink directory'})
                    dirs.remove(child)
            for name in sorted(names):copy(str((Path(parent)/name).relative_to(workspace)))
    write_json(output/'manifest.json',{'schema':'returned-agent-workspace-v1','files':copied,
        'excluded':excluded,'qualification':'Unadmitted output from a stopped role process; not approved implementation or successful research.'})


def archive_files(directory):
    directory=Path(directory);manifest=directory/'returned-workspace/manifest.json'
    if not manifest.exists():return []
    value=read_json(manifest)
    if value.get('schema')!='returned-agent-workspace-v1':raise ContractError('unknown returned workspace schema')
    result=['returned-workspace/manifest.json']
    for name,ref in value['files'].items():
        path=safe_file(manifest.parent,name)
        if file_hash(path)!=ref['sha256'] or path.stat().st_size!=ref['size']:
            raise ContractError('returned workspace file changed')
        result.append('returned-workspace/'+name)
    return result
