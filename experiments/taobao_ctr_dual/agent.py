"""Generic file-producing researcher behind chroot and an unprivileged UID."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

from common import read,write,sha
from knowledge import stage_resources,REVISION

PROJECT=Path(os.environ.get('REC_CTR_PROJECT',Path(__file__).resolve().parents[2]))
DESIGN=Path(__file__).resolve().parent/'task-design'
if not DESIGN.exists():DESIGN=PROJECT/'task_designs/autonomous_research_v1'
PRIVATE=Path('/root/.local/share/taobao-ctr-dual-runtime')
CODEX=PROJECT/'portable/runtime/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex'


def prepare_template():
    template=PRIVATE/'template'
    sys.path.insert(0,str(PROJECT))
    from rec_harness.isolation import prepare,copy_readonly
    if (template/'jail-manifest.json').is_file():
        copy_readonly(CODEX.with_name('codex-code-mode-host'),template,'/root/.local/bin/codex-code-mode-host')
        return template
    binaries=['/bin/bash','/bin/sh','/bin/cat','/bin/ls','/bin/sed','/bin/cp','/bin/mv','/bin/mkdir','/bin/rm',
              '/usr/bin/python3','/usr/bin/env','/usr/bin/head','/usr/bin/tail','/usr/bin/wc','/usr/bin/find']
    paths=set(binaries)
    for path in binaries:
        info=subprocess.run(['ldd',path],capture_output=True,text=True)
        for word in info.stdout.split():
            if word.startswith('/') and Path(word).is_file():paths.add(word)
    paths.update(['/etc/resolv.conf','/etc/hosts','/etc/nsswitch.conf'])
    PRIVATE.mkdir(parents=True,exist_ok=True);PRIVATE.chmod(0o700)
    prepare(template,runtime_dirs=['/usr/lib/python3.10','/etc/ssl/certs'],runtime_files=sorted(paths),
            inputs={str(CODEX):'/root/.local/bin/codex',str(CODEX.with_name('codex-code-mode-host')):'/root/.local/bin/codex-code-mode-host'})
    return template


def run(directory,task,runtime,workspace,evidence,timeout,resources=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    template=prepare_template();identifier=uuid.uuid4().hex
    jail=PRIVATE/identifier;jail.mkdir();uid=300000+int(identifier[:6],16)
    for source in template.iterdir():
        if source.name in {'tmp','workspace','output','dev','jail-manifest.json'}:continue
        if source.is_dir():shutil.copytree(source,jail/source.name,symlinks=True,copy_function=os.link)
        elif source.is_file():os.link(source,jail/source.name)
    import stat
    (jail/'dev').mkdir()
    for name in ['null','zero','random','urandom']:
        info=os.stat('/dev/'+name);os.mknod(jail/'dev'/name,stat.S_IFCHR|0o666,info.st_rdev);(jail/'dev'/name).chmod(0o666)
    # A synthetic current-executable link satisfies the static CLI; no host proc is mounted.
    (jail/'proc/self').mkdir(parents=True,exist_ok=True)
    (jail/'proc/self/exe').symlink_to('/root/.local/bin/codex')
    for name in ['tmp','workspace','output','root/.codex']:
        path=jail/name;path.mkdir(parents=True,exist_ok=True);path.chmod(0o700);os.chown(path,uid,uid)
    # Config/auth copies are private runtime artifacts and never enter study exports.
    for name in ['auth.json','config.toml']:
        source=Path('/root/.codex')/name
        if source.exists():
            target=jail/'root/.codex'/name;shutil.copyfile(source,target);target.chmod(0o600);os.chown(target,uid,uid)
    public=jail/'workspace'
    generic=(DESIGN/'agent/instructions.txt').read_text()+'\nResolve each authorized reference by running python3 /workspace/tools.py resolve REF. Search available evidence with python3 /workspace/tools.py search TERMS. The resolved runtime contract declares the current phase and expected file outputs. Only inspect public workspace inputs and authorized sources. Credentials and runtime internals are not research inputs. Do not launch other agents, training or installations.\n'
    (public/'AGENTS.md').write_text(generic)
    registry={'task://t-001':'task.json','run://r-001':'runtime.json','workspace://w-001':'workspace.json'}
    manifest=stage_resources(public,registry,resources or [])
    for name,value in [('registry.json',registry),('task.json',task),('runtime.json',runtime),('workspace.json',workspace)]:
        write(public/name,value)
    (public/'evidence.jsonl').write_text(''.join(json.dumps(card,ensure_ascii=False)+'\n' for card in evidence))
    shutil.copyfile(Path(__file__).with_name('agent_tools.py'),public/'tools.py')
    shutil.copyfile(Path(__file__).with_name('knowledge.py'),public/'knowledge_engine.py')
    for source in directory.glob('*.py'):
        shutil.copyfile(source,public/source.name)
    for path in public.iterdir():
        if path.is_dir():path.chmod(0o555)
        elif path.name.endswith('.py') and path.name not in {'tools.py','knowledge_engine.py','baseline_reference.py'} and not path.name.startswith('history_'):
            os.chown(path,uid,uid);path.chmod(0o600)
        else:path.chmod(0o444)
    launch={'task_ref':'task://t-001','runtime_ref':'run://r-001','workspace_ref':'workspace://w-001'}
    write(directory/'startup.json',launch)
    write(directory/'task-export.json',task);write(directory/'runtime-export.json',runtime)
    write(directory/'workspace-export.json',workspace)
    write(directory/'knowledge-resource-manifest.json',manifest)
    write(directory/'resource-registry-export.json',registry)
    (directory/'supplied-evidence.jsonl').write_text((public/'evidence.jsonl').read_text())
    write(jail/'jail-manifest.json',{'uid':uid,'phase':runtime['phase'],'run_id':identifier})
    argv=['/root/.local/bin/codex','exec','--json','--ephemeral','--skip-git-repo-check',
          '--model','gpt-6.1-sol','-c','model_reasoning_effort="high"','-c','approval_policy="never"',
          '-c','features.multi_agent=false','-c','web_search="live"','--sandbox','danger-full-access',
          '--cd','/workspace','--output-last-message','/workspace/last-message.txt','-']
    command=[sys.executable,'-m','rec_harness.isolation','enter',str(jail),'--uid',str(uid),'--',*argv]
    start=time.monotonic();status='success';code=None
    with (directory/'codex.jsonl').open('w') as stream:
        process=subprocess.Popen(command,cwd=PROJECT,stdin=subprocess.PIPE,stdout=stream,stderr=subprocess.STDOUT,text=True,start_new_session=True)
        write(directory/'agent-process.json',{'pid':process.pid,'start_ticks':Path(f'/proc/{process.pid}/stat').read_text().split()[21]})
        try:
            process.communicate(json.dumps(launch),timeout=max(1,timeout));code=process.returncode
            if code:status='failed'
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait();status='timeout'
    usage=[]
    for line in (directory/'codex.jsonl').read_text().splitlines():
        try:
            event=json.loads(line)
            if event.get('type')=='turn.completed':usage.append(event.get('usage',{}))
        except ValueError:pass
    receipt={'status':status,'returncode':code,'wall_seconds':time.monotonic()-start,'model':'gpt-6.1-sol','reasoning_effort':'high',
             'usage':usage,'fixed_instructions_sha256':sha(public/'AGENTS.md'),
             'knowledge_revision':REVISION,'knowledge_resources':manifest,
             'isolation':'chroot, unique nonprivileged UID, synthetic current-executable link; no host proc or GPU devices','startup_fields':sorted(launch),'usd_cost':None}
    write(directory/'agent-receipt.json',receipt)
    if (public/'knowledge-access.jsonl').is_file():
        shutil.copyfile(public/'knowledge-access.jsonl',directory/'knowledge-access.jsonl')
    # Copy outputs by expected filename only; do not archive auth or unrelated jail state.
    for name in runtime['expected_files']:
        source=public/name
        if source.is_file() and not source.is_symlink():shutil.copyfile(source,directory/name)
    if runtime['phase']=='propose' and (public/'proposal.json').is_file():
        proposal=read(public/'proposal.json')
        for variant in proposal.get('variants',[]):
            name=variant['file']
            if Path(name).name!=name or not name.endswith('.py'):raise ValueError('Unsafe candidate filename')
            source=public/name
            if source.is_symlink() or not source.is_file():raise ValueError('Missing candidate source')
            shutil.copyfile(source,directory/name)
    if status!='success':raise RuntimeError('Researcher call '+status+'; inspect '+str(directory/'codex.jsonl'))
    for name in runtime['expected_files']:
        if not (directory/name).is_file():raise ValueError('Researcher omitted '+name)
    return receipt
