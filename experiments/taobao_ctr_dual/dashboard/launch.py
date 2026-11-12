"""Create only a dashboard session; never move or restart experiment processes."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from collect import process_alive

HERE=Path(__file__).resolve().parent
SOCKET='ctr-research'
TMUX_BIN='/usr/bin/tmux'


def tmux_environment():
    # The dashboard server uses the distro binary. Conda's tmux/libtinfo may be
    # incompatible even when the client error misleadingly mentions the TTY.
    return {key:value for key,value in os.environ.items() if key not in {'LD_LIBRARY_PATH','LD_PRELOAD'}}


def attach_command(name):
    return shlex.join(['/usr/bin/env','-u','LD_LIBRARY_PATH','-u','LD_PRELOAD',TMUX_BIN,
                       '-L',SOCKET,'attach-session','-t',name])


def tmux(*args,check=True):
    result=subprocess.run([TMUX_BIN,'-L',SOCKET,*args],capture_output=True,text=True,env=tmux_environment())
    if check and result.returncode:raise RuntimeError(result.stderr.strip() or 'tmux command failed')
    return result.stdout.strip()


def launch(root,name):
    global SOCKET
    SOCKET=name
    output=root/'dashboard';output.mkdir(exist_ok=True)
    sampler=output/'sampler.json'
    previous=json.loads(sampler.read_text()) if sampler.exists() else {}
    if not process_alive(previous):
        with (output/'sampler.log').open('ab') as log:
            subprocess.Popen([sys.executable,'-u',str(HERE/'app.py'),'sample','--root',str(root)],
                             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    deadline=time.time()+20
    while not (output/'snapshot.json').exists():
        if time.time()>deadline:raise RuntimeError('Dashboard sampler not ready; inspect dashboard/sampler.log')
        time.sleep(0.2)
    existing=subprocess.run([TMUX_BIN,'-L',SOCKET,'has-session','-t','='+name],capture_output=True,env=tmux_environment()).returncode==0
    if existing:
        bound=tmux('show-options','-t',name,'-v','@ctr_dashboard_root')
        if bound!=str(root):raise ValueError('Session name belongs to another workspace; choose --session')
        print(json.dumps({'session':name,'status':'already_running','attach':attach_command(name)},ensure_ascii=False));return
    def command(view):
        return 'exec '+shlex.join([sys.executable,'-u',str(HERE/'app.py'),'view','--root',str(root),'--view',view])
    created=tmux('new-session','-d','-s',name,'-n','总览','-x','200','-y','60','-c',str(root),'-P','-F','#{session_id}\t#{window_id}\t#{pane_id}',command('overview'))
    session,window,overview=created.split('\t')
    tmux('set-option','-t',session,'@ctr_dashboard_root',str(root))
    state=json.loads((output/'snapshot.json').read_text())
    scope='验证集 · 测试比较受影响' if state['campaign'].get('recovery') else '验证集 · 只读'
    for option,value in [('mouse','on'),('status-style','bg=colour234,fg=colour252'),('status-left',f' #[bold,fg=cyan]{name} '),
                         ('status-right','#[fg=yellow]'+scope+'#[default] | C-b n 切换 · C-b z 放大 '),('status-left-length','30'),('status-right-length','100')]:
        tmux('set-option','-t',session,option,value)
    tmux('set-window-option','-t',window,'remain-on-exit','on')
    tmux('set-window-option','-t',window,'pane-border-status','top')
    tmux('set-window-option','-t',window,'pane-border-format',' #[fg=colour111]#{pane_title} ')
    resources=tmux('split-window','-h','-p','48','-t',overview,'-P','-F','#{pane_id}',command('resources'))
    experiments=tmux('split-window','-v','-p','50','-t',overview,'-P','-F','#{pane_id}',command('experiments'))
    events=tmux('split-window','-v','-p','50','-t',resources,'-P','-F','#{pane_id}',command('events'))
    panes={overview:'研究总览',resources:'资源 / LLM',experiments:'实验队列',events:'证据 / 事件'}
    def new_window(title,view):
        response=tmux('new-window','-d','-t',session,'-n',title,'-c',str(root),'-P','-F','#{window_id}\t#{pane_id}',command(view))
        win,pane=response.split('\t')
        tmux('set-window-option','-t',win,'remain-on-exit','on')
        return win,pane
    _,auto=new_window('研究','research-auto')
    enhanced=tmux('split-window','-h','-p','50','-t',auto,'-P','-F','#{pane_id}',command('research-enhanced'))
    panes.update({auto:'AUTO 当前假设 / 归因',enhanced:'强化版 当前假设 / 归因'})
    _,cost=new_window('成本','costs');panes[cost]='效果—成本前沿'
    _,slices_auto=new_window('切片','slices-auto')
    slices_enhanced=tmux('split-window','-h','-p','50','-t',slices_auto,'-P','-F','#{pane_id}',command('slices-enhanced'))
    panes.update({slices_auto:'AUTO 验证切片',slices_enhanced:'强化版 验证切片'})
    for pane,title in panes.items():tmux('select-pane','-t',pane,'-T',title)
    tmux('select-window','-t',window);tmux('select-pane','-t',overview)
    manifest={'session':name,'socket':SOCKET,'tmux_binary':TMUX_BIN,'session_id':session,'created_unix':time.time(),'panes':panes,'attach':attach_command(name),
              'read_only':True,'test_scores_displayed':False,'windows':['总览','研究','成本','切片']}
    (output/'tmux.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);parser.add_argument('--session',default='ctr-research')
    args=parser.parse_args();launch(args.root.resolve(),args.session)
