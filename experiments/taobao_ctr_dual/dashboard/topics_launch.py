"""Point the existing dashboard socket at a new topic campaign; preserve old views."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from collect import process_alive
import launch as terminal

HERE=Path(__file__).resolve().parent


def launch(root,name='ctr-research'):
    terminal.SOCKET=name
    output=root/'dashboard';output.mkdir(exist_ok=True)
    previous=json.loads((output/'sampler.json').read_text()) if (output/'sampler.json').exists() else {}
    if not process_alive(previous):
        with (output/'sampler.log').open('ab') as log:
            subprocess.Popen([sys.executable,'-u',str(HERE/'topics_app.py'),'sample','--root',str(root)],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    deadline=time.time()+20
    while not (output/'snapshot.json').exists():
        if time.time()>deadline:raise RuntimeError('Topic sampler not ready: inspect dashboard/sampler.log')
        time.sleep(.2)
    tmux=terminal.tmux;archived=None;clients=[]
    existing=subprocess.run([terminal.TMUX_BIN,'-L',name,'has-session','-t','='+name],
        capture_output=True,env=terminal.tmux_environment()).returncode==0
    if existing:
        bound=tmux('show-options','-t',name,'-v','@ctr_dashboard_root')
        if bound==str(root):
            print(json.dumps({'session':name,'status':'already_running','attach':terminal.attach_command(name)}));return
        clients=[line.split('\t')[0] for line in tmux('list-clients','-F','#{client_name}\t#{session_name}').splitlines()
                 if line.endswith('\t'+name)]
        archived='ctr-history-'+datetime.now(timezone.utc).strftime('%m%d-%H%M%S')
        tmux('rename-session','-t','='+name,archived)
    def command(view):return 'exec '+shlex.join([sys.executable,'-u',str(HERE/'topics_app.py'),
        'view','--root',str(root),'--view',view])
    session,window,overview=tmux('new-session','-d','-s',name,'-n','总览','-x','220','-y','64',
        '-c',str(root),'-P','-F','#{session_id}\t#{window_id}\t#{pane_id}',command('overview')).split('\t')
    tmux('set-option','-t',session,'@ctr_dashboard_root',str(root))
    for option,value in [('mouse','on'),('status-style','bg=colour234,fg=colour252'),
        ('status-left',f' #[bold,fg=cyan]{name} '),('status-left-length','25'),
        ('status-right','#[fg=yellow]4课题 · 6轮决策 · 验证集#[default] | C-b n 切换 · C-b z 放大 '),
        ('status-right-length','110')]:tmux('set-option','-t',session,option,value)
    def style(win):
        for option,value in [('remain-on-exit','on'),('pane-border-status','top'),
            ('pane-border-format',' #[fg=colour111]#{pane_title} ')]:tmux('set-window-option','-t',win,option,value)
    style(window)
    resources=tmux('split-window','-h','-p','45','-t',overview,'-P','-F','#{pane_id}',command('resources'))
    decision=tmux('split-window','-v','-p','45','-t',resources,'-P','-F','#{pane_id}',command('decisions'))
    panes={overview:'课题 / AUC曲线 / 预算',resources:'GPU / LLM / 训练进度',decision:'调度 / 异常'}
    def new_window(title,view):
        win,pane=tmux('new-window','-d','-t',session,'-n',title,'-c',str(root),'-P','-F',
            '#{window_id}\t#{pane_id}',command(view)).split('\t');style(win);return win,pane
    gpus=json.loads((root/'budget.json').read_text())['gpus']
    topic_window,first=new_window('课题',f'lane-{gpus[0]}');panes[first]=f'GPU {gpus[0]} · 研究轨迹'
    for gpu in gpus[1:]:
        pane=tmux('split-window','-h','-t',first,'-P','-F','#{pane_id}',command(f'lane-{gpu}'))
        panes[pane]=f'GPU {gpu} · 研究轨迹';tmux('select-layout','-t',topic_window,'tiled')
    _,pane=new_window('实验','experiments');panes[pane]='已提交验证结果'
    _,pane=new_window('成本','costs');panes[pane]='训练 / LLM / 参数 / 推理'
    _,pane=new_window('切片','slices');panes[pane]='当前课题最佳 vs 公共参考'
    for pane,title in panes.items():tmux('select-pane','-t',pane,'-T',title)
    tmux('select-window','-t',window);tmux('select-pane','-t',overview)
    for client in clients:tmux('switch-client','-c',client,'-t',session,check=False)
    manifest={'session':name,'socket':name,'session_id':session,'created_unix':time.time(),
        'root':str(root),'panes':panes,'windows':['总览','课题','实验','成本','切片'],
        'archived_session':archived,'attach':terminal.attach_command(name),
        'read_only':True,'test_scores_displayed':False}
    (output/'tmux.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--session',default='ctr-research');args=parser.parse_args()
    launch(args.root.resolve(),args.session)
