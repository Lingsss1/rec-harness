"""Unchanged accepted helper functions extracted from formal_worker.py."""
import csv,errno,os,signal,sys,time,subprocess
from pathlib import Path
def require(ok, message):
    if not ok:
        raise ValueError(message)

def gpu_snapshot(gpu, timeout, expected_uuid=None, idle=False):
    text=subprocess.check_output(['nvidia-smi','--id',str(gpu),'--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True,timeout=timeout)
    rows=list(csv.reader(text.strip().splitlines()))
    require(len(rows)==1 and len(rows[0])==3,'Ambiguous assigned physical GPU')
    index,uuid,used=[x.strip() for x in rows[0]]
    require(int(index)==gpu and uuid.startswith('GPU-') and (expected_uuid is None or uuid==expected_uuid), 'Physical GPU identity differs')
    apps_text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=timeout)
    pids=[]
    for record in csv.reader(apps_text.strip().splitlines()):
        require(len(record)==2,'Malformed NVML process row')
        if record[0].strip()==uuid:pids.append(int(record[1]))
    if idle:require(not pids and int(used)<256,'Assigned GPU occupied; no wait or retry')
    return dict(index=gpu,uuid=uuid,memory_used_bytes=int(used)*2**20,compute_pids=pids,
                pid_scope='Opaque NVML IDs; no local/host PID equivalence asserted')

def parse_process_stat(text, page_size, clock_ticks):
    parts=text[text.rfind(')')+2:].split()
    return dict(state=parts[0],ppid=int(parts[1]),pgrp=int(parts[2]),session=int(parts[3]),
        cpu_seconds=sum(int(parts[i]) for i in (11,12,13,14))/clock_ticks,
        rss_bytes=max(0,int(parts[21]))*page_size,start_ticks=int(parts[19]))

def process_records():
    page=os.sysconf('SC_PAGE_SIZE');ticks=os.sysconf('SC_CLK_TCK');records={}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():continue
        try:records[int(entry.name)]=parse_process_stat((entry/'stat').read_text(),page,ticks)
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
    return records

def owned_process_records(owner_pid=None):
    """Subreaper adoption retains even descendants that leave the runner session."""
    owner_pid=os.getpid() if owner_pid is None else owner_pid
    records=process_records();selected={owner_pid}
    while True:
        more={p for p,v in records.items() if v['ppid'] in selected}
        if more<=selected:break
        selected.update(more)
    selected.discard(owner_pid)
    return {p:records[p] for p in selected}

def process_tree_snapshot():
    records=owned_process_records()
    return dict(pids=sorted(records),cpu_seconds=sum(v['cpu_seconds'] for v in records.values()),
                rss_bytes=sum(v['rss_bytes'] for v in records.values()))

def enable_subreaper():
    import ctypes
    require(sys.platform.startswith('linux'), 'Linux subreaper support required')
    libc=ctypes.CDLL(None,use_errno=True)
    require(libc.prctl(36,1,0,0,0)==0, 'Cannot adopt orphaned owned compiler descendants')

def signal_owned(pid, record, sig):
    """Prefer pidfd; ENOSYS alone permits a fresh owned-chain/start-time fallback."""
    if record['state']=='Z':return False
    try:
        current=parse_process_stat(Path('/proc',str(pid),'stat').read_text(),
            os.sysconf('SC_PAGE_SIZE'),os.sysconf('SC_CLK_TCK'))
        require(current['start_ticks']==record['start_ticks'], 'Owned PID identity changed')
        try:
            if not (hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal')):
                raise OSError(errno.ENOSYS,'pidfd API unavailable')
            fd=os.pidfd_open(pid,0)
            try:
                current=parse_process_stat(Path('/proc',str(pid),'stat').read_text(),
                    os.sysconf('SC_PAGE_SIZE'),os.sysconf('SC_CLK_TCK'))
                require(current['start_ticks']==record['start_ticks'], 'Owned PID identity changed')
                signal.pidfd_send_signal(fd,sig)
                return 'pidfd'
            finally:os.close(fd)
        except OSError as exc:
            if exc.errno!=errno.ENOSYS:raise
            owned=owned_process_records().get(pid)
            if owned is None and not Path('/proc',str(pid)).exists():return False
            require(owned is not None and owned['start_ticks']==record['start_ticks'],
                    'ENOSYS fallback requires a current owned descendant with the same start time')
            current=parse_process_stat(Path('/proc',str(pid),'stat').read_text(),
                os.sysconf('SC_PAGE_SIZE'),os.sysconf('SC_CLK_TCK'))
            require(current['start_ticks']==record['start_ticks'], 'Fallback PID identity changed')
            os.kill(pid,sig)
            return 'ppid_start_ticks_os_kill'
    except (FileNotFoundError,ProcessLookupError):return False

def cleanup_owned_processes(processes, grace, term_grace=None):
    """Bounded TERM then KILL; preserve Popen return-code ownership and reap adoptees.

    The caller must enable subreaper before spawning any phase process. A dead
    session leader is never taken as evidence that its compiler children exited.
    """
    started=time.monotonic();events=[];reaped=[];known_roots={p.pid for p in processes}
    remaining={}
    for sig in (signal.SIGTERM,signal.SIGKILL):
        deadline=time.monotonic()+(term_grace if sig==signal.SIGTERM and term_grace is not None else grace);signalled=set()
        while True:
            for p in processes:p.poll()
            remaining=owned_process_records()
            for pid,record in list(remaining.items()):
                if record['ppid']==os.getpid() and pid not in known_roots:
                    try:
                        reaped_pid,status=os.waitpid(pid,os.WNOHANG)
                        if reaped_pid:reaped.append(dict(pid=pid,start_ticks=record['start_ticks'],wait_status=status))
                    except ChildProcessError:pass
                token=(pid,record['start_ticks'])
                if token not in signalled:
                    method=signal_owned(pid,record,sig)
                    if method:
                        signalled.add(token);events.append(dict(pid=pid,start_ticks=token[1],signal=int(sig),method=method))
            remaining=owned_process_records()
            if not remaining:
                for p in processes:p.poll()
                return dict(completed=True,signals=events,reaped_adoptees=reaped,remaining=[],
                    wall_seconds=time.monotonic()-started,subreaper=True,
                    pidfd_identity_checked=bool(events) and all(e['method']=='pidfd' for e in events),
                    signalling_methods=sorted({e['method'] for e in events}),
                    fallback_scope='ENOSYS only: fresh owned PPID chain and start-time checks before os.kill; these checks and kill are not atomic like pidfd.')
            if time.monotonic()>=deadline:break
            time.sleep(min(.05,max(0.,deadline-time.monotonic())))
    return dict(completed=False,signals=events,reaped_adoptees=reaped,
        remaining=[dict(pid=p,**v) for p,v in remaining.items()],wall_seconds=time.monotonic()-started,
        subreaper=True,pidfd_identity_checked=bool(events) and all(e['method']=='pidfd' for e in events),
        signalling_methods=sorted({e['method'] for e in events}),
        fallback_scope='ENOSYS only: fresh owned PPID chain and start-time checks before os.kill; these checks and kill are not atomic like pidfd.')
