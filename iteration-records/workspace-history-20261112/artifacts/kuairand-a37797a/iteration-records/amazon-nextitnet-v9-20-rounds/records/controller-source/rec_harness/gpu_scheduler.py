"""Refill idle devices after each completion, keeping accounting on the caller thread."""
from concurrent.futures import FIRST_COMPLETED, wait


def dispatch(jobs, devices, can_start, launch, complete):
    pending=list(jobs);free=list(devices);running={}
    if not free or len(set(free))!=len(free):
        raise ValueError('distinct nonempty device slots required')
    while pending or running:
        while pending and free and can_start():
            device=free.pop(0);job=pending.pop(0)
            future=launch(job,device)
            running[future]=(job,device)
        if not running:break
        done,_=wait(running,return_when=FIRST_COMPLETED)
        for future in done:
            job,device=running.pop(future)
            complete(job,device,future)
            free.append(device)
    return pending
