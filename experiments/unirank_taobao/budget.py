"""Immutable research clock, including explicitly requested runs on cached data."""
from pathlib import Path
import time

from common import read,write


def budget_state(policy,setup,acquisition,manifest,previous=None):
    seconds=policy['duration_seconds']
    if type(seconds)!=int or seconds<=0:raise ValueError('Positive integer duration required')
    if policy['start_event'] not in ['dataset_download_and_checksum_verification_completed',
                                     'new_campaign_started_with_verified_dataset']:
        raise ValueError('Unknown budget start event')
    previous=previous or {};started=None
    prior=previous.get('start_unix')
    already_started=(previous.get('start_event')==policy['start_event'] and
                     previous.get('clock_phase')=='running' and
                     isinstance(prior,(int,float)) and not isinstance(prior,bool) and prior>0)
    if already_started:
        started=prior
    elif acquisition.get('phase')=='downloaded_and_verified':
        verified=manifest.get('verified_unix')
        if not isinstance(verified,(int,float)) or isinstance(verified,bool) or verified<=0:
            raise ValueError('Verified release must have a recorded verification completion timestamp')
        started=(verified if policy['start_event']=='dataset_download_and_checksum_verification_completed'
                 else policy.get('campaign_start_unix'))
        if not isinstance(started,(int,float)) or isinstance(started,bool) or started<=0:
            raise ValueError('A new run on cached data requires an explicit recorded campaign start')
        if started<verified:
            raise ValueError('A new campaign cannot start before dataset verification')
    return {**previous,'revision':'dataset-ready-budget-v1','seconds':seconds,
        'start_event':policy['start_event'],'start_unix':started,
        'deadline_unix':started+seconds if started is not None else None,
        'clock_phase':'running' if started is not None else 'waiting_for_verified_dataset',
        'gpus':previous.get('gpus',[0,1,2,3]),'model':previous.get('model','gpt-6.1-sol'),
        'reasoning_effort':previous.get('reasoning_effort','high'),'personal_enabled':False,'decision_every':6,
        'cost_scope':('All preparation, calibration, research calls, failures, training, controls and scoring after the recorded start count toward this research wall clock. Cached-data reruns use their explicitly requested new campaign start; original dataset verification timestamps are preserved.'),
        'pre_budget':{'start_unix':setup['created_unix'],'end_unix':started,
            'wall_seconds':max(0,started-setup['created_unix']) if started is not None else None,
            'scope':'Dataset acquisition, environment setup and checks before recorded dataset verification; excluded from research clock, retained as preparation cost.'},
        'assessment':previous.get('assessment','validation during research; held-out closed until a frozen final suite'),
        'single_experiment_limit_seconds':previous.get('single_experiment_limit_seconds',21600)}


def refresh(root):
    root=Path(root);policy=read(root/'budget-policy.json')
    old=read(root/'budget.json') if (root/'budget.json').exists() else {}
    manifest=read(root/'dataset-source-manifest.json') if (root/'dataset-source-manifest.json').exists() else {}
    state=read(root/'acquisition-status.json') if (root/'acquisition-status.json').exists() else {}
    current=budget_state(policy,read(root/'setup.json'),state,manifest,old)
    if current!=old:
        with (root/'budget-changes.jsonl').open('a') as stream:
            import json
            stream.write(json.dumps({'at_unix':time.time(),'authority':policy.get('authority'),
                'before':old,'after':current},ensure_ascii=False)+'\n')
        write(root/'budget.json',current)
    return current
