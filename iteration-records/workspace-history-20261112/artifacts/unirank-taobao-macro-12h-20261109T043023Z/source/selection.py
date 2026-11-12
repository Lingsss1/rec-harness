"""One fixed four-task selection rule shared by every research decision."""
import math
from pathlib import Path

from common import read

POLICY=read(Path(__file__).with_name('selection-policy.json'))
TASKS=tuple(POLICY['tasks'])
PRIMARY_LABEL='four-task macro ROC-AUC'


def validate_policy(policy):
    if policy!=POLICY:raise ValueError('Selection policy differs from the frozen four-task rule')
    return policy


def aggregate(per_task,allow_undefined=False):
    if set(per_task)!=set(TASKS):raise ValueError('All four registered tasks are required for selection')
    values=[];losses=[]
    for task in TASKS:
        auc=per_task[task].get('auc');loss=per_task[task].get('logloss')
        if auc is None and allow_undefined:values.append(None)
        elif isinstance(auc,bool) or not isinstance(auc,(int,float)) or not math.isfinite(auc) or not 0<=auc<=1:
            raise ValueError('Invalid AUC for '+task)
        else:values.append(float(auc))
        if isinstance(loss,bool) or not isinstance(loss,(int,float)) or not math.isfinite(loss) or loss<0:
            raise ValueError('Invalid Logloss for '+task)
        losses.append(float(loss))
    mean_auc=None if None in values else math.fsum(POLICY['weights'][t]*v for t,v in zip(TASKS,values))
    mean_loss=math.fsum(POLICY['weights'][t]*v for t,v in zip(TASKS,losses))
    return {'selection_metric':'macro_auc','selection_policy_id':POLICY['id'],'selection_score':mean_auc,
        'auc':mean_auc,'logloss':mean_loss,'macro_auc':mean_auc,'mean_task_auc':mean_auc,
        'macro_logloss':mean_loss,'mean_task_logloss':mean_loss,
        'click_auc':per_task['is_click']['auc'],'click_logloss':per_task['is_click']['logloss']}


def validation(row):
    value=row.get('metrics',{}).get('validation')
    if not value:return None
    result=aggregate(value.get('per_task',{}))
    declared=value.get('selection_policy_id')
    if declared and declared!=POLICY['id']:raise ValueError('Foreign selection-policy identity')
    if value.get('selection_metric') and value['selection_metric']!='macro_auc':
        raise ValueError('Conflicting declared selection metric')
    if 'selection_score' in value and abs(value['selection_score']-result['selection_score'])>1e-12:
        raise ValueError('Stored selection score disagrees with task metrics')
    return result


def trial_score(row):
    value=validation(row);return value['macro_auc'] if value else None


def trial_logloss(row):
    value=validation(row);return value['macro_logloss'] if value else None


def checkpoint_key(metrics):
    value=aggregate(metrics.get('per_task',{}))
    return (-value['macro_auc'],value['macro_logloss'])


def best_trial(state):
    rows=[row for row in state['trials'] if row.get('status')=='success' and row.get('metrics',{}).get('validation')]
    def key(row):
        validation(row)
        return checkpoint_key(row['metrics']['validation'])+(row.get('metrics',{}).get('parameters',math.inf),row['id'])
    return min(rows,key=key) if rows else None


def task_deltas(row,reference):
    current=row.get('metrics',{}).get('validation',{}).get('per_task',{}) if row else {}
    previous=reference.get('metrics',{}).get('validation',{}).get('per_task',{}) if reference else {}
    return {task:{metric:current[task][metric]-previous[task][metric] for metric in ['auc','logloss']}
            for task in TASKS if task in current and task in previous}
