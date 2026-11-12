"""Read-only validation reranking; never reads test files or changes old trials."""
import argparse
import csv
import json
from pathlib import Path
import time

from common import read,write,sha
from selection import POLICY,TASKS,best_trial,trial_score,trial_logloss,task_deltas


def audit(root,output):
    sources=[root/'protocol.json',root/'references.json']+sorted((root/'topics').glob('gpu-*/state.json'))
    identities={str(path):sha(path) for path in sources}
    protocol=read(root/'protocol.json');references=read(root/'references.json')['trials'];rows=list(references)
    for path in sources[2:]:rows+=read(path)['trials']
    eligible=[row for row in rows if row['status']=='success' and row.get('metrics',{}).get('validation')]
    # Old receipts may have auc=click AUC. Selection helpers recompute the mean
    # from all four per-task validation records; no historical field is edited.
    selected=best_trial({'trials':eligible});din=next(row for row in references if row.get('baseline_kind')=='din')
    ordered=sorted(eligible,key=lambda row:(-trial_score(row),trial_logloss(row),row['metrics'].get('parameters',float('inf')),row['id']))
    def brief(row):
        return {'id':row['id'],'role':row.get('role'),'validation_macro_auc':trial_score(row),
            'validation_macro_logloss':trial_logloss(row),'per_task':row['metrics']['validation']['per_task'],
            'deltas_vs_din':task_deltas(row,din),'parameters':row['metrics'].get('parameters'),
            'model_path':row['model_path'],'source_sha256':row['source_sha256']}
    report={'schema':'historical-validation-selection-audit-v1','at_unix':time.time(),
        'scope':'Read-only reranking of historical validation receipts. The48h search optimized click AUC; this is not a new multitask search or a new final test selection.',
        'old_primary':protocol['primary'],'audit_selection_policy':POLICY,'trials':len(eligible),
        'original_reference':brief(din),'audit_validation_best':brief(selected),
        'top_validation':[brief(row) for row in ordered[:15]],
        'test_accessed':False,'training_started':False,'historical_files_rewritten':False,
        'input_sha256':identities}
    for path,digest in identities.items():
        if sha(Path(path))!=digest:raise ValueError('Historical input changed during audit')
    output.mkdir(parents=True,exist_ok=True);write(output/'report.json',report)
    with (output/'validation-ranking.csv').open('w') as stream:
        fields=['id','role','macro_auc','macro_logloss']+[task+'_auc' for task in TASKS]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in ordered:
            writer.writerow({'id':row['id'],'role':row.get('role'),'macro_auc':trial_score(row),
                'macro_logloss':trial_logloss(row),**{task+'_auc':row['metrics']['validation']['per_task'][task]['auc'] for task in TASKS}})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    report=audit(args.root.resolve(),args.output.resolve())
    print(json.dumps({'trials':report['trials'],'audit_best':report['audit_validation_best']['id'],
        'macro_auc':report['audit_validation_best']['validation_macro_auc'],'test_accessed':False}))
