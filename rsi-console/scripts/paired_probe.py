"""Candidate probe proposed by the RSI agent: paired hit/ranking trade-off.
Never select a model from this diagnostic; the frozen evaluator owns retention.
"""
import argparse, hashlib, json, math
from pathlib import Path

def paired(parent, candidate):
    if parent['protocolId'] != candidate['protocolId'] or parent['split'] != 'validation' or candidate['split'] != 'validation':
        raise ValueError('stale protocol or held-out test evidence')
    def rows(receipt):
        result={}
        for row in receipt['records']:
            if row['user'] in result: raise ValueError('duplicate user')
            rank=row['rank']
            if not isinstance(rank,int) or rank<0: raise ValueError('invalid rank')
            expected_hr=float(rank<5); expected_ndcg=1/math.log2(rank+2) if rank<5 else 0.
            for key,value in [('hr',expected_hr),('ndcg',expected_ndcg)]:
                if not math.isfinite(row[key]) or abs(row[key]-value)>1e-10: raise ValueError('rank/metric mismatch')
            result[row['user']]=row
        if not result: raise ValueError('empty records')
        return result
    a,b=rows(parent),rows(candidate)
    if a.keys()!=b.keys(): raise ValueError('unpaired users')
    groups={k:{'count':0,'ndcgDeltaSum':0.} for k in ('hit_gained','hit_lost','both_hit','both_miss')}
    for user,x in a.items():
        y=b[user]
        key='hit_gained' if y['hr']>x['hr'] else 'hit_lost' if y['hr']<x['hr'] else 'both_hit' if x['hr'] else 'both_miss'
        groups[key]['count']+=1;groups[key]['ndcgDeltaSum']+=y['ndcg']-x['ndcg']
    n=len(a)
    return {'protocolId':parent['protocolId'],'parentCode':parent['codeIdentity'],'candidateCode':candidate['codeIdentity'],'split':'validation','users':n,'groups':groups,'hrDelta':(groups['hit_gained']['count']-groups['hit_lost']['count'])/n,'ndcgDelta':sum(g['ndcgDeltaSum'] for g in groups.values())/n,'interpretation':'Descriptive paired decomposition, not causal attribution or evidence of future utility.'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('parent');parser.add_argument('candidate');parser.add_argument('--output',required=True);args=parser.parse_args()
    report=paired(json.loads(Path(args.parent).read_text()),json.loads(Path(args.candidate).read_text()))
    report['evidenceHashes']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [args.parent,args.candidate]}
    Path(args.output).write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='evidenceHashes'}))
if __name__=='__main__':main()
