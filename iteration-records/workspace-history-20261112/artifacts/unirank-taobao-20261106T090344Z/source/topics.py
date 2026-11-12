"""Optional auto wrapper: local research per GPU, compact separate topic decisions."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import threading
import time
import traceback

from jsonschema import Draft202012Validator

import controller as base
from common import read,write,sha
from knowledge import scoped_cards,REVISION
from references import observed_trial_refs


DECISION_INSTRUCTION=(
    'Use only the supplied compact evidence to choose the next topic action for this GPU. '
    'Write topic-decision.json: {action: continue|switch|merge, reason: concise string, '
    'evidence_refs: real trial IDs, topic: {title,question,hypothesis} for switch/merge, '
    'start_trial_ref: optional visible trial ID for switch, source_topic_ids: two topic IDs for merge, '
    'merge_control_refs: both frozen source best trial IDs for merge}. '
    'Continue deepens the current question; switch starts a different question; merge starts a new '
    'combined question using two topics with observed gains over their own entry references and the shared reference. '
    'For a merge, retain both source bests as utility controls, account for extra parameters/cost, '
    'and test whether the combination beats the stronger source; gains do not automatically add. '
    'Use topic-specific evidence, controls, cost and outstanding falsifiers, not a novelty score. '
    'Do not design, edit or train a model in this decision step. Model code, full histories and papers '
    'are intentionally absent. The subsequent original auto research step handles implementation. '
    'Keep the reason short and distinguish observations from unresolved compatibility. '
    'A single-seed positive result is provisional; citing a source is not mechanism confirmation.')


def valid_topic(value):
    if not isinstance(value,dict) or any(not isinstance(value.get(k),str) or not value[k].strip() for k in ['title','question','hypothesis']):
        raise ValueError('A topic requires title, question and hypothesis strings')
    return {key:value[key].strip() for key in ['title','question','hypothesis']}


def question_key(topic):return ' '.join(topic['question'].casefold().split())


def validate_gpu_budget(gpus,budget):
    allowed=budget.get('gpus',[])
    if isinstance(allowed,dict):allowed=allowed.get('auto',[])
    if not gpus or not set(gpus)<=set(allowed):raise ValueError('GPU lanes exceed the declared auto resource budget')


class TopicPool:
    def __init__(self,root,gpus,topics,references,protocol,deadline,decision_every=6):
        self.root=Path(root);self.gpus=list(gpus);self.protocol=protocol;self.deadline=deadline
        if not self.gpus or len(set(self.gpus))!=len(self.gpus) or any(type(g)!=int or g<0 for g in self.gpus):
            raise ValueError('GPU lanes must be distinct nonnegative integers')
        if len(topics)!=len(gpus):raise ValueError('Provide exactly one initial topic per GPU')
        topics=[valid_topic(topic) for topic in topics]
        if len({question_key(t) for t in topics})!=len(topics):raise ValueError('Initial topic questions must be distinct')
        ids=[r['id'] for r in references]
        if len(ids)!=len(set(ids)) or any(r.get('status')!='success' or not r.get('metrics',{}).get('validation') for r in references):
            raise ValueError('Shared references require unique successful validation records')
        if not references:raise ValueError('Shared references are required')
        self.references=copy.deepcopy(references);self.anchor=base.best_trial({'trials':references})
        if type(decision_every)!=int or decision_every<1:raise ValueError('decision_every must be a positive round interval')
        self.decision_every=decision_every
        self.lock=threading.RLock();self.cancelled=threading.Event();self.states={}
        for gpu,topic in zip(gpus,topics):
            episode={**topic,'id':f'g{gpu}-t000','status':'active','source_topic_ids':[],
                     'entry_trial_id':None,'entry_auc':None,'import_trial_refs':[]}
            self.states[gpu]={'mode':'auto','gpu':gpu,'phase':'ready','round':0,'completed_research_rounds':0,'trials':[],
                'agent_calls':[],'memory':[],'topics':[episode],'active_topic_id':episode['id'],'decisions':[]}

    def save(self):
        # Caller holds the RLock. Each lane file is atomic; the board contains only committed work.
        for gpu,state in self.states.items():write(self.root/'topics'/f'gpu-{gpu}'/'state.json',state)
        write(self.root/'topic-board.json',self.board())

    def board(self):
        with self.lock:
            rows=[]
            for gpu,state in self.states.items():
                for topic in state['topics']:
                    trials=[t for t in state['trials'] if t['topic_id']==topic['id']]
                    best=base.best_trial({'trials':trials})
                    rows.append({**topic,'gpu':gpu,'training_attempts':len(trials),
                        'failed_training_attempts':sum(t['status']=='failed' for t in trials),
                        'interrupted_training_attempts':sum(t['status'] in ['interrupted_by_user','budget_exhausted'] for t in trials),
                        'best_trial_id':best['id'] if best else None,
                        'best_auc':best['metrics']['validation']['auc'] if best else None,
                        'best_logloss':best['metrics']['validation']['logloss'] if best else None,
                        'observed_gain_vs_reference':best['metrics']['validation']['auc']-self.anchor['metrics']['validation']['auc'] if best else None,
                        'incremental_gain':best['metrics']['validation']['auc']-topic['entry_auc'] if best and topic['entry_auc'] is not None else None,
                        'job_wall_seconds':sum(t['wall_seconds'] for t in trials),
                        'parameters':best['metrics'].get('parameters') if best else None,
                        'inference_seconds':best['metrics']['validation'].get('inference_seconds') if best else None,
                        'evidence':'validation observations; same training seed is not independent replication'})
            return {'updated_unix':time.time(),'shared_reference':base.public_trial(self.anchor),'topics':rows,
                'lanes':[{'gpu':g,'phase':s['phase'],'active_topic_id':s['active_topic_id'],'round':s['round'],
                          'agent_wall_seconds':sum(c['wall_seconds'] for c in s['agent_calls'])} for g,s in self.states.items()]}

    def call_state(self,gpu):
        with self.lock:
            lane=self.states[gpu]
            topic=next(t for t in lane['topics'] if t['id']==lane['active_topic_id'])
            own=[t for t in lane['trials'] if t['topic_id']==lane['active_topic_id']]
            catalog={t['id']:t for t in self.references+[r for s in self.states.values() for r in s['trials']]}
            imported=[catalog[key] for key in topic['import_trial_refs']]
            local={t['id']:t for t in self.references+imported+own}
            chosen=base.best_trial({'trials':list(local.values())}) or self.anchor
            return {'mode':'auto','trials':copy.deepcopy(list(local.values())),'incumbent':chosen['id'],'agent_calls':[],'memory':[],
                'last_runtime_error':lane.get('last_runtime_error'),
                'additional_context_trial_ids':list(topic['import_trial_refs']),
                'additional_source_trial_ids':list(topic['import_trial_refs']),
                'latest_knowledge_usage':copy.deepcopy(lane.get('latest_knowledge_usage')) if lane.get('latest_knowledge_topic')==topic['id'] else None,
                'research_topic':copy.deepcopy(topic)}

    def decision_snapshot(self,gpu):
        with self.lock:
            lane=self.states[gpu];board=self.board()
            active=[t for t in board['topics'] if t['status']=='active']
            parked=sorted([t for t in board['topics'] if t['status']!='active' and (t['incremental_gain'] or 0)>0],
                          key=lambda t:t['incremental_gain'],reverse=True)[:4]
            compact=[]
            fields=['id','gpu','status','source_topic_ids','entry_trial_id','entry_auc','best_trial_id','best_auc','best_logloss',
                    'incremental_gain','observed_gain_vs_reference','training_attempts','failed_training_attempts',
                    'job_wall_seconds','parameters','inference_seconds']
            for topic in active+parked:
                compact.append({**{key:topic[key] for key in fields},'title':topic['title'][:120],
                    'question':topic['question'][:400],'hypothesis':topic['hypothesis'][:400]})
            catalog={t['id']:t for t in self.references+[r for s in self.states.values() for r in s['trials']]}
            own=[t for t in lane['trials'] if t['topic_id']==lane['active_topic_id']][-4:]
            visible={t['id'] for t in self.references+own}
            visible.update(t[key] for t in compact for key in ['entry_trial_id','best_trial_id'] if t[key])
            def brief(t):
                metrics=t.get('metrics',{});val=metrics.get('validation',{})
                return {'id':t['id'],'status':t['status'],'auc':val.get('auc'),'logloss':val.get('logloss'),
                        'wall_seconds':t['wall_seconds'],'parameters':metrics.get('parameters'),
                        'inference_seconds':val.get('inference_seconds')}
            return {'trials':copy.deepcopy([catalog[key] for key in visible]),
                'topic_portfolio':{'current_topic_id':lane['active_topic_id'],'board':{'topics':compact}},
                'light_context':{'gpu':gpu,'current_topic_id':lane['active_topic_id'],'topics':compact,
                    'shared_references':[brief(t) for t in self.references],'recent_own_results':[brief(t) for t in own],
                    'protocol':{key:self.protocol.get(key) for key in ['identity','primary','train_seed','epochs','batch_size','history_limit']},
                    'remaining_seconds':self.deadline-time.time(),'allowed_evidence_refs':sorted(visible),
                    'omitted_parked_topics':len(board['topics'])-len(compact),
                    'last_decision_error':lane.get('last_decision_error')}}

    def validate_decision(self,gpu,proposal,snapshot):
        decision=proposal.get('topic_decision')
        if not isinstance(decision,dict) or decision.get('action') not in ['continue','switch','merge']:
            raise ValueError('Require topic_decision action continue, switch or merge')
        if not isinstance(decision.get('reason'),str) or not decision['reason'].strip():raise ValueError('Topic decision requires an evidence/cost rationale')
        known={t['id'] for t in snapshot['trials']}
        observed_trial_refs(decision.get('evidence_refs',[]),known)
        action=decision['action']
        if action!='continue':valid_topic(decision.get('topic'))
        if action=='switch' and decision.get('start_trial_ref'):
            selected=next((t for t in snapshot['trials'] if t['id']==decision['start_trial_ref'] and t['status']=='success'),None)
            if selected is None:raise ValueError('Switch can import only a visible successful reference')
        if action=='merge':
            source_ids=decision.get('source_topic_ids',[])
            board={t['id']:t for t in snapshot['topic_portfolio']['board']['topics']}
            if not isinstance(source_ids,list) or len(source_ids)!=2 or len(set(source_ids))!=2 or not set(source_ids)<=set(board):
                raise ValueError('Merge requires two distinct observed topics')
            if any(board[k]['incremental_gain'] is None or board[k]['incremental_gain']<=0 or
                   board[k]['observed_gain_vs_reference'] is None or board[k]['observed_gain_vs_reference']<=0 for k in source_ids):
                raise ValueError('Merge requires positive increments and improvement over the shared reference in both source topics')
            required={board[k]['best_trial_id'] for k in source_ids}
            controls=decision.get('merge_control_refs',[])
            if not isinstance(controls,list) or len(controls)!=2 or set(controls)!=required or not set(controls)<=known:
                raise ValueError('Merge must retain both frozen source best trial IDs as controls')
        return copy.deepcopy(decision)

    def apply_decision(self,gpu,decision,entry_reference=None):
        with self.lock:
            lane=self.states[gpu];current=next(t for t in lane['topics'] if t['id']==lane['active_topic_id'])
            if decision['action']!='continue':
                topic=valid_topic(decision['topic'])
                active=[t for s in self.states.values() for t in s['topics'] if t['status']=='active']
                if any(question_key(t)==question_key(topic) for t in active):raise ValueError('This question is already active; do not duplicate another GPU topic')
                current['status']='parked'
                topic.update(id=f'g{gpu}-t{len(lane["topics"]):03d}',status='active',
                             source_topic_ids=decision.get('source_topic_ids',[]),entry_trial_id=None,entry_auc=None,
                             import_trial_refs=list(decision.get('merge_control_refs',[])) if decision['action']=='merge' else
                                               [entry_reference['id']] if entry_reference and entry_reference['id'] not in {t['id'] for t in self.references} else [])
                lane['topics'].append(topic);lane['active_topic_id']=topic['id']
                current=topic
            if current['entry_trial_id'] is None:
                reference=entry_reference or self.anchor
                current['entry_trial_id']=reference['id'];current['entry_auc']=reference['metrics']['validation']['auc']
            lane['decisions'].append({'round':lane['round'],'at_unix':time.time(),**decision,
                                      'resulting_topic_id':lane['active_topic_id']})
            self.save();return lane['active_topic_id']

    def decide(self,gpu,index):
        with self.lock:
            lane=self.states[gpu];lane['phase']='topic_decision';snapshot=self.decision_snapshot(gpu);self.save()
        directory=self.root/'topics'/f'gpu-{gpu}'/'rounds'/f'r{index:04d}'/'decision'
        directory.mkdir(parents=True,exist_ok=False)
        interface=base.phase_contract('propose','auto',{'trials':[]},self.protocol,0)['candidate_api']
        task={'objective':'Choose a research-topic allocation from compact committed evidence.',
              'information_boundary':'Same frozen data and training protocol; no model implementation in this step.',
              'available_inputs':interface['batch'],'personal_knowledge':'disabled'}
        runtime={'phase':'topic_decision','expected_files':['topic-decision.json'],'context':snapshot['light_context'],
                 'protocol_hash':self.protocol['identity'],'action':DECISION_INSTRUCTION}
        workspace={'editable_outputs':['topic-decision.json'],'model_files':[],
                   'available_evidence':'The supplied compact summaries only; no code, full trial histories or papers are staged.'}
        timeout=min(180,max(1,self.deadline-time.time()))
        try:
            receipt=base.agent_run(directory,task,runtime,workspace,[],timeout,resources=[])
        except Exception:
            if (directory/'agent-receipt.json').exists():
                with self.lock:lane['agent_calls'].append({'phase':'topic_decision','round':index,**read(directory/'agent-receipt.json')});self.save()
            raise
        with self.lock:lane['agent_calls'].append({'phase':'topic_decision','round':index,**receipt});self.save()
        if self.stopping():return
        decision=self.validate_decision(gpu,{'topic_decision':read(directory/'topic-decision.json')},snapshot)
        catalog={t['id']:t for t in snapshot['trials']};entry=self.anchor
        if decision['action']=='merge':entry=base.best_trial({'trials':[catalog[key] for key in decision['merge_control_refs']]})
        elif decision['action']=='switch' and decision.get('start_trial_ref'):entry=catalog[decision['start_trial_ref']]
        self.apply_decision(gpu,decision,entry)
        with self.lock:lane.pop('last_decision_error',None);lane['consecutive_decision_errors']=0;self.save()

    def execute_round(self,gpu):
        with self.lock:
            lane=self.states[gpu];index=lane['round'];lane['phase']='propose';snapshot=self.call_state(gpu);self.save()
        directory=self.root/'topics'/f'gpu-{gpu}'/'rounds'/f'r{index:04d}'
        proposedir=directory/'propose';proposedir.mkdir(parents=True,exist_ok=False)
        industry=[json.loads(line) for line in (base.DESIGN/'tasks/taobao_display_ctr/industry/cards.jsonl').read_text().splitlines() if line.strip()]
        papers=read(base.DESIGN/'tasks/taobao_display_ctr/paper_sources.json')
        evidence=scoped_cards(industry,papers,[],self.protocol['data_identity'],self.protocol['identity'])
        try:
            receipt=base.researcher(proposedir,'propose','auto',snapshot,self.protocol,self.deadline,evidence,campaign_root=self.root)
        except Exception:
            failed=proposedir/'agent-receipt.json'
            if failed.exists():
                with self.lock:
                    lane['agent_calls'].append({'phase':'propose','round':index,'topic_id':lane['active_topic_id'],**read(failed)})
                    self.save()
            raise
        with self.lock:lane['agent_calls'].append({'phase':'propose','round':index,'topic_id':lane['active_topic_id'],**receipt});self.save()
        if self.stopping():return
        proposal=read(proposedir/'proposal.json')
        Draft202012Validator(read(base.DESIGN/'agent/study.schema.json')).validate(proposal['study'])
        if proposal['study']['protocol_ref']!=self.protocol['identity']:raise ValueError('Incorrect protocol identity')
        observed_trial_refs(proposal['study']['evidence_refs'],{r['id'] for r in snapshot['trials']})
        parent=next((t for t in snapshot['trials'] if t['id']==proposal.get('parent_trial') and t['status']=='success'),None)
        if parent is None:raise ValueError('Parent must be an authorized successful trial')
        variants=proposal.get('variants',[])
        if not isinstance(variants,list) or not 1<=len(variants)<=4 or not any(v.get('role')=='candidate' for v in variants):raise ValueError('Require 1-4 variants including candidate')
        if len({v['name'] for v in variants})!=len(variants):raise ValueError('Duplicate variant names')
        from train import check_source
        for v in variants:
            if not re.fullmatch('[A-Za-z0-9_-]{1,60}',v['name']) or Path(v['file']).name!=v['file'] or not v['file'].endswith('.py') or v.get('role') not in ['candidate','control']:
                raise ValueError('Invalid variant identity or role')
            check_source(proposedir/v['file'])
        topic=snapshot['research_topic']
        current_trials=[t for t in snapshot['trials'] if t.get('topic_id')==topic['id']]
        if topic['source_topic_ids'] and not current_trials:
            controls=set(topic['import_trial_refs'])
            cited=observed_trial_refs(proposal['study']['evidence_refs'],controls)
            compared=observed_trial_refs([json.dumps(proposal['study']['contrasts'])],controls)
            if not controls<=set(cited) or not controls<=set(compared):
                raise ValueError('Merge study must cite and compare both frozen source controls')
        estimate=len(variants)*self.protocol['estimated_unit_seconds']
        if self.deadline-time.time()<estimate:
            with self.lock:lane['phase']='budget_insufficient';self.save()
            return
        with self.lock:
            current=next(t for t in lane['topics'] if t['id']==lane['active_topic_id'])
            if current['entry_trial_id'] is None:
                current['entry_trial_id']=parent['id'];current['entry_auc']=parent['metrics']['validation']['auc']
            topic_id=current['id'];self.save()
        write(directory/'knowledge-usage.json',base.used_knowledge(proposedir,proposal,evidence,None))
        for v in variants:
            if self.stopping():break
            with self.lock:lane['phase']='training';self.save()
            identifier=f'g{gpu}-r{index:04d}-'+v['name']
            # A lane never dispatches another variant until this one has returned.
            row=base.train_job(data=self.root/'prepared',source=proposedir/v['file'],config=v.get('config',{}),
                out=self.root/'topics'/f'gpu-{gpu}'/'trials'/identifier,seed=self.protocol['train_seed'],gpu=gpu,
                protocol=self.protocol,deadline=self.deadline)
            if row['status']!='success' and self.stopping():
                row['status']='interrupted_by_user' if self.cancelled.is_set() or (self.root/'STOP').exists() else 'budget_exhausted'
            row.update(id=identifier,parent=proposal['parent_trial'],role=v['role'],round=index,
                variant_name=v['name'],topic_id=topic_id,note=v.get('note'),model_path=str(Path(row['output'])/'model.py'),
                study_ref=str(proposedir/'proposal.json'),knowledge_revision=REVISION)
            with self.lock:lane['trials'].append(row);self.save()
        results=[t for t in lane['trials'] if t.get('round')==index]
        usage=base.used_knowledge(proposedir,proposal,evidence,None,results)
        write(directory/'knowledge-usage.json',usage)
        with self.lock:
            lane['latest_knowledge_usage']=usage;lane['latest_knowledge_topic']=topic_id
            lane.pop('last_runtime_error',None);lane['consecutive_errors']=0;self.save()
        return True

    def stopping(self):return self.cancelled.is_set() or (self.root/'STOP').exists() or time.time()>=self.deadline

    def stop_children(self):
        for pattern in ['topics/**/process.json','topics/**/agent-process.json']:
            for path in self.root.glob(pattern):
                try:
                    record=read(path);pid=record.get('pid');proc=Path(f'/proc/{pid}/stat')
                    if not pid or not proc.exists():continue
                    fields=proc.read_text().split()
                    if fields[2]!='Z' and fields[21]==record.get('start_ticks') and os.getpgid(pid)!=os.getpgrp():
                        os.killpg(pid,signal.SIGKILL)
                except (FileNotFoundError,ProcessLookupError):pass

    def lane_loop(self,gpu,max_rounds=None):
        state=self.states[gpu]
        while not self.stopping() and (max_rounds is None or state['round']<max_rounds):
            if self.deadline-time.time()<self.protocol['estimated_unit_seconds']:break
            completed=False;index=state['round']
            try:
                completed=self.execute_round(gpu)
                if state['phase']=='budget_insufficient':break
            except Exception as error:
                if self.stopping():break
                with self.lock:
                    state['consecutive_errors']=state.get('consecutive_errors',0)+1
                    state['last_runtime_error']={'round':state['round'],'error':str(error),'consecutive':state['consecutive_errors']}
                    with (self.root/'topics'/f'gpu-{gpu}'/'failures.jsonl').open('a') as stream:
                        stream.write(json.dumps({'round':state['round'],'error':repr(error),'traceback':traceback.format_exc(),'at':time.time()})+'\n')
                    self.save()
                if state['consecutive_errors']>=3:
                    with self.lock:state['phase']='failed_before_assessment';self.save()
                    return
            finally:
                with self.lock:
                    state['round']+=1
                    if completed:state['completed_research_rounds']+=1
                    self.save()
            if (completed and not self.stopping() and state['completed_research_rounds']%self.decision_every==0 and
                    (max_rounds is None or state['round']<max_rounds) and
                    self.deadline-time.time()>self.protocol['estimated_unit_seconds']):
                try:self.decide(gpu,index)
                except Exception as error:
                    if self.stopping():break
                    with self.lock:
                        state['consecutive_decision_errors']=state.get('consecutive_decision_errors',0)+1
                        state['last_decision_error']={'round':index,'error':str(error)}
                        with (self.root/'topics'/f'gpu-{gpu}'/'decision-failures.jsonl').open('a') as stream:
                            stream.write(json.dumps({'round':index,'error':repr(error),'at':time.time()})+'\n')
                        self.save()
                    if state['consecutive_decision_errors']>=3:
                        with self.lock:state['phase']='failed_before_assessment';self.save()
                        return
        with self.lock:
            state['phase']='cancelled' if self.cancelled.is_set() or (self.root/'STOP').exists() else 'research_stopped'
            self.save()

    def run(self,max_rounds=None):
        if (self.root/'STOP').exists():raise ValueError('User-stopped campaign cannot be restarted')
        if (self.root/'topic-board.json').exists() or (self.root/'status.json').exists():raise ValueError('Use a new campaign root; existing state cannot be overwritten')
        if time.time()>=self.deadline:raise ValueError('Budget already expired')
        base.prepare_template()
        with self.lock:self.save()
        write(self.root/'topic-runtime-manifest.json',{'revision':'auto-gpu-topics-v2','created_unix':time.time(),
            'context_policy':'original auto inside each topic; compact separate allocation decisions','decision_every':self.decision_every,
            'gpus':self.gpus,'protocol_hash':self.protocol['identity'],'knowledge_revision':REVISION,
            'sources':{name:sha(Path(__file__).with_name(name)) for name in ['topics.py','controller.py','agent.py','train.py','data.py','baseline.py','knowledge.py']}})
        write(self.root/'runner-process.json',{'pid':os.getpid(),'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split()[21]})
        old_handlers={}
        def cancel(signum,frame):self.cancelled.set();self.stop_children()
        for signum in [signal.SIGTERM,signal.SIGINT]:old_handlers[signum]=signal.signal(signum,cancel)
        try:
            with ThreadPoolExecutor(len(self.gpus)) as executor:
                pending=[executor.submit(self.lane_loop,gpu,max_rounds) for gpu in self.gpus]
                while not all(f.done() for f in pending):
                    write(self.root/'status.json',{'phase':'topic_research','updated_unix':time.time(),'deadline_unix':self.deadline})
                    if self.stopping():self.stop_children()
                    time.sleep(0.2)
                for future in as_completed(pending):future.result()
        finally:
            for signum,handler in old_handlers.items():signal.signal(signum,handler)
            if self.stopping():self.stop_children()
            failed=any(s['phase']=='failed_before_assessment' for s in self.states.values())
            write(self.root/'status.json',{'phase':'terminated_by_user' if self.cancelled.is_set() or (self.root/'STOP').exists() else 'topic_research_with_failures' if failed else 'topic_research_stopped',
                'updated_unix':time.time(),'assessment_launched':False,'note':'Research wrapper does not launch a held-out assessment.'})


def load_references(path,protocol):
    references=[t for t in read(path)['trials'] if t.get('baseline_kind')]
    if len(references)!=3 or {r.get('baseline_kind') for r in references}!={'static','pooled','din'}:raise ValueError('Require all three shared references')
    for row in references:
        if row.get('status')!='success' or row.get('metrics',{}).get('data_identity')!=protocol['data_identity']:
            raise ValueError('Reference data identity/status mismatch')
        if not Path(row['model_path']).is_file() or sha(row['model_path'])!=row['source_sha256']:raise ValueError('Reference source digest mismatch')
        if row.get('seed')!=protocol['train_seed']:raise ValueError('Reference training seed mismatch')
        expected=protocol.get('source_sha256',{}).get('baseline.py')
        if expected and row['source_sha256']!=expected:raise ValueError('Reference baseline differs from frozen protocol')
        command=read(Path(row['output'])/'process.json')['argv']
        for flag,key in [('--epochs','epochs'),('--batch-size','batch_size'),('--history-limit','history_limit')]:
            if flag not in command or command[command.index(flag)+1]!=str(protocol[key]):raise ValueError('Reference recipe mismatch: '+flag)
    return references


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--gpus',required=True);parser.add_argument('--topics',required=True,type=Path)
    parser.add_argument('--references',required=True,type=Path)
    parser.add_argument('--decision-every',type=int,default=6,help='Run a separate compact topic decision every N completed research rounds')
    args=parser.parse_args();root=args.root.absolute();protocol=read(root/'protocol.json');budget=read(root/'budget.json')
    gpus=[int(g) for g in args.gpus.split(',')];validate_gpu_budget(gpus,budget)
    if read(root/'prepared/metadata.json')['identity']!=protocol['data_identity']:raise ValueError('Prepared data identity mismatch')
    for required in ['knowledge-assets-v2/manifest.json','auto-validation-only.json']:
        if not (root/required).is_file():raise ValueError('Missing campaign input: '+required)
    if not read(root/'auto-validation-only.json').get('enabled'):raise ValueError('This wrapper is validation-only')
    for name in ['train.py','data.py','baseline.py']:
        expected=protocol.get('source_sha256',{}).get(name)
        if expected and sha(base.HERE/name)!=expected:raise ValueError('Frozen numeric source mismatch: '+name)
    pool=TopicPool(root,gpus,read(args.topics)['topics'],
                   load_references(args.references,protocol),protocol,budget['deadline_unix'],args.decision_every)
    with (root/'topics.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);pool.run()


if __name__=='__main__':main()
