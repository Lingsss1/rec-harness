"""Online memory comparison from one audited pre-decision baseline."""
import argparse
import copy
import fcntl
import shutil
import sqlite3
from pathlib import Path
from .core import ContractError,digest,file_hash,read_json,write_json
from .runtime import Research
from .online_validation import load_suite,episode
from .memory_comparison import audit_pair


def fork_baseline(source,destination,mode):
    """Copy a settled baseline, retaining its event ancestry and charged costs."""
    if mode not in {'graph','flat'}:raise ContractError('invalid memory mode')
    source=Path(source).resolve();destination=Path(destination).resolve()
    if source==destination or source in destination.parents:raise ContractError('fork must be outside source')
    research=Research(source,agent=object())
    try:
        with research.lock():
            integrity=research.verify();state=read_json(research.state_path)
            events=research.store.events()
            if state['round']!=0 or not state['baseline'] or not state['baseline']['valid']:
                raise ContractError('valid pre-decision baseline required')
            if any(e['kind'] in {'agent_submitted','round_started','baseline_forked'} for e in events):
                raise ContractError('cannot fork a research trajectory')
            if research.store.db.execute("SELECT 1 FROM budget WHERE status='running'").fetchone():
                raise ContractError('baseline has unsettled work')
            config=copy.deepcopy(research.config);config['memory_presentation']=mode
            limit=config.get('context_byte_limit')
            if type(limit) is not int or limit<=0:raise ContractError('shared context byte limit required')
            binding={'source_head':integrity['head'],'source_config_hash':digest(research.config),
                'state_hash':digest(state),'config_hash':digest(config),
                'shared_cost':research.store.budget()['charged_or_reserved']}
            if destination.exists():
                if not (destination/'fork.json').exists() or read_json(destination/'fork.json')!=binding:
                    raise ContractError('fork destination or ancestry changed')
                child=Research(destination,agent=object())
                try:
                    child.verify()
                    if child.config!=config:raise ContractError('fork configuration changed')
                    if not any(e['kind']=='baseline_forked' and e['payload']==binding for e in child.store.events()):
                        raise ContractError('fork ancestry event missing')
                finally:child.store.close()
                return binding
            staging=destination.with_name(destination.name+'.staging')
            if staging.exists():raise ContractError('incomplete fork staging requires inspection')
            staging.mkdir(parents=True)
            # CAS contains the collected worker receipts; omit disposable jail trees.
            for name in ['artifacts','research_archive','evaluator','semantic-retrieval']:
                if (source/name).exists():
                    shutil.copytree(source/name,staging/name,ignore=shutil.ignore_patterns('jail'))
            (staging/'memory').mkdir()
            database=sqlite3.connect(staging/'memory/memory.sqlite')
            try:research.store.db.backup(database)
            finally:database.close()
            write_json(staging/'campaign.json',config);write_json(staging/'state.json',state)
            manifest=read_json(staging/'research_archive/manifest.json');manifest['config_hash']=digest(config)
            manifest['baseline_fork']=binding
            write_json(staging/'research_archive/manifest.json',manifest)
            child=Research(staging,agent=object())
            try:
                child.store.event('baseline_forked',binding);child.verify()
            finally:child.store.close()
            write_json(staging/'fork.json',binding)
            staging.rename(destination)
            return binding
    finally:research.store.close()


def run_validation(suite_path,output,stage,limit,*,agent_factory=None):
    if stage not in {'development','confirmation'}:raise ContractError('invalid stage')
    if type(limit) is not int or limit<=0:raise ContractError('positive context limit required')
    suite,inputs=load_suite(suite_path);output=Path(output).resolve()
    root=output.parent/'memory-online';root.mkdir(parents=True,exist_ok=True)
    binding={'suite_hash':suite['hash'],'stage':stage,'context_byte_limit':limit,
        'decision_backend':'injected' if agent_factory is not None else 'codex',
        'implementation':{p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')}}
    with (root/'controller.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ContractError('memory comparison already active') from error
        if (root/'binding.json').exists() and read_json(root/'binding.json')!=binding:raise ContractError('memory comparison inputs changed')
        write_json(root/'binding.json',binding);rows=[]
        for index,original in enumerate(suite['cases']):
            if original['stage']!=stage:continue
            case=copy.deepcopy(original);config=case['config']
            if any(config.get(key) for key in ['initial_memory_files','initial_memory','measurement_discovery']):
                raise ContractError('memory comparison requires a frozen built-in baseline')
            config.update(context_byte_limit=limit,memory_presentation='graph')
            config.setdefault('codex',{})['web_search']='disabled'
            parent=root/case['id'];baseline=parent/'baseline';source=parent/'baseline-inputs'
            inputs.restore(case['inputs'],source)
            start={**config,'source':str(source),'research_policy':suite['baseline_policy']}
            if not (baseline/'campaign.json').exists():Research.init(baseline,start)
            reference=Research(baseline,agent=object())
            try:
                from .semantic_encoder import verify_options
                verify_options(start,reference.config)
                for key,value in start.items():
                    if key not in {'source','protocol','execution','evaluator_files','semantic_encoder'} and reference.config.get(key)!=value:
                        raise ContractError('baseline conditions changed: '+key)
                reference.run(rounds=0);reference.verify()
                if reference.config.get('semantic_retrieval'):
                    # Freeze the common first ranking (including its cost receipt)
                    # before copying graph/flat branches; later trajectories diverge.
                    reference.context('evidence',read_json(reference.state_path))
            finally:reference.store.close()
            pair={};shared=None
            # Alternate execution order across independent cases; record it explicitly.
            order=['graph','flat'] if index%2==0 else ['flat','graph']
            for mode in order:
                ancestry=fork_baseline(baseline,parent/mode,mode);shared=ancestry['shared_cost']
                branch=copy.deepcopy(case);branch['config']['memory_presentation']=mode
                agent=agent_factory(case,mode) if agent_factory else None
                pair[mode]=episode(branch,suite['baseline_policy'],parent/mode,inputs,agent=agent)
                pair[mode]['incremental_cost']=pair[mode]['charged_cost']-shared
            audit=audit_pair(parent/'graph',parent/'flat')
            first=[p for p in audit['pairs'] if p['role']=='evidence' and p['ordinal']==0]
            if len(first)!=1 or not first[0]['equal_visible_information']:
                raise ContractError('first decision did not receive identical information')
            sign=1 if config['protocol']['direction']=='maximize' else -1
            rows.append({'evaluation_id':case['id'],'order':order,'shared_baseline_cost':shared,
                'delta_graph_minus_flat':sign*(pair['graph']['score']-pair['flat']['score']),
                'input_audit':audit,**pair})
        if not rows:raise ContractError('no evaluation cases')
        report={'schema':'memory-online-comparison-v1','online':agent_factory is None,
            'evaluation_mode':'end_to_end_candidate_generation',
            'binding':binding,'rows':rows,
            'total_cost':sum(r['shared_baseline_cost']+sum(r[m]['incremental_cost'] for m in ['graph','flat']) for r in rows),
            'qualification':'Shared audited pre-decision baseline; independent subsequent trajectories. Later input divergence is retained. One task family; individual stochastic runs do not establish general utility.'}
        write_json(output,report);return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--stage',choices=['development','confirmation'],required=True)
    parser.add_argument('--context-byte-limit',type=int,required=True)
    args=parser.parse_args();run_validation(args.suite,args.output,args.stage,args.context_byte_limit)


if __name__=='__main__':main()
