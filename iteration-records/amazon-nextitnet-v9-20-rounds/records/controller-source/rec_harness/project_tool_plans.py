"""Predeclared independent validation plans for successive project tools."""
import copy
from .core import ContractError,file_hash,required,number
from .tool_method import freeze_plan
from .evolution import freeze_validator_files


def plans(config):
    return [config['tool']['plan']]+config.get('tool_followup_plans',[])


def freeze(config):
    if 'tool_followup_plans' not in config:return
    followups=config['tool_followup_plans']
    limit=config.get('max_tool_updates',1)
    if ('tool' not in config or not isinstance(followups,list) or
            type(limit) is not int or limit<1 or len(followups)!=limit-1 or
            any(not isinstance(plan,dict) for plan in followups)):
        raise ContractError('tool_followup_plans must contain one complete plan for each subsequent tool update')
    config['tool_followup_plans']=[freeze_plan(plan) for plan in followups]
    from .evolution import STAGES
    for plan in plans(config):
        required(plan,['budget','agent_cost','development_ids','identity','stages','rollback_version'])
        required(plan['identity'],['base_llm','tools','start','graph','budget','protocol'])
        if set(plan['stages'])!=set(STAGES['tool']):raise ContractError('scheduled tool plan requires all validation stages')
        for name in ['budget','agent_cost']:number(plan[name],name,0)
        for rule in plan['stages'].values():
            required(rule,['command','timeout','cost','minimum_gain','maximum_regression','maximum_total_cost'])
            for name in ['timeout','cost','minimum_gain','maximum_regression','maximum_total_cost']:number(rule[name],name,0)
        if not isinstance(plan['development_ids'],list) or any(not isinstance(item,str) or not item for item in plan['development_ids']):
            raise ContractError('scheduled tool development IDs must be strings')



def preflight(config):
    from .validation_preflight import preflight as check
    previous=set();hashes=[]
    for plan in plans(config):
        plan['development_ids']=sorted(set(plan['development_ids'])|previous)
        result=check(plan)
        previous.update(plan['development_ids'])
        previous.update(identity for row in result['checks'] for identity in row.get('evaluation_ids',[]))
        hashes.append(freeze_validator_files(plan['stages']))
    config['tool_plan_validator_hashes']=hashes


def select(config,index,*,verify=False):
    if 'tool_followup_plans' not in config:return copy.deepcopy(config['tool']['plan'])
    schedule=plans(config)
    if type(index) is not int or not 0<=index<len(schedule):raise ContractError('tool validation plan schedule exhausted')
    if verify:
        frozen=config['tool_plan_validator_hashes'][index]
        for name,expected in frozen.items():
            if file_hash(name)!=expected:raise ContractError('scheduled tool validator changed: '+name)
    return copy.deepcopy(schedule[index])
