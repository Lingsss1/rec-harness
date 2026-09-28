"""Single-entry creation or resumption of a frozen autonomous research campaign."""
import os
from pathlib import Path

from .core import ContractError,read_json
from .runtime import Research


def resolve_plan_paths(plan,base):
    """Resolve declared files; do not guess the meaning of custom argv values."""
    from .validation_preflight import SUPPORTED
    for rule in plan.get('stages',{}).values():
        if 'validator_files' in rule:
            rule['validator_files']=[str((base/value).resolve()) for value in rule['validator_files']]
        command=rule.get('command',[])
        if '-m' in command:
            index=command.index('-m')+1
            if index<len(command) and command[index] in SUPPORTED and '--suite' in command:
                index=command.index('--suite')+1
                if index<len(command) and '{' not in command[index]:
                    command[index]=str((base/command[index]).resolve())
        # A Python script argument is a declared entrypoint, unlike arbitrary flags.
        if len(command)>1 and command[1].endswith('.py') and '{' not in command[1]:
            command[1]=str((base/command[1]).resolve())
        if command and '/' in command[0] and '{' not in command[0]:
            command[0]=str((base/command[0]).resolve())


def configuration(path):
    path=Path(path).resolve()
    def expand(value):
        if isinstance(value,dict):return {k:expand(v) for k,v in value.items()}
        if isinstance(value,list):return [expand(v) for v in value]
        if isinstance(value,str) and '${' in value:
            result=os.path.expandvars(value)
            if '${' in result:raise ContractError('unset configuration environment variable: '+value)
            return result
        return value
    config=expand(read_json(path))
    if config.get('adapter') not in {None,'amazon-nextitnet-v1','research-evolution-v1','research-project-v1'}:
        raise ContractError('unsupported autoresearch adapter')
    if config.get('adapter')=='research-project-v1':
        for kind in ('model','tool','harness'):
            key=kind+'_config'
            if key in config:config[kind]=configuration(path.parent/config.pop(key))
    for key in ('source','study'):
        if key in config:config[key]=str((path.parent/config[key]).resolve())
    backend=config.get('agent_backend',{})
    for key in ('template','runtime_root'):
        if key in backend:backend[key]=str((path.parent/backend[key]).resolve())
    if 'uid_registry' in config:config['uid_registry']=str((path.parent/config['uid_registry']).resolve())
    if 'plan' in config:resolve_plan_paths(config['plan'],path.parent)
    if 'tool_followup_plans' in config:
        if not isinstance(config['tool_followup_plans'],list):raise ContractError('tool_followup_plans must be a list')
        for plan in config['tool_followup_plans']:
            if not isinstance(plan,dict):raise ContractError('tool followup plan must be an object')
            resolve_plan_paths(plan,path.parent)
    def execution_recipe(execution):
        if 'uid_registry' in execution:execution['uid_registry']=str((path.parent/execution['uid_registry']).resolve())
        if 'recipe_file' in execution:
            if 'recipe' in execution:raise ContractError('supply recipe or recipe_file, not both')
            execution['recipe']=read_json((path.parent/execution.pop('recipe_file')).resolve())
    execution_recipe(config.get('execution',{}))
    # Each nested configuration resolves its own paths before project assembly;
    # project-owned tool runtimes use the project configuration directory.
    for key in ('diagnostic_registry','tool_runtime','activation_diagnostic'):
        if key in config:execution_recipe(config[key].get('execution',{}))
    if 'harness_registry' in config:config['harness_registry']=str((path.parent/config['harness_registry']).resolve())
    if 'diagnostic_registry' in config:
        config['diagnostic_registry']['registry']=str((path.parent/config['diagnostic_registry']['registry']).resolve())
    if 'measurement_source' in config:
        config['measurement_source']['campaign']=str((path.parent/config['measurement_source']['campaign']).resolve())
    if 'optimizer_measurement_source' in config:
        config['optimizer_measurement_source']=str((path.parent/config['optimizer_measurement_source']).resolve())
    if 'trace_source' in config:
        config['trace_source']['campaign']=str((path.parent/config['trace_source']['campaign']).resolve())
    for key in ['trace_sources','tool_history_sources']:
        if key not in config:continue
        if not isinstance(config[key],list):raise ContractError(key+' must be a list')
        for source in config[key]:
            source['campaign']=str((path.parent/source['campaign']).resolve())
    for key in ('development_memory_files','development_history','initial_memory_files'):
        if key in config:config[key]=[str((path.parent/value).resolve()) for value in config[key]]
    if 'rescore_recipe_file' in config:
        config['rescore_recipe']=read_json((path.parent/config.pop('rescore_recipe_file')).resolve())
    return config


def autoresearch(root,config_file=None,*,agent=None):
    root=Path(root).resolve()
    from .adapters.amazon_autonomous import AmazonAutoresearch
    from .evolution_auto import AutonomousEvolution
    from .project import ResearchProject
    amazon=(root/'amazon-auto.json').exists()
    evolving=(root/'evolution-auto.json').exists()
    project=(root/'project.json').exists()
    implementation=ResearchProject if project else AmazonAutoresearch if amazon else AutonomousEvolution if evolving else Research
    if not amazon and not evolving and not project and not (root/'campaign.json').exists():
        if config_file is None:raise ContractError('new campaign requires --config')
        config=configuration(config_file);amazon=config.get('adapter')=='amazon-nextitnet-v1'
        evolving=config.get('adapter')=='research-evolution-v1'
        project=config.get('adapter')=='research-project-v1'
        implementation=ResearchProject if project else AmazonAutoresearch if amazon else AutonomousEvolution if evolving else Research
        implementation.init(root,config)
    elif config_file is not None:
        raise ContractError('campaign already frozen; resume without --config')
    research=implementation(root,agent=agent)
    try:return research.run()
    finally:research.store.close()
