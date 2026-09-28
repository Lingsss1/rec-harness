"""Development-only activation summaries for Amazon research role inputs."""
from pathlib import Path
from rec_harness.activation_runtime import run_probe
from rec_harness.core import ContractError,digest,number,public_only,write_json,read_json,Artifacts
from rec_harness.diagnostics import validate_measurement
from rec_harness.runtime import node
from rec_harness.store import Store


def validate_options(options,study):
    if options is None:return
    required={'settings','execution','budget','cost','timeout'}
    if not isinstance(options,dict) or not required<=set(options) or set(options)-required-{'datasets','seeds','history_sampling'}:
        raise ContractError('invalid activation_diagnostic configuration')
    for key in ['budget','cost','timeout']:number(options[key],key,0)
    if options['cost']>options['budget'] or not 0<options['timeout']<=120:raise ContractError('invalid activation cost or timeout')
    datasets=options.get('datasets',study['datasets']);seeds=options.get('seeds',study['development_seeds'])
    if not datasets or len(set(datasets))!=len(datasets) or not set(datasets)<=set(study['datasets']):raise ContractError('activation datasets outside study')
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds))!=len(seeds) or not set(seeds)<=set(study['development_seeds']) or set(seeds)&set(study['confirmation_seeds']):
        raise ContractError('activation diagnostics require development seeds only')


def cost_summary(root):
    total=0;ledgers=[]
    for path in sorted((Path(root)/'activation-probes').glob('*/memory')):
        store=Store(path)
        try:
            store.verify();amount=store.budget()['charged_or_reserved'];total+=amount
            ledgers.append({'directory':str(path),'charged_or_reserved':amount})
        finally:store.close()
    return {'charged_or_reserved':total,'unit':'BCU','ledgers':ledgers,'accounting':'sum of child probe ledgers, not additional charges'}


def activation_context(controller,base,study):
    options=controller.config.get('activation_diagnostic');validate_options(options,study)
    if options is None:return {}
    rows=[]
    for dataset in options.get('datasets',study['datasets']):
        for seed in options.get('seeds',study['development_seeds']):
            config={**{k:options[k] for k in ['settings','execution','cost','timeout']},'budget':options['cost'],
                'source':study['model_versions'][base],
                'checkpoint_source':{'study':str(controller.study),'model_version':base,'dataset':dataset,'seed':seed}}
            if 'history_sampling' in options:config['history_sampling']=options['history_sampling']
            identity=digest(config);directory=controller.root/'activation-probes'/identity
            if not (directory/'memory/memory.sqlite').exists() and cost_summary(controller.root)['charged_or_reserved']+options['cost']>options['budget']:
                return {'activation_diagnostics':rows,'measurement_failure':'activation diagnostic budget unavailable'}
            result=run_probe(config,directory)
            if result['status']!='success':
                return {'activation_diagnostics':rows,'measurement_failure':'activation diagnostic unavailable; inspect local probe receipt'}
            measurement=validate_measurement(result['measurement'])
            observation='activation-'+identity
            # Public projection is copied separately; private weight/source archives stay local.
            public={'observation_id':observation,'model_version':base,'dataset':dataset,'seed':seed,
                'measurements':measurement['measurements'],'limitations':measurement['limitations'],
                'scope':('trained development model on sampled training histories' if 'history_sampling' in options else
                         'trained development model on synthetic histories')+'; not recommendation accuracy or mechanism proof'}
            public_only(public)
            path=controller.root/'research_archive/activation-summaries'/(identity+'.json');write_json(path,public)
            ref=controller.artifacts.put(path);known={n['id'] for n in controller.store.snapshot()['nodes']}
            if observation not in known:
                controller.store.patch({'base_version':controller.store.version,'nodes':[
                    node(observation,'Observation',{**public,'summary_ref':ref,'valid':True},controller.task,controller.protocol_hash)],'edges':[]},
                    accompanying_events=[('activation_summary_recorded',{'observation_id':observation,'summary_ref':ref,'model_version':base,'dataset':dataset,'seed':seed})])
            rows.append(public)
    return {'activation_diagnostics':rows}


def audit_costs(root,options):
    """Read-only reconciliation of the child probe ledgers and frozen options."""
    from rec_harness.cost_audit import audit_cost
    root=Path(root);ledgers=[];total=0;pending=0
    controller=read_json(root/'amazon-auto.json')
    if digest(controller)!=read_json(root/'auto-lock.json')['config_hash'] or controller.get('activation_diagnostic')!=options:
        raise ContractError('activation controller options changed')
    from .amazon_controls import load_config
    study=load_config(controller['study']);validate_options(options,study)
    for memory in sorted((root/'activation-probes').glob('*/memory')):
        folder=memory.parent;config=read_json(folder/'config.json');binding=read_json(folder/'binding.json')
        expected={**{k:options[k] for k in ['settings','execution','cost','timeout']},'budget':options['cost']}
        if 'history_sampling' in options:expected['history_sampling']=options['history_sampling']
        if ({k:v for k,v in config.items() if k not in {'source','checkpoint_source'}}!=expected
                or binding['config_hash']!=digest(config) or folder.name!=digest(config)
                or config['checkpoint_source']['study']!=controller['study']):
            raise ContractError('activation probe differs from frozen options')
        source=config['checkpoint_source']
        if (source['dataset'] not in options.get('datasets',study['datasets'])
                or source['seed'] not in options.get('seeds',study['development_seeds'])
                or source['model_version'] not in study['model_versions']
                or config['source']!=study['model_versions'][source['model_version']]):
            raise ContractError('activation probe condition differs')
        store=Store(memory,read_only=True)
        try:
            cost=audit_cost(store,options['cost'])
            completed=[e['payload'] for e in store.events() if e['kind']=='activation_probe_completed']
            if len(completed)!=1 or completed[0]!=read_json(folder/'result.json') or completed[0]['binding']!=binding:
                raise ContractError('activation probe lacks a committed result')
            objects=Artifacts(folder/'objects')
            for ref in completed[0]['artifacts'].values():objects.verify(ref)
            if completed[0]['status']=='success':
                from rec_harness.activation_checkpoint import resolve_checkpoint
                if resolve_checkpoint(config)['sha256']!=binding['checkpoint_hash']:raise ContractError('activation checkpoint binding changed')
                measurement=read_json(objects.verify(completed[0]['artifacts']['measurement.json']))
                if measurement!=completed[0]['measurement']:raise ContractError('activation measurement archive changed')
        finally:store.close()
        total+=cost['charged_or_reserved'];pending+=cost['pending']
        ledgers.append({'directory':str(memory.resolve()),'cost':cost})
    if not ledgers:raise ContractError('configured activation probes have no committed ledgers')
    if total>options['budget']:raise ContractError('activation probes exceed frozen budget')
    return {'charged_or_reserved':total,'pending':pending,'ledgers':ledgers,
            'limit':options['budget'],'unit':'activation BCU','accounting':'sum of child probe ledgers; not additional charges'}
