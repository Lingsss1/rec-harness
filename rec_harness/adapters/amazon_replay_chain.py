"""Serial observed campaigns linked by committed cost and development lineage."""
from pathlib import Path

from rec_harness.core import Artifacts,ContractError,digest,read_json,write_json
from rec_harness.replay import build_world
from rec_harness.store import Store
from .amazon_replay import archive_cycle


def validate_lineage(lineage,previous,previous_campaign,previous_study):
    if Path(lineage['role_predecessor']).resolve()!=previous_campaign or Path(lineage['gpu_predecessor']).resolve()!=previous_study:
        raise ContractError('lineage does not identify preceding campaign and study')
    for kind,head,amount in [('role',previous['source']['controller_head'],previous['resources']['campaign_cli_calls']),
                             ('gpu',previous['source']['study_head'],previous['resources']['retained_gpu_seconds'])]:
        cost=lineage[kind+'_cost']
        if cost['integrity']['head']!=head or cost['charged_or_reserved']!=amount or cost['pending']:
            raise ContractError('lineage cost or source head differs')


def compose_chain(worlds,artifacts):
    """Compose already lineage-verified cycles; every prior transition is required."""
    if len(worlds)<2:raise ContractError('at least two observed cycles required')
    nodes=[];prior=None;totals=[]
    for index,world in enumerate(worlds):
        if world['hash']!=digest({k:v for k,v in world.items() if k!='hash'}):
            raise ContractError('cycle world changed')
        if world.get('schema')!='amazon-observed-cycle-world-v1' or world['qualification']!='replay_ready' or len(world['nodes'])!=1:
            raise ContractError('complete single-cycle worlds required')
        if world['cost_model']!=worlds[0]['cost_model']:raise ContractError('chain cost model differs')
        node=dict(world['nodes'][0]);resources=world['resources'];weights=world['cost_model']
        total=resources['retained_gpu_seconds']*weights['gpu_second_weight']+resources['campaign_cli_calls']*weights['role_call_weight']
        if prior:
            previous=prior['nodes'][0]
            if resources['retained_gpu_seconds']<prior['resources']['retained_gpu_seconds'] or resources['campaign_cli_calls']<prior['resources']['campaign_cli_calls']:
                raise ContractError('cumulative chain cost decreased')
            visible=read_json(artifacts.verify(node['request_ref']))
            if previous['observation']['development_hash'] not in {r.get('hash') for r in visible.get('history',[])}:
                raise ContractError('predecessor readout absent from next design context')
            expected=previous['workspace_after'] if previous['observation']['selection']=='keep' else previous['workspace_before']
            if node['workspace_before']!=expected or world['root_observation']['score']!=previous['observation']['score']:
                raise ContractError('chain incumbent or visible baseline differs')
            node['cost']=total-totals[-1]
        node.update(primary_parent_id=nodes[-1]['node_id'] if nodes else 'root',creation_seq=index+1,
                    dependencies=['root']+[n['node_id'] for n in nodes])
        nodes.append(node);totals.append(total);prior=world
    chain=build_world(nodes,artifacts=artifacts,workers=1,root_observation=worlds[0]['root_observation'],initial_cost=worlds[0]['initial_cost'])
    chain.update(schema='amazon-observed-chain-world-v1',online_utility_evidence=False,
        cycle_world_hashes=[w['hash'] for w in worlds],cost_model=worlds[0]['cost_model'],
        cumulative_costs=totals,
        support={'actions':['continue next observed cycle','stop'],
                 'unsupported':['new prompt','new tool','new code','alternative within-cycle decision','parallel scheduling'],
                 'scope':'Fixed serial recorded contexts; inherited costs charged once. No online utility, confirmation or final-test evidence.'})
    chain['hash']=digest({k:v for k,v in chain.items() if k!='hash'})
    return chain


def archive_chain(campaigns,output,*,gpu_weight,role_weight):
    campaigns=[Path(p).resolve() for p in campaigns];output=Path(output).resolve()
    if len(campaigns)<2 or len(set(campaigns))!=len(campaigns):raise ContractError('distinct ordered campaigns required')
    artifacts=Artifacts(output/'artifacts/objects');worlds=[];links=[]
    for index,campaign in enumerate(campaigns):
        directory=output/'cycles'/str(index)
        world=archive_cycle(campaign,directory,gpu_weight=gpu_weight,role_weight=role_weight)
        local=Artifacts(directory/'artifacts/objects')
        for path in local.root.iterdir():
            if path.is_file() and len(path.name)==64:
                ref=artifacts.put(path)
                if ref['sha256']!=path.name:raise ContractError('cycle artifact filename mismatch')
        if index:
            store=Store(campaign/'memory',read_only=True)
            try:
                store.db.execute('BEGIN')
                if store.verify()['head']!=world['source']['controller_head']:raise ContractError('chain source changed')
                events=[e['payload'] for e in store.events() if e['kind']=='next_campaign_lineage']
                if len(events)!=1:raise ContractError('unique committed campaign lineage required')
                event=events[0]
                source=Artifacts(campaign/'artifacts/objects').verify(event['artifact'])
                lineage=read_json(source)
                if digest(lineage)!=event['lineage_hash']:raise ContractError('chain lineage changed')
                previous=worlds[-1];previous_campaign=campaigns[index-1]
                previous_study=Path(read_json(previous_campaign/'amazon-auto.json')['study']).resolve()
                validate_lineage(lineage,previous,previous_campaign,previous_study)
                links.append(artifacts.put(source))
            finally:
                if store.db.in_transaction:store.db.execute('ROLLBACK')
                store.close()
        worlds.append(world)
    chain=compose_chain(worlds,artifacts)
    chain.update(lineage_refs=links,sources=[w['source'] for w in worlds])
    chain['hash']=digest({k:v for k,v in chain.items() if k!='hash'})
    for world,campaign in zip(worlds,campaigns):
        for directory,key in [(campaign,'controller_head'),(Path(read_json(campaign/'amazon-auto.json')['study']),'study_head')]:
            store=Store(directory/'memory',read_only=True)
            try:
                if store.verify()['head']!=world['source'][key]:raise ContractError('chain source changed during conversion')
            finally:store.close()
    path=output/'world.json'
    if path.exists() and read_json(path)!=chain:raise ContractError('frozen chain changed; use a new output')
    if not path.exists():write_json(path,chain)
    return chain
