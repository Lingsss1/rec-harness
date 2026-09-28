"""Preflight resuming online comparisons between completed research groups."""
from .core import ContractError,digest,read_json
from .cost_audit import audit_cost
from .store import Store


def completed_groups(target,module,stage):
    """Prove every initialized group is terminal before starting any new group.

    Mid-episode recovery is deliberately not inferred from a state projection:
    the model controller may turn an interrupted round into a failed attempt.
    Such a change would alter a paired comparison's frozen research trajectory.
    """
    amazon=target/('amazon-harness-online' if module=='rec_harness.amazon_harness_online' else 'amazon-tool-online')
    if module in {'rec_harness.amazon_tool_online','rec_harness.amazon_harness_online'} or module=='rec_harness.tool_pipeline' and amazon.exists():
        from .amazon_tool_online import read_result,GROUPS
        if module=='rec_harness.amazon_harness_online':
            from .amazon_harness_online import GROUPS
        for folder in sorted(amazon.glob('*/*')):
            if not folder.is_dir() or folder.parent.name in {'objects','candidate','existing'}:continue
            if folder.name not in GROUPS:raise ContractError('unexpected Amazon comparison directory')
            if not (folder/'completed.json').is_file():raise ContractError('Amazon comparison group outcome unknown')
            if read_json(folder/'completed.json')!=read_result(folder):raise ContractError('completed Amazon comparison changed')
        return
    tool=module in {'rec_harness.tool_online','rec_harness.tool_pipeline'}
    root=target/('tool-online-episodes' if tool else 'online-episodes')
    groups={'no_tool','existing_tool','candidate_tool'} if tool else {'baseline','candidate'}
    for folder in sorted(root.glob('*/*')):
        if not folder.is_dir():continue
        if folder.parent.name in {'objects','candidate-inputs','existing-inputs'}:continue
        if folder.name in {name+'-inputs' for name in groups}:continue
        if folder.name not in groups:raise ContractError('unexpected online comparison directory')
        config=read_json(folder/'campaign.json')
        manifest=read_json(folder/'research_archive/manifest.json')
        if manifest['config_hash']!=digest(config):raise ContractError('online recovery configuration changed')
        store=Store(folder/'memory',read_only=True)
        try:
            events=store.events();cost=audit_cost(store,config['budget'])
            stopped=[e['payload']['state'] for e in events if e['kind']=='controller_stopped']
            state=read_json(folder/'state.json')
            if (not stopped or state!=stopped[-1] or state.get('status')!='stopped'
                    or state.get('stop_reason') not in {'budget','max_rounds'} or cost['pending']):
                raise ContractError('online research group is not durably completed; refusing automatic continuation')
            if not (state.get('baseline') or {}).get('valid'):
                raise ContractError('online research group has no valid terminal incumbent')
        finally:store.close()
