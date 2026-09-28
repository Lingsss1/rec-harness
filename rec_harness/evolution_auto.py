"""Single-entry tool/Harness candidate lifecycle over independently supplied validators."""
from pathlib import Path
from .core import ContractError,digest,identifier,read_json,write_json
from .evolution import Evolution,STAGES
from .jailed_agent import JailedCodexAgent


class AutonomousEvolution:
    @staticmethod
    def init(root,configuration):
        root=Path(root).resolve()
        if root.exists() and any(root.iterdir()):raise ContractError('evolution campaign directory must be empty')
        config=dict(configuration);identifier(config['candidate_id'])
        if config['kind'] not in STAGES:raise ContractError('tool or harness evolution required')
        if 'trace_source' in config:
            from .decision_traces import collect_traces
            if config['kind']!='harness' or config.get('context') or 'measurement_source' in config:raise ContractError('trace trigger requires a Harness candidate without overriding context')
            source=config.pop('trace_source')
            config['context']=collect_traces(source['campaign'],roles=source.get('roles'),cutoff_seq=source.get('cutoff_seq'))
            if config['context']['source']['task_id'] not in config['plan']['development_ids']:
                raise ContractError('Harness validation must exclude the trace source development task')
        if 'measurement_source' in config:
            from .measurement_gaps import evolution_context
            if config['kind']!='tool' or config.get('context'):raise ContractError('measurement trigger requires a tool candidate without an overriding context')
            source=config.pop('measurement_source')
            config['context']=evolution_context(source['campaign'],source['gap_id'])
            ids=set(config['plan']['development_ids'])
            if not set(config['context']['measurement_gap']['experiment_refs']).issubset(ids):
                raise ContractError('tool validation must exclude the measurement trigger development experiments')
        if 'optimizer_measurement_source' in config:
            from .optimizer_handoff import load_context
            if config['kind']!='tool' or config.get('context'):raise ContractError('optimizer handoff requires a tool without overriding context')
            config['context']=load_context(config.pop('optimizer_measurement_source'))
            if config['context']['source']['task_id'] not in config['plan']['development_ids']:
                raise ContractError('tool validation must exclude optimizer source task')
        if config['kind']=='tool':
            from .diagnostics import CONTRACT,runtime_contract
            if config['plan'].get('tool_contract')==CONTRACT:
                context=config.get('context',{});contract=runtime_contract()
                if 'diagnostic_contract' in context and context['diagnostic_contract']!=contract:
                    raise ContractError('diagnostic authoring contract differs from runtime')
                config['context']={**context,'diagnostic_contract':contract}
        root.mkdir(parents=True,exist_ok=True);evolution=Evolution(root)
        try:
            if 'source' in config:
                config['source_snapshot']=evolution.artifacts.snapshot(config.pop('source'),config.pop('model_files'))
            if config.get('tool_predecessor'):
                if config['kind']!='tool' or digest(config.get('source_snapshot'))!=config['tool_predecessor']['implementation_hash']:
                    raise ContractError('tool predecessor differs from inherited implementation')
                config['context']={**config['context'],'tool_predecessor':config['tool_predecessor']}
            if 'harness_base_policy' in config:
                from .evolution import validate_policy
                if config['kind']!='harness':raise ContractError('base policy requires Harness evolution')
                validate_policy(config['harness_base_policy'])
                config['context']={**config['context'],'base_policy':config['harness_base_policy'],
                    'patch_semantics':'Apply patch_diff to base_policy: replace supplied top-level fields, merge supplied prompt roles, preserve omitted fields. Validators evaluate the complete resulting policy.'}
            if 'optimizer_handoff_budget' in config:
                from .optimizer_handoff import request
                from .agents import schema_for
                if config['kind']!='harness':raise ContractError('optimizer routing requires Harness evolution')
                config['context']['measurement_handoff_request']=request(config['context'],config['optimizer_handoff_budget'])
                config['context']['optimizer_payload_schemas']={'harness_patch':schema_for('optimizer'),'measurement_gap':schema_for('measurement')}
            if config['kind']=='harness' and config.setdefault('compact_optimizer_traces',True):
                from .trace_context import compact_traces,expand_traces
                original=config['context'];config['context']=compact_traces(original)
                if expand_traces(config['context'])!=original:raise ContractError('trace sharing changed optimizer evidence')
            write_json(root/'evolution-auto.json',config)
            write_json(root/'auto-lock.json',{'config_hash':digest(config)})
            write_json(root/'state.json',{'candidate_id':config['candidate_id'],'kind':config['kind'],'status':'ready'})
        finally:evolution.store.close()

    def __init__(self,root,agent=None):
        self.root=Path(root).resolve();self.config=read_json(self.root/'evolution-auto.json')
        if digest(self.config)!=read_json(self.root/'auto-lock.json')['config_hash']:raise ContractError('evolution configuration changed')
        self.evolution=Evolution(root);self.store=self.evolution.store
        options=self.config.get('agent_backend',{})
        self.agent=agent or JailedCodexAgent(options['template'],options['runtime_root'],
                                            model=options.get('model'),timeout=options.get('timeout',300),web_search=options.get('web_search'))

    def run(self):
        c=self.config
        result=self.evolution.run(c['candidate_id'],c['kind'],c['context'],c['plan'],
                                   agent=self.agent,source=c.get('source_snapshot'))
        write_json(self.root/'run-report.json',result)
        write_json(self.root/'state.json',result)
        return result
