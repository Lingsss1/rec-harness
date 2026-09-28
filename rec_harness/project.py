"""Sequential model/tool/Harness episodes with explicit frozen activation boundaries."""
import copy
import fcntl
import tempfile
from contextlib import contextmanager
from pathlib import Path
from .core import Artifacts,ContractError,digest,identifier,read_json,write_json
from .process_identity import assert_stopped
from .store import Store


class ResearchProject:
    @staticmethod
    def init(root,configuration):
        root=Path(root).resolve();config=copy.deepcopy(configuration)
        config.setdefault('measurement_lineage',True)
        config.setdefault('inherit_tool_source',True)
        config.setdefault('inherit_tool_history',True)
        if type(config['inherit_tool_history']) is not bool:raise ContractError('tool history inheritance must be boolean')
        config.setdefault('inherit_harness_policy',True)
        config.setdefault('optimizer_measurement_handoff',True)
        config.setdefault('enforce_tool_measurement_scope',True)
        config.setdefault('verify_successor_memory',True)
        config.setdefault('continue_after_harness_rejection',True)
        if type(config['continue_after_harness_rejection']) is not bool:raise ContractError('Harness rejection continuation must be boolean')
        if type(config['verify_successor_memory']) is not bool:raise ContractError('successor memory verification must be boolean')
        if type(config['enforce_tool_measurement_scope']) is not bool:raise ContractError('tool measurement scope enforcement must be boolean')
        if 'tool' in config:
            from .tool_method import freeze_plan
            config['tool']['plan']=freeze_plan(config['tool']['plan'])
        scope=config.get('tool',{}).get('plan',{}).get('measurement_scope')
        if scope is not None:
            outputs=scope.get('supported_outputs') if isinstance(scope,dict) else None
            if (not isinstance(outputs,dict) or not outputs
                or not all(isinstance(k,str) and k.strip() and isinstance(v,str) and v.strip() for k,v in outputs.items())):
                raise ContractError('tool measurement scope requires named outputs with definitions')
        identifier(config['task_id'])
        if config['model'].get('adapter') not in {None,'amazon-nextitnet-v1'}:raise ContractError('unsupported project model adapter')
        if config['model'].get('adapter')=='amazon-nextitnet-v1':
            config['amazon_study_hash']=digest(read_json(Path(config['model']['study'])/'study.json'))
        if type(config.get('max_model_episodes')) is not int or config['max_model_episodes']<1:raise ContractError('positive max_model_episodes required')
        for kind in ['tool','harness']:
            if kind in config and config[kind]['kind']!=kind:raise ContractError('evolution kind differs from project slot')
            limit=config.get('max_'+kind+'_updates',1)
            if type(limit) is not int or limit<0:raise ContractError('nonnegative evolution update limit required')
        harness=config.get('harness',{})
        if 'trace_source' in harness and 'trace_sources' in harness:
            raise ContractError('supply trace_source or trace_sources, not both')
        if 'trace_sources' in harness and config.get('max_harness_updates',1)>0 and config['max_model_episodes']>1:
            from .decision_traces import collect_trace_set
            # Verify external history before spending calls on the first model.
            # Evolution still freezes its own prefix when that episode starts.
            collect_trace_set(harness['trace_sources'])
        if (config.get('tool',{}).get('tool_history_sources') is not None
                and config.get('max_tool_updates',1)>0 and config['max_model_episodes']>1):
            from .tool_history import collect
            history,excluded=collect(config['tool']['tool_history_sources'])
            config['tool']['tool_history_sources']=[{**spec,'cutoff_seq':binding['cutoff_seq'],'event_head':binding['event_head']}
                for spec,binding in zip(config['tool']['tool_history_sources'],history['sources'])]
            config['tool']['plan']['development_ids']=sorted(set(config['tool']['plan']['development_ids'])|set(excluded))
        if 'tool_runtime' in config:
            from .diagnostics import validate_runtime
            validate_runtime(config['tool_runtime'])
        elif 'tool' in config and config.get('max_tool_updates',1)>0 and config['max_model_episodes']>1:
            raise ContractError('tool_runtime required for post-validation remeasurement')
        if 'tool' in config and not config['model'].get('measurement_requirements'):
            config['model'].setdefault('measurement_discovery',{'development_budget':config['tool']['plan']['budget']})
        from .validation_preflight import preflight
        for kind in ['tool','harness']:
            if kind in config and config.get('max_'+kind+'_updates',1)>0 and config['max_model_episodes']>1:
                preflight(config[kind]['plan'])
        if root.exists() and any(root.iterdir()):raise ContractError('project directory must be empty')
        root.mkdir(parents=True,exist_ok=True);artifacts=Artifacts(root/'objects')
        for kind in ['model','tool','harness']:
            if kind not in config:continue
            template=config[kind]
            if 'source' in template:
                files=list(dict.fromkeys(template['model_files']+template.get('evaluator_files',[])))
                template['source_snapshot']=artifacts.snapshot(template.pop('source'),files)
            if template.get('initial_memory_files'):
                refs=[artifacts.put(path) for path in template.pop('initial_memory_files')]
                template['initial_memory_refs']=refs
            if template.get('development_memory_files'):
                template['development_memory_refs']=[artifacts.put(path) for path in template.pop('development_memory_files')]
        write_json(root/'project.json',config);write_json(root/'project-lock.json',{'config_hash':digest(config)})
        store=Store(root/'memory');store.configure_budget(0)
        state={'phase':'model','model_index':1,'tool_updates':0,'harness_updates':0,'status':'ready'}
        store.event('project_transition',{'state':state});write_json(root/'state.json',state);store.close()

    def __init__(self,root,agent=None):
        self.root=Path(root).resolve();self.config=read_json(self.root/'project.json')
        if digest(self.config)!=read_json(self.root/'project-lock.json')['config_hash']:raise ContractError('frozen project changed')
        self.store=Store(self.root/'memory');self.artifacts=Artifacts(self.root/'objects');self.agent=agent

    def transition(self,state,episode=None):
        self.store.event('project_transition',{'state':state,**({'episode':episode} if episode else {})});write_json(self.root/'state.json',state)

    def template(self,kind,directory):
        config=copy.deepcopy(self.config[kind]);snapshot=config.pop('source_snapshot',None)
        if snapshot:
            self.artifacts.restore(snapshot,directory/'inputs');config['source']=str(directory/'inputs')
        refs=config.pop('initial_memory_refs',[])
        if refs:config['initial_memory_files']=[str(self.artifacts.verify(ref)) for ref in refs]
        refs=config.pop('development_memory_refs',[])
        if refs:config['development_memory_files']=[str(self.artifacts.verify(ref)) for ref in refs]
        return config

    def child(self,directory,config):
        from .autonomous import autoresearch
        path=directory/'episode-config.json';campaign=directory/'campaign'
        if path.exists() and read_json(path)!=config:raise ContractError('project child configuration changed')
        write_json(path,config)
        frozen=any((campaign/name).exists() for name in ['campaign.json','amazon-auto.json','evolution-auto.json'])
        if not frozen and config.get('adapter')=='research-evolution-v1':
            # Parent plans were frozen at project creation. Do not apply new
            # defaults to a legacy project's not-yet-created tool episode.
            from .autonomous import configuration
            from .evolution_auto import AutonomousEvolution
            AutonomousEvolution.init(campaign,configuration(path),_freeze_tool_method=False)
            frozen=True
        return autoresearch(campaign,None if frozen else path,agent=self.agent)

    def model_config(self,state,directory):
        config=self.template('model',directory);config['task_id']=self.config['task_id']+'-model-'+str(state['model_index'])
        if config.get('adapter')=='amazon-nextitnet-v1' and digest(read_json(Path(config['study'])/'study.json'))!=self.config['amazon_study_hash']:
            raise ContractError('project-bound Amazon study changed')
        if state.get('last_model'):
            previous=self.root/state['last_model'];old=read_json(previous/'state.json')
            recorded=[e['payload']['episode'] for e in self.store.events() if e['kind']=='project_transition' and e['payload'].get('episode',{}).get('campaign')==str(previous)]
            report=read_json(previous/'run-report.json')
            if not recorded or digest(report)!=recorded[-1]['result_hash'] or old!=report['state']:raise ContractError('parent model result differs from committed project episode')
            if self.config.get('verify_successor_memory'):
                parent_store=Store(previous/'memory',read_only=True)
                try:
                    if parent_store.verify()!=report['integrity']:
                        raise ContractError('parent model history differs from committed project episode')
                finally:parent_store.close()
            package=directory/'source-memory.json'
            if config.get('adapter')=='amazon-nextitnet-v1':
                from .adapters.amazon_memory import export_development_memory
                source_config=read_json(previous/'amazon-auto.json')
                if Path(source_config['study']).resolve()!=Path(config['study']).resolve():raise ContractError('Amazon successor must retain the bound study')
                config['base_model_id']=old['incumbent']
                self.successor_memory(package,lambda path:export_development_memory(previous,path))
                config['development_memory_files']=config.get('development_memory_files',[])+[str(package)]
            else:
                if set(old['workspace']) & set(config.get('evaluator_files',[])):raise ContractError('inherited model overlaps frozen evaluator')
                Artifacts(previous/'artifacts/objects').restore(old['workspace'],directory/'inputs')
                config['model_files']=list(old['workspace'])
                from .memory_policy import export_memory
                source_config=read_json(previous/'campaign.json');store=Store(previous/'memory')
                try:
                    self.successor_memory(package,lambda path:export_memory(store,path,task_id=source_config['task_id'],protocol_hash=source_config['protocol_hash']))
                finally:store.close()
                config['initial_memory_files']=config.get('initial_memory_files',[])+[str(package)]
        requirements={r['measurement']:r for r in config.get('measurement_requirements',[])}
        for gap in state.get('measurement_obligations',[]):
            rule={'measurement':gap['required_information'],**{k:gap[k] for k in ['blocked_decision','competing_explanations','cheapest_test','development_budget','acceptance_contract']}}
            previous=requirements.get(rule['measurement'])
            if previous is not None and {k:v for k,v in previous.items() if k!='source_measurement_gap_ids'}!=rule:raise ContractError('successor measurement contract conflicts with promoted tool obligation')
            if self.config.get('measurement_lineage'):
                rule['source_measurement_gap_ids']=sorted(set((previous or {}).get('source_measurement_gap_ids',[]))|{gap['measurement_gap_id']})
            requirements[rule['measurement']]=rule
        if requirements:config['measurement_requirements']=list(requirements.values())
        if state.get('active_tool'):
            config['diagnostic_registry']={**self.config['tool_runtime'],'registry':str(self.root/state['active_tool']),
                'identity':self.config['tool']['plan']['identity']}
        if state.get('active_harness'):
            config.pop('research_policy',None)
            config['harness_registry']=str(self.root/state['active_harness'])
            config['harness_identity']=self.config['harness']['plan']['identity']
        return config

    def successor_memory(self,package,export):
        """Reconcile an interrupted export with its source before child admission."""
        if not package.exists():
            export(package)
        elif self.config.get('verify_successor_memory'):
            with tempfile.TemporaryDirectory(prefix='verify-memory-',dir=package.parent) as temporary:
                expected=Path(temporary)/'expected.json'
                export(expected)
                if read_json(package)!=read_json(expected):
                    raise ContractError('successor memory differs from committed parent export')

    def evolve_config(self,state,directory,kind):
        config=self.template(kind,directory);config['adapter']='research-evolution-v1'
        config['candidate_id']=self.config['task_id']+'-'+kind+'-'+str(state[kind+'_updates']+1)
        config.pop('context',None);source=self.root/state['last_model']
        config['plan']['development_ids']=list(config['plan']['development_ids'])
        if kind=='tool':
            if self.config.get('inherit_tool_history'):
                from .tool_history import project_sources
                sources=list(config.get('tool_history_sources',[]))
                known={Path(item['campaign']).resolve() for item in sources}
                sources.extend(item for item in project_sources(self) if Path(item['campaign']).resolve() not in known)
                if sources:config['tool_history_sources']=sources
            config['measurement_source']={'campaign':str(source),'gap_id':state['pending_gap']['measurement_gap_id']}
            ids=state['pending_gap']['experiment_refs']
            if state.get('pending_optimizer_handoff'):
                config.pop('measurement_source')
                config['optimizer_measurement_source']=str(self.root/state['pending_optimizer_handoff'])
                from .optimizer_handoff import load_context
                from .decision_traces import source_task_ids
                ids=ids+sorted(source_task_ids(load_context(config['optimizer_measurement_source'])))
            if state.get('active_tool') and self.config.get('inherit_tool_source'):
                from .evolution import Evolution
                previous=Evolution(self.root/state['active_tool'])
                try:
                    previous.verify();active=previous._active('tool')
                    if active is None:raise ContractError('previous project tool is no longer active')
                    candidate=previous.verified_candidate(active['candidate_id'])
                    inherited=directory/'previous-tool-source'
                    previous.artifacts.restore(candidate['implementation'],inherited)
                    config['source']=str(inherited);config['model_files']=sorted(candidate['implementation'])
                    config['tool_predecessor']={'candidate_id':candidate['candidate_id'],
                        'implementation_hash':candidate['implementation_hash'],'plan_hash':candidate['plan_hash']}
                finally:previous.store.close()
        else:
            source_config=read_json(source/('amazon-auto.json' if (source/'amazon-auto.json').exists() else 'campaign.json'))
            ids=[source_config['task_id']]
            if 'trace_sources' in config:
                config.pop('trace_source',None)
                sources=config['trace_sources']
                if not isinstance(sources,list) or not sources:raise ContractError('nonempty project trace_sources required')
                if not any(Path(item['campaign']).resolve()==source.resolve() for item in sources):
                    sources.append({'campaign':str(source)})
                for item in sources:
                    folder=Path(item['campaign'])
                    ids.append(read_json(folder/('amazon-auto.json' if (folder/'amazon-auto.json').exists() else 'campaign.json'))['task_id'])
            else:config['trace_source']={'campaign':str(source)}
            if self.config.get('inherit_harness_policy'):
                config['harness_base_policy']=source_config.get('research_policy',{})
            if self.config.get('optimizer_measurement_handoff'):
                config['optimizer_handoff_budget']=self.config.get('tool',config)['plan']['budget']
        config['plan']['development_ids']=sorted(set(config['plan']['development_ids'])|set(ids))
        return config

    @contextmanager
    def lock(self):
        with (self.root/'controller.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:raise ContractError('project already running') from error
            assert_stopped(self.root/'episodes');self.store.verify()
            yield

    def verify(self):
        if digest(read_json(self.root/'project.json'))!=read_json(self.root/'project-lock.json')['config_hash']:
            raise ContractError('project configuration changed')
        if 'amazon_study_hash' in self.config and digest(read_json(Path(self.config['model']['study'])/'study.json'))!=self.config['amazon_study_hash']:
            raise ContractError('project-bound Amazon study changed')
        integrity=self.store.verify();checked=[]
        for kind in ['model','tool','harness']:
            for ref in self.config.get(kind,{}).get('source_snapshot',{}).values():self.artifacts.verify(ref)
            for ref in self.config.get(kind,{}).get('initial_memory_refs',[]):self.artifacts.verify(ref)
            for ref in self.config.get(kind,{}).get('development_memory_refs',[]):self.artifacts.verify(ref)
        from .runtime import Research
        from .evolution import Evolution
        for event in self.store.events():
            entry=event['payload'].get('episode') if event['kind']=='project_transition' else None
            if entry is None:continue
            root=Path(entry['campaign'])
            if not root.resolve().is_relative_to(self.root/'episodes'):raise ContractError('child campaign outside project')
            if digest(read_json(root/'run-report.json'))!=entry['result_hash']:raise ContractError('project child report changed')
            if (root/'amazon-auto.json').exists():
                from .adapters.amazon_autonomous import AmazonAutoresearch
                child=AmazonAutoresearch(root,load_agent=False)
            else:child=Research(root,agent=self.agent) if entry['kind']=='model' else Evolution(root)
            try:checked.append({'campaign':str(root),'integrity':child.verify()})
            finally:child.store.close()
        return {**integrity,'verified_episodes':checked}

    def route_tool(self,state,gap):
        state['pending_gap']=gap
        scope=self.config['tool']['plan'].get('measurement_scope')
        if (self.config.get('enforce_tool_measurement_scope') and scope is not None
                and gap['required_information'] not in scope['supported_outputs']):
            state.update(status='stopped',stop_reason='measurement_contract_unsupported',
                measurement_contract_mismatch={'required_information':gap['required_information'],
                    'supported_outputs':sorted(scope['supported_outputs']),'scope_hash':digest(scope),
                    'qualification':'The frozen validator does not declare this output. Gap retained; no tool was generated or scientifically rejected. A matching validation contract is needed in a new project.'})
            return
        state['phase']='tool'

    def run(self):
        with self.lock():
            state=[e['payload']['state'] for e in self.store.events() if e['kind']=='project_transition'][-1]
            while state['status']!='stopped':
                kind=state['phase'];index=state['model_index'] if kind=='model' else state[kind+'_updates']+1
                directory=self.root/'episodes'/(kind+'-'+str(index))
                config=self.model_config(state,directory) if kind=='model' else self.evolve_config(state,directory,kind)
                result=self.child(directory,config)
                if kind=='model' and result['state'].get('stop_reason')=='role_transport_failure':
                    failure={'campaign':str(directory/'campaign'),'result_hash':digest(result),
                             'reason':'role_transport_failure'}
                    prior=[event['payload'] for event in self.store.events() if event['kind']=='project_child_transport_stopped']
                    if not prior or prior[-1]!=failure:
                        self.store.event('project_child_transport_stopped',failure)
                    raise ContractError('model role transport failed; repair and explicitly reopen the child before resuming project: '+failure['campaign'])
                episode={'kind':kind,'index':index,'campaign':str(directory/'campaign'),'result_hash':digest(result)}
                relative=str((directory/'campaign').relative_to(self.root))
                if kind=='model':
                    state={**state,'last_model':relative}
                    if result['state']['stop_reason']=='measurement_required':
                        store=Store(directory/'campaign/memory')
                        try:gaps=[e['payload']['gap'] for e in store.events() if e['kind']=='measurement_gap_opened']
                        finally:store.close()
                        if gaps and state['model_index']<self.config['max_model_episodes'] and 'tool' in self.config and state['tool_updates']<self.config.get('max_tool_updates',1):
                            if 'tool_runtime' not in self.config:raise ContractError('tool_runtime required for post-validation remeasurement')
                            self.route_tool(state,gaps[0])
                        else:state.update(status='stopped',stop_reason='measurement_unresolved')
                    elif state['model_index']>=self.config['max_model_episodes']:
                        state.update(status='stopped',stop_reason='model_episode_limit')
                    elif 'harness' in self.config and state['harness_updates']<self.config.get('max_harness_updates',1):
                        state.update(phase='harness')
                    else:state.update(status='stopped',stop_reason='model_research_finished')
                elif kind=='harness' and result['status']=='measurement_required':
                    state={**state,'harness_updates':state['harness_updates']+1,
                        'pending_gap':result['measurement_gap'],'pending_optimizer_handoff':relative}
                    if ('tool' in self.config and 'tool_runtime' in self.config
                        and state['tool_updates']<self.config.get('max_tool_updates',1)
                        and state['model_index']<self.config['max_model_episodes']):self.route_tool(state,state['pending_gap'])
                    else:state.update(status='stopped',stop_reason='measurement_unresolved')
                elif result['status']=='promoted':
                    if kind=='tool':
                        obligations=list(state.get('measurement_obligations',[]))
                        gap=state['pending_gap']
                        if not any(item['measurement_gap_id']==gap['measurement_gap_id'] for item in obligations):obligations.append(gap)
                        state={**state,'measurement_obligations':obligations}
                    state={**state,kind+'_updates':state[kind+'_updates']+1,'active_'+kind:relative,
                        'phase':'model','model_index':state['model_index']+1}
                    state.pop('pending_gap',None)
                    state.pop('pending_optimizer_handoff',None)
                elif (kind=='harness' and result.get('rejection_kind')=='validation_gate'
                      and self.config.get('continue_after_harness_rejection')
                      and state['model_index']<self.config['max_model_episodes']):
                    # Consume this attempt, retain any previously active policy,
                    # and continue from the last model without retrying the patch.
                    state={**state,'harness_updates':state['harness_updates']+1,
                        'phase':'model','model_index':state['model_index']+1}
                else:state={**state,'status':'stopped','stop_reason':kind+'_rejected'}
                self.transition(state,episode)
            write_json(self.root/'state.json',state)
            report={'state':state,'integrity':self.store.verify(),
                'episodes':[e['payload']['episode'] for e in self.store.events() if e['kind']=='project_transition' and 'episode' in e['payload']],
                'cost_policy':'costs remain in each child campaign; validation child costs must not be counted twice'}
            write_json(self.root/'run-report.json',report);return report
