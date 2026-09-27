"""Resumable six-role Amazon research; no handwritten per-round launch scripts."""
import fcntl
import json
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

from rec_harness.agents import schema_for,validate_schema
from rec_harness.core import Artifacts,ContractError,digest,file_hash,read_json,write_json,validate_card,identifier
from rec_harness.jailed_agent import JailedCodexAgent
from rec_harness.knowledge import validate_knowledge_patch
from rec_harness.process_identity import assert_stopped
from rec_harness.store import Store
from rec_harness.measurement_gaps import MeasurementDeferred,validate_requirements,block_model_proposal
from .amazon_controls import load_config
from .amazon_report import audit_job
from .amazon_plan import register_plan
from .amazon_admission import admit_plan
from .amazon_phases import run_phase
from .amazon_study import run_study
from .amazon_selection import development_readout,release_confirmation
from .amazon_research_graph import ingest_development
from .amazon_cycle import finalize_cycle
from .amazon_rescore import rescore_job
from .amazon_memory import validate_memory,validate_readout
from .model_source_contract import review_source,MODULE_CALLS,TENSOR_METHODS


class AmazonAutoresearch:
    @staticmethod
    def init(root,configuration,*,review_only=False):
        root=Path(root).resolve()
        if root.exists() and any(root.iterdir()):raise ContractError('autonomous directory must be empty')
        config=dict(configuration);config['study']=str(Path(config['study']).resolve())
        if config.get('review_only') and not review_only:raise ContractError('review-only configuration requires review entry point')
        if review_only:config['review_only']=True
        if config.get('review_predecessor'):
            if not review_only:raise ContractError('review predecessor requires review-only entry point')
            config['review_predecessor']=str(Path(config['review_predecessor']).resolve())
        study=load_config(config['study']);identifier(config['task_id'])
        if type(config.get('max_rounds')) is not int or config['max_rounds']<1:raise ContractError('positive max_rounds required')
        if type(config.get('role_call_limit')) is not int or config['role_call_limit']<6:raise ContractError('role_call_limit must cover six roles')
        if type(config.get('history_diagnostic',False)) is not bool:raise ContractError('history_diagnostic must be boolean')
        if not config.get('rescore_recipe') or not config.get('uid_registry'):raise ContractError('independent CPU rescoring configuration required')
        config['study_config_hash']=digest(read_json(Path(config['study'])/'study.json'))
        config['protocol_hash']=study['protocol_hash']
        if 'execution_id' in config:
            identifier(config['execution_id'])
            from .amazon_execution import inspect_execution,resolve_execution
            execution_store=Store(Path(config['study'])/'memory')
            try:(inspect_execution if review_only else resolve_execution)(execution_store,config['execution_id'])
            finally:execution_store.close()
        validate_requirements(config.get('measurement_requirements',[]))
        from rec_harness.measurement_discovery import validate_config
        validate_config(config.get('measurement_discovery'))
        from .amazon_activation import validate_options
        validate_options(config.get('activation_diagnostic'),study)
        if 'block_score_diagnostics' in config:raise ContractError('block_score_diagnostics is controller-owned; supply block_score_diagnostic_dirs')
        if config.get('block_score_diagnostic_dirs'):
            from .amazon_block_context import freeze_reports
            config['block_score_diagnostics']=freeze_reports(config.pop('block_score_diagnostic_dirs'),study,Artifacts(root/'artifacts/objects'))
        if 'block_interaction_diagnostics' in config:raise ContractError('block_interaction_diagnostics is controller-owned')
        if config.get('block_interaction_groups'):
            from .amazon_interaction_context import freeze_interactions
            config['block_interaction_diagnostics']=freeze_interactions(config.pop('block_interaction_groups'),study,Artifacts(root/'artifacts/objects'))
        if 'diagnostic_tool' in config:raise ContractError('diagnostic_tool is controller-owned; supply diagnostic_registry')
        if 'diagnostic_registry' in config:
            from rec_harness.diagnostics import freeze_tool
            from rec_harness.core import number
            number(config.get('diagnostic_budget'),'diagnostic_budget',0)
            config['diagnostic_tool']=freeze_tool(config.pop('diagnostic_registry'),Artifacts(root/'artifacts/objects'))
        if 'harness_registry' in config:
            from rec_harness.harness_activation import load_active_policy
            if config.get('research_policy'):raise ContractError('cannot override a promoted harness policy')
            config['research_policy'],config['harness_activation']=load_active_policy(config.pop('harness_registry'),config.pop('harness_identity',None))
        from rec_harness.evolution import validate_policy
        validate_policy(config.get('research_policy',{}))
        config['development_memory']=[validate_memory(pack,study['protocol_hash'],study['development_seeds'])
                                      for pack in config.get('development_memory',[])]+[
                                      validate_memory(read_json(path),study['protocol_hash'],study['development_seeds'])
                                      for path in config.pop('development_memory_files',[])]
        base=config.get('base_model_id','legacy_incumbent')
        if base not in study['model_versions']:raise ContractError('unregistered initial model')
        root.mkdir(parents=True,exist_ok=True);write_json(root/'amazon-auto.json',config)
        write_json(root/'auto-lock.json',{'config_hash':digest(config)})
        write_json(root/'state.json',{'round':0,'incumbent':base,'status':'ready'})
        store=Store(root/'memory');store.configure_budget(config['role_call_limit']);store.close()

    def __init__(self,root,agent=None,*,load_agent=True,read_only=False):
        self.root=Path(root).resolve();self.config=read_json(self.root/'amazon-auto.json')
        if digest(self.config)!=read_json(self.root/'auto-lock.json')['config_hash']:raise ContractError('autonomous configuration changed')
        self.study=Path(self.config['study']);self.store=Store(self.root/'memory',read_only=read_only)
        self.task=self.config['task_id'];self.protocol_hash=self.config['protocol_hash']
        self.artifacts=Artifacts(self.root/'artifacts/objects');backend=self.config['agent_backend']
        self.agent=(agent or JailedCodexAgent(backend['template'],backend['runtime_root'],timeout=self.config.get('role_timeout',300),
            model=backend.get('model'),web_search=backend.get('web_search'),prompts=self.config.get('research_policy',{}).get('prompts'))) if load_agent else None
        if digest(read_json(self.study/'study.json'))!=self.config['study_config_hash']:raise ContractError('bound study changed')

    def close(self):self.store.close()

    @contextmanager
    def lock(self):
        with (self.root/'controller.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:raise ContractError('Amazon controller already active') from error
            assert_stopped(self.root/'rounds')
            assert_stopped(self.root/'research_archive')
            yield

    def verify(self):
        from .amazon_auto_audit import verify_campaign
        with self.lock():return verify_campaign(self)

    def _role(self,folder,role,context,source=None,validator=None):
        target=folder/role
        artifacts=Artifacts(self.store.root.parent/'artifacts/objects')
        def complete(task,wall_seconds):
            if any(e['kind']=='role_completed' and e['payload']['task']==task for e in self.store.events()):return
            self.store.event('role_completed',{'task':task,'role':role,**read_json(target/'accepted.json'),
                'task_id':task,'status':'success','wall_seconds':wall_seconds,
                'archive':artifacts.snapshot(target,[name for name in ['context.json','request.json','response.json','resources.json','schema.json','literature-search.json','stdout.jsonl'] if (target/name).exists()])})
        def accept(response):
            record={'response_hash':digest(response),'context_hash':digest(read_json(target/'context.json'))}
            if (target/'model/model.py').exists():record['source_hash']=file_hash(target/'model/model.py')
            write_json(target/'accepted.json',record)
        def recover_completion():
            receipt=target/'resources.json'
            if not receipt.exists() or read_json(receipt).get('status')!='success':return
            submitted=[e['payload'] for e in self.store.events() if e['kind']=='agent_submitted'
                and e['payload']['task_id'].startswith(folder.name+'-'+role+'-')
                and e['payload']['input_hash']==digest(read_json(target/'context.json'))]
            if submitted:complete(submitted[-1]['task_id'],read_json(receipt).get('wall_seconds',0))
        def transport_status():
            for name in ['resources.json','process.json']:
                if (target/name).exists():return read_json(target/name).get('status')
            return None
        if (target/'accepted.json').exists():
            response=read_json(target/'response.json');validate_schema(response,schema_for(role))
            accepted=read_json(target/'accepted.json')
            if digest(response)!=accepted['response_hash'] or digest(read_json(target/'context.json'))!=accepted['context_hash']:raise ContractError('cached role changed')
            if 'source_hash' in accepted and file_hash(target/'model/model.py')!=accepted['source_hash']:raise ContractError('cached model changed')
            if validator:validator(response,target)
            recover_completion()
            return response
        if ((target/'rejection.json').exists() and
            read_json(target/'rejection.json').get('retryable') is False) or transport_status() not in (None,'success'):
            raise ContractError(role+' transport failed; repair the host and explicitly reopen before retrying')
        previous=len(list(folder.glob(role+'-rejected-*')))
        for retry in range(max(0,2-previous)):
            if target.exists() and (target/'request.json').exists():
                # Successful transport may have finished just before controller interruption.
                if (target/'resources.json').exists() and read_json(target/'resources.json')['status']=='success':
                    try:
                        response=read_json(target/'response.json');validate_schema(response,schema_for(role))
                        if validator:validator(response,target)
                        accept(response)
                        recover_completion()
                        return response
                    except (ContractError,ValueError,KeyError,TypeError,OSError):pass
                suffix=len(list(folder.glob(role+'-rejected-*')))+1
                target.rename(folder/(role+'-rejected-'+str(suffix)))
                if suffix>=2:break
            if self.store.budget()['remaining']<1:raise ContractError('role call budget exhausted')
            target.mkdir(parents=True,exist_ok=True)
            if source:
                (target/'model').mkdir(exist_ok=True);shutil.copyfile(source,target/'model/model.py')
            task=f"{folder.name}-{role}-{len(self.store.events())}";self.store.reserve(task,1)
            write_json(target/'context.json',context)
            before=artifacts.snapshot(target/'model',['model.py']) if source else {}
            self.store.event('agent_submitted',{'task_id':task,'role':role,'input_hash':digest(context),
                'model_version':context.get('model_version'),'graph_version':self.store.version,
                'visible_state_ref':artifacts.put(target/'context.json'),'source_snapshot':before})
            started=time.monotonic()
            try:
                response=self.agent.run(role,context,target);validate_schema(response,schema_for(role))
                if validator:validator(response,target)
                accept(response)
                self.store.settle(task,1,'success');complete(task,time.monotonic()-started)
                return response
            except (ContractError,ValueError,KeyError,TypeError,OSError) as error:
                self.store.settle(task,1,'failed')
                retryable=transport_status()=='success'
                write_json(target/'rejection.json',{'status':'rejected','reason':str(error),'retryable':retryable})
                self.store.event('role_rejected',{'task':task,'role':role,'reason':str(error),'retryable':retryable})
                if not retryable:
                    raise ContractError(role+' transport failed; repair the host and explicitly reopen before retrying') from error
                context={**context,'validation_error':str(error),'repair_instruction':'Correct this role output; preserve frozen bindings.'}
        raise ContractError(role+' failed after automatic retry')

    def _rescore(self,folder,version,seeds):
        config=load_config(self.study);summaries=[]
        for dataset in config['datasets']:
            for seed in seeds:
                directory=folder/'independent-rescores'/f'{version}-{dataset}-{seed}'
                receipt=directory/'acceptance.json'
                if (directory/'rescore-plan.json').exists():
                    result=rescore_job(self.study,version,dataset,seed,directory,self.config['rescore_recipe'],
                                       self.config['uid_registry'],resume=True)
                elif receipt.exists():
                    # Legacy completed scoring predates frozen recovery plans.
                    # Keep its existing contract; never synthesize a new plan.
                    result=read_json(receipt);training,_=audit_job(self.study,config,version,dataset,seed)
                    if result['hash']!=digest({k:v for k,v in result.items() if k!='hash'}) or result['checkpoint_hash']!=training['checkpoint_sha256'] or result['source_hash']!=training['source_hash']:
                        raise ContractError('cached independent rescore changed')
                else:
                    result=rescore_job(self.study,version,dataset,seed,directory,self.config['rescore_recipe'],
                                       self.config['uid_registry'],resume=directory.exists())
                if result['rank_disagreements'] or result['positive_tied_users']:
                    raise ContractError('independent rescore differs or contains unresolved ties')
                summaries.append(result)
        return summaries

    def _context(self,state,folder):
        cfg=load_config(self.study);base=state['incumbent'];rows=[];execution_evidence=[]
        from .amazon_review_evidence import protocol_summary
        for dataset in cfg['datasets']:
            for seed in cfg['development_seeds']:
                result,manifest=audit_job(self.study,cfg,base,dataset,seed)
                if result.get('execution_id','direct-reviewed-v1')!=(self.config.get('execution_id') or 'direct-reviewed-v1'):
                    raise ContractError('development baseline execution differs from autonomous execution')
                rows.append({'dataset':dataset,'seed':seed,'metrics':result['metrics']})
                execution_evidence.append(protocol_summary(cfg,result,manifest))
        ledger=Store(self.study/'memory')
        try:budget=ledger.budget()
        finally:ledger.close()
        pid=self.config['task_id']+'-'+folder.name
        history=[];research_memory=[]
        for pack in self.config.get('development_memory',[]):
            validate_memory(pack,cfg['protocol_hash'],cfg['development_seeds']);research_memory.extend(pack['entries'])
            history.extend(entry['development'] for entry in pack['entries'] if 'development' in entry)
        from .amazon_memory import execution_failure
        failures=[]
        failure_paths=[path for path in sorted((self.root/'rounds').glob('*/failure.json'))
                       if path.parent!=folder and not (path.parent/'private-confirmation-started.json').exists()]
        events=self.store.events() if failure_paths else []
        for path in failure_paths:
            failure=execution_failure(path.parent,events)
            if failure is not None:failures.append({'round':path.parent.name,**failure})
        for path in sorted((self.root/'rounds').glob('*/development-readout.json')):
            if path.parent!=folder:
                history.append(read_json(path))
                entry={'round':path.parent.name,'development_hash':read_json(path)['hash']}
                for name,key in [('handoff.json','design'),('lead/response.json','development_decision'),('knowledge/response.json','provisional_knowledge')]:
                    previous=path.parent/name
                    if previous.exists():entry[key]=read_json(previous)['cards'] if key=='design' else read_json(previous)
                research_memory.append(entry)
        for path in self.config.get('development_history',[]):
            report=read_json(path)
            validate_readout(report,cfg['protocol_hash'],cfg['development_seeds'])
            history.append(report)
        available_history=history
        limit=self.config.get('research_policy',{}).get('retrieval_limit')
        if limit is not None:
            history=history[-limit:];research_memory=research_memory[-limit:];failures=failures[-limit:]
        from .amazon_memory import compact_development_context
        history,research_memory=compact_development_context(history,research_memory,cfg['protocol_hash'],cfg['development_seeds'],available_history)
        from .amazon_diagnostics import diagnostic_context
        diagnostics=diagnostic_context(self,base,rows)
        if self.config.get('block_score_diagnostics') or self.config.get('block_interaction_diagnostics'):
            from .amazon_block_context import block_context
            diagnostics.update(block_context(self,base,cfg))
        if not diagnostics.get('measurement_failure'):
            from .amazon_activation import activation_context
            diagnostics.update(activation_context(self,base,cfg))
        return {**diagnostics,'task_id':self.config['task_id'],'model_version':base,'protocol_hash':cfg['protocol_hash'],
            'protocol':cfg['protocol'],'seeds':cfg['development_seeds']+cfg['confirmation_seeds'],
            'development_baseline':rows,'baseline_execution_evidence':execution_evidence,
            'history':history,'research_memory':research_memory,'prior_execution_failures':failures,'budget':budget,
            'bindings':{'proposal_id':pid,'candidate_model_version':pid+'-model','base_model_id':base,
                        'controls':[base],'train_protocol_hash':cfg['protocol_hash'],
                        'seeds':cfg['development_seeds']+cfg['confirmation_seeds']},
            'source_contract':{'editable_classes':['NextItNetModel','NextItNetBlock'],'library_calls':sorted(MODULE_CALLS),
                               'tensor_methods':sorted(TENSOR_METHODS),'preserve':'all module configuration, imports, training/data helpers and method signatures'},
            'instructions':['Use exact supplied bindings. Only model.py is editable; no manual source-template admission is needed.',
                'Prior development readouts appear once in history, keyed by hash. Resolve research_memory.development_hash against history; all referenced readouts are included.',
                'Evidence GapCard.model_version MUST equal bindings.base_model_id (the incumbent being diagnosed), never candidate_model_version. GapCard.protocol_hash MUST equal bindings.train_protocol_hash.',
                'Proposal.controls is a list of model IDs, not prose descriptions. Copy bindings.controls verbatim. Put explanations and CPU checks in mechanism/insertion_contract/falsification_condition.',
                'Numerical contract must hold after learning, not only at initialization: encode_sequence returns causal [batch,length,hidden] final representations; score_candidates equals item_emb(candidates) @ encode_sequence(seq)[0,-1]. Forward uses the same encoded representations. Insert learnable readout transforms inside encode_sequence, never only in forward/score_candidates.',
                'Propose one distinct falsifiable architectural change and reuse incumbent as the sole matched control. Do not reopen rejected mechanisms.',
                'Proposal budget must fit remaining GPU budget and include4 development plus8 conditional confirmation fits and overhead. Never shorten protocol.',
                'Independent initialization for every category/seed. Only public development aggregates are supplied; no raw users, checkpoints or confirmation outcomes.',
                'Experiment returns changed_files exactly [model.py]. Do not train. Host performs source admission, CPU contracts and GPU evaluation.',
                'Role output IDs and seeds must match bindings. Scientific conclusions stay unknown/provisional.']}

    def _round(self,state,folder):
        folder.mkdir(parents=True,exist_ok=True)
        if not (folder/'input.json').exists():write_json(folder/'input.json',self._context(state,folder))
        context=read_json(folder/'input.json');binding=context['bindings'];pid=binding['proposal_id']
        if context.get('measurement_failure'):raise MeasurementDeferred(context['measurement_failure'])
        cfg=load_config(self.study);source=Path(cfg['model_versions'][binding['base_model_id']]['path'])
        def gap_check(value,_):
            validate_card('gap',value)
            if value['model_version']!=binding['base_model_id'] or value['protocol_hash']!=binding['train_protocol_hash']:
                raise ContractError('GapCard diagnoses the incumbent: model_version must be '+binding['base_model_id']+'; protocol_hash must be '+binding['train_protocol_hash'])
        gap=self._role(folder,'evidence',context,source,gap_check)
        if not context.get('measurement_gaps'):
            from rec_harness.measurement_discovery import discover,validate_review
            context['measurement_gaps']=discover(self,{'incumbent':state['incumbent'],'baseline':context.get('measurement_baseline')},
                context,gap,folder.name,lambda request:self._role(folder,'measurement',request,source,lambda response,_:validate_review(request,response)))
        block_model_proposal(self,folder.name,gap,context.get('measurement_gaps',[]))
        def proposal_check(value,_):
            validate_card('method',value['method']);validate_card('proposal',value['proposal']);p=value['proposal']
            for key in ['proposal_id','base_model_id','controls','train_protocol_hash','seeds']:
                if p[key]!=binding[key]:raise ContractError('proposal binding mismatch: '+key+' must equal the literal JSON value '+json.dumps(binding[key]))
            if p['gap_ids']!=[gap['gap_id']] or p['method_ids']!=[value['method']['method_id']]:raise ContractError('unresolved proposal references')
            if not 0<p['budget']<=context['budget']['remaining']:raise ContractError('proposal budget outside frozen envelope')
        generated=self._role(folder,'hypothesis',{**context,'gap':gap},source,proposal_check)
        def source_check(value,directory):
            if value['changed_files']!=['model.py']:raise ContractError('only model.py may change')
            review_source(source.read_text(),(directory/'model/model.py').read_text())
        self._role(folder,'experiment',{**context,'gap':gap,**generated},source,source_check)
        candidate=folder/'experiment/model/model.py';cards={'gap':gap,**generated}
        versions={'candidate':{'model_version':binding['candidate_model_version'],'source':{'path':str(candidate),'sha256':file_hash(candidate)}},
                  'controls':[{'model_version':binding['base_model_id'],'source':cfg['model_versions'][binding['base_model_id']]}]}
        write_json(folder/'handoff.json',{'cards':cards,'versions':versions})
        plan=register_plan(self.study,cards,versions);write_json(folder/'preregistration.json',plan)
        write_json(folder/'admission.json',admit_plan(self.study,pid,recipe=self.config['rescore_recipe'],uid_registry=self.config['uid_registry']))
        run_phase(self.study,pid,'development',execution_id=self.config.get('execution_id'))
        self._rescore(folder,binding['candidate_model_version'],cfg['development_seeds'])
        report=development_readout(self.study,pid);write_json(folder/'development-readout.json',report)
        history=None
        if self.config.get('history_diagnostic'):
            from .amazon_history_context import development_history
            history=development_history(self,folder,binding)
        lead,cycle=self._review_roles(folder,context,binding,cards,versions,candidate,report,history)
        selected=False;confirmation=None
        if report['summary']['eligible_for_confirmation'] and lead['selection']=='keep':
            write_json(folder/'private-confirmation-started.json',{'scope':'host-only phase boundary; excludes later failures from agent context'})
            release_confirmation(self.study,pid,lead,report['hash'])
            # Comparators need not consume confirmation resources for rejected
            # development candidates. Existing matching receipts are reused.
            run_study(self.study,model_versions=[binding['base_model_id']],
                      seeds=cfg['confirmation_seeds'],execution_id=self.config.get('execution_id'))
            self._rescore(folder,binding['base_model_id'],cfg['confirmation_seeds'])
            run_phase(self.study,pid,'confirmation',execution_id=self.config.get('execution_id'))
            self._rescore(folder,binding['candidate_model_version'],cfg['confirmation_seeds'])
            cfg=load_config(self.study);effects={}
            for seed in cfg['confirmation_seeds']:
                deltas=[]
                for dataset in cfg['datasets']:
                    candidate_result,cm=audit_job(self.study,cfg,binding['candidate_model_version'],dataset,seed)
                    baseline_result,bm=audit_job(self.study,cfg,binding['base_model_id'],dataset,seed)
                    if (cm['candidates_hash'],cm['data_hashes'],candidate_result.get('execution_id','direct-reviewed-v1'))!=(bm['candidates_hash'],bm['data_hashes'],baseline_result.get('execution_id','direct-reviewed-v1')):
                        raise ContractError('confirmation pairing differs')
                    deltas.append(candidate_result['metrics'][cfg['protocol']['metric']]-baseline_result['metrics'][cfg['protocol']['metric']])
                effects[str(seed)]=sum(deltas)/len(deltas)
            selected=all(effect>cfg['protocol']['minimum_effect'] for effect in effects.values())
            confirmation={'effects':effects,'selected':selected,'scope':'host-only confirmation; not sent to Codex roles or final test'}
            write_json(folder/'private-confirmation.json',confirmation)
        return {'cycle_identity':cycle['identity'],'candidate':binding['candidate_model_version'],
                'development_hash':report['hash'],'selected':selected,'confirmation_executed':confirmation is not None}

    def _review_roles(self,folder,context,binding,cards,versions,candidate,report,history=None):
        cfg=load_config(self.study);pid=binding['proposal_id']
        graph_path=self.study/'research-readouts'/(pid+'-graph.json')
        graph=read_json(graph_path) if graph_path.exists() else ingest_development(self.study,pid)
        from .amazon_review_evidence import admission_evidence
        implementation=admission_evidence(self.study,pid,cfg['protocol_hash'],binding['candidate_model_version'],candidate)
        review={**context,'model_version':binding['candidate_model_version'],
                'visible_model_source':None,
                'implementation_evidence':implementation,
                'handoff':{'cards':cards,'versions':versions},'development':report,
                'instructions':['Evaluate the actual complete development readout. Include its exact hash in evidence_refs: '+report['hash'],
                                'implementation_evidence reports host-verified admission checks; its report hashes are references, not files available in your workspace. Distinguish these checks from missing scientific evidence. Evaluation has no source file; lead receives candidate source separately.',
                                'Keep only if eligible_for_confirmation is true. No confirmation results are supplied; conclusion unknown/provisional.']}
        if history is not None:review['development_history']=history
        def decision_check(value,_):
            validate_card('decision',value)
            if report['hash'] not in value['evidence_refs']:raise ContractError('exact development readout hash required')
            if value['conclusion_status'] not in ['unknown','provisional']:raise ContractError('development cannot establish scientific confirmation')
        evaluation=self._role(folder,'evaluation',review,validator=decision_check)
        lead=self._role(folder,'lead',{**review,'evaluation':evaluation,
            'visible_model_source':{'path':'model/model.py','model_version':binding['candidate_model_version'],
                                    'sha256':file_hash(candidate)}},candidate,decision_check)
        knowledge_context={'graph':graph,'development':report,'decision':lead,
            'patch_contract':{'base_version':graph['graph_version'],'fields':['base_version','nodes','edges'],
                'allowed_node_types':['Claim','GapHypothesis','MeasurementGap'],'origin':'agent_hypothesis',
                'task_id':graph['task_id'],'protocol_hash':cfg['protocol_hash'],'status':'provisional',
                'evidence_refs':[n['id'] for n in graph['nodes']]},
            'instructions':['Append new provisional explanations only. Every node attrs must include task_id/protocol_hash/origin; Claims additionally need scope/status/evidence_refs.',
                            'Evidence refs must be supplied graph node IDs, never file paths. Retain all negative slices; empty patch is valid.']}
        self._role(folder,'knowledge',knowledge_context,validator=lambda v,d:validate_knowledge_patch(json.loads(v['patch_json']),graph))
        cycle=finalize_cycle(self.study,pid,folder)
        return lead,cycle

    def run(self):
        if self.config.get('review_only'):raise ContractError('review-only controller cannot run training or generate designs')
        with self.lock():
            self.store.recover();state=read_json(self.root/'state.json')
            # The event is authoritative if publication of state.json was interrupted.
            for event in self.store.events():
                if event['kind'] in {'auto_round_completed','auto_round_reopened'}:state=event['payload']['state']
            transitions=[e for e in self.store.events() if e['kind'] in {'auto_round_completed','auto_round_reopened'}]
            if transitions and transitions[-1]['kind']=='auto_round_reopened':
                source_root=Path(__file__).resolve().parents[1]
                for name,ref in transitions[-1]['payload']['controller_sources'].items():
                    self.artifacts.verify(ref)
                    if file_hash(source_root/name)!=ref['sha256']:raise ContractError('reopened controller source changed')
            reason=state.get('stop_reason','max_rounds')
            while state['round']<self.config['max_rounds'] and reason!='measurement_required':
                ledger=Store(self.study/'memory')
                try:
                    from .amazon_campaign import campaign_guard
                    campaign_guard(load_config(self.study),ledger)
                    remaining=ledger.budget()['remaining']
                finally:ledger.close()
                folder=self.root/'rounds'/('round-'+str(state['round']+1))
                roles=['evidence','hypothesis','experiment','evaluation','lead','knowledge']
                if 'measurement_discovery' in self.config:roles.append('measurement')
                required=sum(not (folder/role/'accepted.json').exists() for role in roles)
                if self.store.budget()['remaining']<required:reason='role_budget';break
                if remaining<self.config.get('minimum_round_gpu_seconds',1800):reason='gpu_budget';break
                folder=self.root/'rounds'/('round-'+str(state['round']+1))
                try:
                    result=self._round(state,folder)
                    if result['selected']:state['incumbent']=result['candidate']
                except MeasurementDeferred as error:
                    result={'status':'deferred','reason':str(error),'selected':False};reason='measurement_required'
                    self.store.event('auto_round_deferred',{'round':state['round']+1,**result})
                except (ContractError,OSError,ValueError,KeyError,TypeError) as error:
                    result={'status':'failed','reason':str(error)};write_json(folder/'failure.json',result)
                    self.store.event('auto_round_failed',{'round':state['round']+1,**result})
                state={**state,'round':state['round']+1,'status':'ready'}
                if reason=='measurement_required':state.update(status='stopped',stop_reason=reason)
                self.store.event('auto_round_completed',{'result':result,'state':state})
                write_json(self.root/'state.json',state)
            state={**state,'status':'stopped','stop_reason':reason};write_json(self.root/'state.json',state)
            rounds=[e['payload']['result'] for e in self.store.events() if e['kind']=='auto_round_completed']
            result={'state':state,'role_calls':self.store.budget(),'integrity':self.store.verify(),
                    'rounds':rounds,'failed_rounds':sum(r.get('status')=='failed' for r in rounds)}
            if self.config.get('activation_diagnostic'):
                from .amazon_activation import cost_summary
                result['activation_costs']=cost_summary(self.root)
            if self.config.get('block_score_diagnostics') or self.config.get('block_interaction_diagnostics'):
                from .amazon_block_context import cost_summary
                result['block_score_costs']=cost_summary(self.config)
            write_json(self.root/'run-report.json',result);return result

    def reopen_source_failure(self,reason):
        """Explicitly retry a pre-admission failure, retaining all calls and artifacts."""
        if not isinstance(reason,str) or not reason.strip():raise ContractError('repair reason required')
        with self.lock():
            events=self.store.events()
            transitions=[e for e in events if e['kind'] in {'auto_round_completed','auto_round_reopened'}]
            if not transitions:raise ContractError('no failed round to reopen')
            latest=transitions[-1]
            if latest['kind']=='auto_round_reopened':return latest['payload']
            payload=latest['payload'];state=payload['state'];folder=self.root/'rounds'/('round-'+str(state['round']))
            if payload['result'].get('status')!='failed' or not (folder/'failure.json').exists():
                raise ContractError('latest round is not a source failure')
            if any((folder/name).exists() for name in ['handoff.json','preregistration.json','admission.json','private-confirmation-started.json']):
                raise ContractError('only pre-admission failures can be reopened')
            if self.store.db.execute("SELECT 1 FROM budget WHERE status='running'").fetchone():
                raise ContractError('unsettled role work cannot be reopened')
            if read_json(folder/'failure.json')!=payload['result']:raise ContractError('failure projection changed')
            source_root=Path(__file__).resolve().parents[1]
            source_refs={str(p.relative_to(source_root)):self.artifacts.put(p) for p in sorted(source_root.rglob('*.py'))}
            record={'failed_event_hash':latest['hash'],'reason':reason,
                'prior_artifacts':self.artifacts.snapshot(folder,[str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()]),
                'controller_sources':source_refs,'retained_role_cost':self.store.budget()['charged_or_reserved'],
                'state':{'incumbent':state['incumbent'],'round':state['round']-1,'status':'ready'}}
            self.store.event('auto_round_reopened',record)
            write_json(self.root/'state.json',record['state'])
            return record
