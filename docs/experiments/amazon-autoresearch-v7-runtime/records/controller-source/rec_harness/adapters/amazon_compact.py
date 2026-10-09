"""Four actual calls: researcher, implementer, independent reviewer, analyst."""
import json
from rec_harness.core import ContractError, digest, file_hash, read_json
from rec_harness.knowledge import validate_knowledge_patch


def design(controller, folder, context, source, gap_check, proposal_check):
    graph=context['evidence_graph']
    if graph.get('hash')!=digest({k:v for k,v in graph.items() if k!='hash'}):
        raise ContractError('pre-design evidence graph changed')
    allowed={n['id'] for n in graph['nodes']}
    def validate(value, directory):
        gap_check(value['gap'],directory)
        proposal_check(value,directory,value['gap'])
        if not value['evidence_refs'] or not set(value['evidence_refs'])<=allowed:
            raise ContractError('researcher must cite supplied evidence graph nodes')
        if not value['gap']['observation_refs'] or not set(value['gap']['observation_refs'])<=allowed:
            raise ContractError('gap observation_refs must cite supplied evidence graph nodes')
    result=controller._role(folder,'researcher',context,source,validate)
    return result['gap'],{k:result[k] for k in ('method','proposal')}


def implementation_context(context,gap,generated):
    keys=('task_id','model_version','protocol_hash','bindings','source_contract',
          'insertion_contract_format','generated_controls','previous_development_sources')
    return {**{key:context[key] for key in keys if key in context},'gap':gap,**generated,
            'instructions':['Implement the frozen proposal and each named control; do not redesign the hypothesis or run training.',
                            'Preserve imports, module settings, training helpers and score/encode consistency. changed_files must exactly match model.py followed by generated control files.',
                            'Source changes must stay inside insertion_contract_format; report actual checks and remaining limitations.']}


def review_code(controller, folder, context, cards, baseline, candidate, controls):
    sources={'baseline.py':baseline,'model.py':candidate}
    sources.update({item['file']:controls[item['model_version']] for item in context.get('generated_controls',[])})
    implementation={name:{'sha256':file_hash(path),'source':path.read_text()} for name,path in sources.items()}
    identity=digest({'cards':cards,'sources':implementation})
    request={'task_id':controller.task,'protocol_hash':controller.protocol_hash,
             'model_version':context['bindings']['candidate_model_version'],
             'cards':cards,'implementation_sources':implementation,'implementation_hash':identity,
             'generated_controls':context.get('generated_controls',[]),
             'implementation_author_report':read_json(folder/'experiment/response.json'),
             'instructions':['Independently compare actual baseline, candidate and control code against the frozen design. Author checks are claims, not independent evidence.',
                             'Check correctness and declaration/implementation consistency. Reference implementation_hash exactly. Do not rerun training or relax the design.',
                             'Approve only with no blocking findings. Unknown correctness or mismatched implementation must be unresolved/request_changes. Performance and mechanism judgments occur later.']}
    def validate(value,_):
        if value['evidence_refs']!=[identity]:raise ContractError('reviewer implementation identity mismatch')
        if value['verdict']=='approve' and value['findings']:
            raise ContractError('approve cannot contain blocking findings; put nonblocking caveats in limitations')
    review=controller._role(folder,'reviewer',request,validator=validate)
    # A substantive rejection is final for this implementation, not an output
    # formatting error to retry until the reviewer approves identical code.
    if review['verdict']!='approve':
        raise ContractError('independent code review blocked training: '+review['verdict']+'; '+'; '.join(review['findings']))
    return review


def analyze(controller,folder,review,graph,report,decision_check):
    from .amazon_cycle import finalize_cycle
    contract={'base_version':graph['graph_version'],'fields':['base_version','nodes','edges'],
              'allowed_node_types':['Claim','GapHypothesis','MeasurementGap'],'origin':'agent_hypothesis',
              'task_id':graph['task_id'],'protocol_hash':controller.protocol_hash,'status':'provisional',
              'edge_fields':['id','source','target','relation','attrs'],
              'edge_attrs_required':['scope','evidence_refs','origin','status','created_at','revision_id'],
              'evidence_refs':[n['id'] for n in graph['nodes']]}
    keys=('task_id','model_version','protocol_hash','bindings','protocol','development_baseline',
          'incumbent_measurement_context','baseline_history','implementation_author_report',
          'implementation_evidence','candidate_execution_evidence','generated_control_evidence',
          'handoff','development','development_history')
    request={**{key:review[key] for key in keys if key in review},'graph':graph,'patch_contract':contract,
             'code_review':read_json(folder/'reviewer/response.json'),
             'mechanism_comparisons':report['summary'].get('mechanism_comparisons',[]),
             'instructions':review['instructions']+[
                 'Return decision and knowledge in one response. knowledge.patch_json is an append-only graph patch at patch_contract.base_version. Use only supplied node IDs as evidence; empty patch is valid.',
                 'Record candidate/control interventions, outcome differences, alternative explanations and next discriminating tests as provisional knowledge. Negative/null evidence is retained. Training success and code review do not establish causal mechanisms.',
                 'Existing historical explanation nodes may be linked by BASED_ON, SUPERSEDES or ALTERNATIVE_TO using current scoped development evidence. Never rewrite historical nodes.']}
    def validate(value,directory):
        decision_check(value['decision'],directory)
        validate_knowledge_patch(json.loads(value['knowledge']['patch_json']),graph)
    result=controller._role(folder,'analyst',request,validator=validate)
    return result['decision'],finalize_cycle(controller.study,review['bindings']['proposal_id'],folder)
