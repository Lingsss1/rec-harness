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
            **({'review_revision':context['review_revision']} if 'review_revision' in context else {}),
            'instructions':['Implement the frozen proposal and each named control; do not redesign the hypothesis or run training.',
                            'Preserve imports, module settings, training helpers and score/encode consistency. changed_files must exactly match model.py followed by generated control files.',
                            'Source changes must stay inside insertion_contract_format; report actual checks and remaining limitations.']}


def review_code(controller, folder, context, cards, baseline, candidate, controls, *, require_approval=True):
    sources={'baseline.py':baseline,'model.py':candidate}
    sources.update({item['file']:controls[item['model_version']] for item in context.get('generated_controls',[])})
    implementation={name:{'sha256':file_hash(path),'source':path.read_text()} for name,path in sources.items()}
    identity=digest({'cards':cards,'sources':implementation})
    request={'task_id':controller.task,'protocol_hash':controller.protocol_hash,
             'model_version':context['bindings']['candidate_model_version'],
             'cards':cards,'implementation_sources':implementation,'implementation_hash':identity,
             'generated_controls':context.get('generated_controls',[]),
             'implementation_author_report':read_json(folder/'experiment/response.json'),
             'protocol':context.get('protocol',{}),
             'baseline_execution_evidence':[{key:row[key] for key in ('dataset','seed','status','protocol','protocol_hash','receipt_hash','scope') if key in row}
                 for row in context.get('baseline_execution_evidence',[])],
             'review_stage':{'name':'source_review_before_host_admission',
                 'candidate_cpu_checks':'pending: admit_plan runs after this review and before GPU training',
                 'protocol_authority':'Host execution protocol and receipts, not unchanged helper defaults in model source',
                 'scope':'Source approval does not certify runtime checks or proposal-specific scientific tests.'},
             'instructions':['Independently compare actual baseline, candidate and control code against the frozen design. Author checks are claims, not independent evidence.',
                             'Check correctness and declaration/implementation consistency. Reference implementation_hash exactly. Do not rerun training or relax the design.',
                             'Use checks for successful observations, limitations for nonblocking caveats, and blocking_findings only for concrete blockers; approve requires blocking_findings=[]. Missing scheduled host CPU results alone are a limitation at this stage. Concrete code defects, contradictory proposal checks, and declaration/implementation mismatches remain blockers. Never infer that pending generic host checks cover proposal-specific mechanism tests. Unchanged helper defaults are not evidence of a runtime protocol violation; use supplied host protocol and baseline receipts. Performance and mechanism judgments occur later.']}
    from rec_harness.code_review import validate_code_review
    def validate(value,_):
        validate_code_review(value,identity)
    review=controller._role(folder,'reviewer',request,validator=validate)
    # The controller may route a substantive rejection through bounded revisions.
    validate_code_review(review,identity,require_approval=require_approval)
    return review


def analyze(controller,folder,review,graph,report,decision_check):
    from .amazon_cycle import finalize_cycle
    contract={'base_version':graph['graph_version'],'fields':['base_version','nodes','edges'],
              'allowed_node_types':['Claim','GapHypothesis','MeasurementGap'],'origin':'agent_hypothesis',
              'task_id':graph['task_id'],'protocol_hash':controller.protocol_hash,'status':'provisional',
              'node_fields':['id','type','attrs'],
              'node_attrs_required':['task_id','protocol_hash','origin','scope','status','evidence_refs'],
              'node_attrs_template':{'task_id':graph['task_id'],'protocol_hash':controller.protocol_hash,
                  'origin':'agent_hypothesis','scope':graph['task_id'],'status':'provisional','evidence_refs':[]},
              'empty_patch':{'base_version':graph['graph_version'],'nodes':[],'edges':[]},
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
                 'Every new node.attrs must copy task_id, protocol_hash, origin, scope and status from patch_contract.node_attrs_template. Graph task_id may differ from the campaign task_id. Add evidence_refs and your scientific content inside attrs; do not omit scope on Claims.',
                 'Record candidate/control interventions, outcome differences, alternative explanations and next discriminating tests as provisional knowledge. Negative/null evidence is retained. Training success and code review do not establish causal mechanisms.',
                 'Existing historical explanation nodes may be linked by BASED_ON, SUPERSEDES or ALTERNATIVE_TO using current scoped development evidence. Never rewrite historical nodes.']}
    def validate(value,directory):
        decision_check(value['decision'],directory)
        validate_knowledge_patch(json.loads(value['knowledge']['patch_json']),graph)
    result=controller._role(folder,'analyst',request,validator=validate)
    return result['decision'],finalize_cycle(controller.study,review['bindings']['proposal_id'],folder)
