"""Public pre/post-training contract evidence, restricted to development jobs."""
from pathlib import Path
from rec_harness.core import ContractError,digest,read_json
from rec_harness.model_graph import tensor_graph_summary,sequence_semantics_summary,parameter_sharing_summary
from .amazon_identity import model_sources,record_version


def training_contract_evidence(study,config,version,dataset,seed,result):
    """Project an audit_job-verified receipt; bind its archived probe reports."""
    if seed not in config['development_seeds'] or dataset not in config['datasets']:
        raise ContractError('training contract evidence must be development-only')
    if 'contract_gates' not in result:
        return {'status':'unavailable','scope':'Historical receipt has no pre/post-training contract gates.'}
    if result.get('status')!='success' or result.get('process',{}).get('status')!='success':
        raise ContractError('training contract receipt unsuccessful')
    if record_version(result)!=version or result.get('dataset')!=dataset or result.get('seed')!=seed:
        raise ContractError('training contract receipt identity differs')
    if not isinstance(result['contract_gates'],dict):raise ContractError('training contract gates must be an object')
    source_hash=model_sources(config)[version]['sha256']
    if result.get('source_hash')!=source_hash:raise ContractError('training contract source differs')
    job=f'{version}-{dataset}-{seed}';stages={}
    for stage in ('pre','post'):
        gate=result['contract_gates'].get(stage,{})
        if gate.get('status')!='success':raise ContractError('training contract gate incomplete: '+stage)
        report=read_json(Path(study)/'jobs'/job/(stage+'-contract')/'report.json')
        if digest({k:v for k,v in report.items() if k!='hash'})!=report.get('hash') or report['hash']!=gate.get('report_hash'):
            raise ContractError('training contract report binding differs')
        if report.get('model_id')!=job or report.get('source_hash')!=source_hash:
            raise ContractError('training contract probe identity differs')
        expected_checkpoint=result.get('checkpoint_sha256') if stage=='post' else None
        if (stage=='post' and not expected_checkpoint) or report.get('checkpoint_hash')!=expected_checkpoint:
            raise ContractError('training contract checkpoint binding differs')
        fields=('causal_input_gradient','tail_readout','empty_history_finite','padding_gradient_zero')
        if any(report.get('checks',{}).get(key) is not True for key in fields):
            raise ContractError('training contract checks incomplete')
        graph=tensor_graph_summary(report,job,source_hash)
        graph.pop('model_version',None)  # The inspector identity is the job, not a new model version.
        stages[stage]={'report_hash':report['hash'],'probe_identity':job,
            'weight_state':'fresh_initialization' if stage=='pre' else 'trained_development_checkpoint',
            'checks':{key:True for key in fields},'tensor_graph':graph,
            'sequence_semantics':sequence_semantics_summary(report),
            **({'parameter_sharing':parameter_sharing_summary(report)}
               if 'parameter_sharing' in report.get('structure',{}) else {})}
    return {'status':'verified','model_version':version,'dataset':dataset,'seed':seed,
        'source_sha256':source_hash,'receipt_hash':digest(result),'stages':stages,
        'scope':'Host-verified synthetic contracts before training and on the selected development checkpoint. No checkpoint payload, confirmation result, all-input or scientific mechanism proof.'}
