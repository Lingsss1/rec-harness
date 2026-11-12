"""Host-owned scalar output activation summaries from existing training receipts."""
import math
from .core import ContractError,file_hash,number,read_json
from .toy_isolated import samples,score_response

DATASET='synthetic-output'


def freeze_options(config):
    enabled=config['protocol'].get('output_activation_observations',False)
    if type(enabled) is not bool:raise ContractError('output_activation_observations must be boolean')
    if not enabled:return
    execution=config.get('execution',{})
    if execution.get('kind')!='isolated-toy-v1':raise ContractError('output observations require isolated host scoring')
    expected=file_hash(__file__)
    if execution.get('output_observation_producer_hash',expected)!=expected:
        raise ContractError('output observation producer changed')
    execution['output_observation_producer_hash']=expected


def summarize(predictions,queries):
    if not predictions or len(predictions)!=len(queries):raise ContractError('output observation sample mismatch')
    for value in predictions:number(value,'prediction')
    count=len(predictions);mean=sum(value/count for value in predictions)
    rms=math.hypot(*predictions)/math.sqrt(count)
    std=math.hypot(*(value-mean for value in predictions))/math.sqrt(count)
    values={'output.count':count,'output.mean':mean,'output.rms':rms,'output.std':std}
    limitations=['Scalar final model outputs, not hidden-layer activations or causal interventions.',
                  'Two reproducibility processes share one fit specification; they are not independent training evidence.']
    for positive,label in [(True,'nonnegative'),(False,'negative')]:
        group=[value for value,query in zip(predictions,queries) if (query[0]>=0)==positive]
        prefix='input0.'+label
        values[prefix+'.count']=len(group)
        if group:values[prefix+'.output_mean']=sum(value/len(group) for value in group)
        else:limitations.append(prefix+' has no samples; its output mean is unavailable.')
    for value in values.values():number(value,'output observation')
    return {'measurements':values,'limitations':limitations}


def from_receipts(execution,artifacts,protocol,seed,model,run_id):
    receipts=execution.get('receipts',[])
    if not isinstance(receipts,list) or len(receipts)!=2:raise ContractError('output observations require two reproducible prediction receipts')
    responses=[]
    for receipt in receipts:
        try:
            if receipt['process']['status']!='success':raise ContractError('output observation prediction process failed')
            ref=receipt['collection']['artifacts']['predictions.json']
        except (KeyError,TypeError) as error:raise ContractError('output observation prediction receipt incomplete') from error
        responses.append(read_json(artifacts.verify(ref)))
    if responses[0]!=responses[1]:raise ContractError('output observation predictions differ between processes')
    task=protocol.get('synthetic_task')
    validation=samples(seed+10000,task['validation_rows'] if task else 120,task)
    metrics=score_response(responses[0],[target for _,target in validation])
    summary=summarize(responses[0]['predictions'],[query for query,_ in validation])
    return metrics,{'observation_id':run_id+'-output-activation','model_version':model,'dataset':DATASET,'seed':seed,
        **summary,'scope':'Scalar output activations on synthetic development queries; no validation targets, weights or per-example predictions in this summary.'}


def context(state):
    rows=[run['output_activation_observation'] for run in (state.get('baseline') or {}).get('runs',[])
          if run.get('valid') is True and 'output_activation_observation' in run]
    return {'activation_diagnostics':rows} if rows else {}
