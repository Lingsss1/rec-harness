"""Taobao adapter; retain the original auto research loop and generic agent."""
import copy
from pathlib import Path
import shutil
import sys

import legacy_controller as legacy
from legacy_controller import *
from selection import POLICY,PRIMARY_LABEL,best_trial,trial_score,trial_logloss,task_deltas,validate_policy

HERE=Path(__file__).resolve().parent
_original_contract=legacy.phase_contract
_original_resources=legacy.knowledge_resources


def phase_contract(phase,mode,state,protocol,remaining):
    contract=_original_contract(phase,mode,state,protocol,remaining)
    contract['context']['budget_scope']='The research wall clock starts at protocol.research_budget.start_unix under its recorded start_event. Original acquisition uses verified download completion; an explicitly requested rerun on already verified data has its own fixed campaign start. Every later preparation step, calibration, agent call, failure, training/control and scoring job counts.'
    validate_policy(protocol['selection_policy'])
    contract['context']['selection_policy']=protocol['selection_policy']
    trials={row['id']:row for row in state['trials']}
    for entry in contract['context']['trajectory_index']:
        entry['per_task']=trials[entry['id']].get('metrics',{}).get('validation',{}).get('per_task',{})
    if 'candidate_api' not in contract:return contract
    contract['candidate_api']={
        'Model':'Model(feature_map, model_id=..., gpu=0, **kwargs) -> torch.nn.Module; optional MultiTaskModel utilities. forward(inputs) returns {label+"_pred": probability [B,1]} for all four tasks. Host places the module on the assigned GPU.',
        'architecture_scope':'Whole-model design for unified modeling. Model need not inherit the initial reference, retain its pooling, or use separate task towers; representations, interactions, routing and readout can all be replaced.',
        'lineage':'parent_trial records the empirical comparison and evidence lineage. A complete rewrite may have no inherited architectural components.',
        'proposal_unit':'Coordinated structural changes implementing one coherent mechanism are allowed; isolate their effects through planned replacement, component or factorial controls.',
        'meta':'Read-only UniRank FeatureMap: features (ordered specs, sources, vocab_size), labels, data_dir. Full framework reference files are available through tools.py.',
        'batch':[
            'inputs=(batch_dict,item_dict,history_mask,multi_task_history_masks)',
            'batch_dict: categorical user/context features int64[B]; current labels and lookup metadata are removed by host',
            'item_dict: item_id,cate_id,campaign_id,customer_id,brand,price_bucket,action, each int64[B,101]; last position is current target, preceding100 are pre-padded history',
            'history_mask: float 0/1 [B,100]; four task masks: [B,100]; use only masked history',
            'Target action is always zero (unknown/padding), never a current label. Historical feedback policy is frozen in protocol.history_policy.',
            'Historical item_id is a previous advertisement ID. Shopping events used to create auxiliary labels are only matched by user/category/brand.',
            'No raw timestamps, event-time deltas, raw user IDs, full future histories, current labels or global sample indexes are supplied to forward.'
        ],
        'task_labels':['is_click','cart','fav','buy'],
        'primary':'Validation macro AUC = (is_click AUC + cart AUC + fav AUC + buy AUC)/4. Checkpoint, incumbent, topic gains and final selection all use this rule. Exact ties prefer lower mean Logloss, then fewer parameters, then trial ID. Report every task change; explain trade-offs.',
        'optional':[
            'Literal CONFIG dict for model architecture only; all frozen host parameters are rejected.',
            'Model.training_loss(inputs, probabilities[B,4], train_labels[B,4]) -> finite differentiable scalar; labels supplied only during training. Default is sum of four unweighted BCEs.'
        ],
        'ownership':'Edit candidate modules/proposal only. No filesystem/network operations in model source. Imports: torch,numpy,math,typing,copy,functools,collections,itertools,dataclasses,contextlib,unirank,model_zoo. Host owns training, input filtering and scoring.',
        'local_checks':'Coding jail has no torch; check syntax/source there. Host trains and scores. Do not install or train inside coding jail.',
        'training':'From scratch; same frozen data, seed, epochs, batch, history limit and dense AdamW / embedding Adagrad. Architecture may be rewritten completely. Retain causal controls and account for added parameters/computation.'
    }
    return contract


def make_task(protocol):
    task=read(DESIGN/'tasks/taobao_display_ctr/task.json')
    task.update(status='active_external_task_contract',protocol_ref='runtime contract: protocol',protocol_hash=protocol['identity'])
    task['dataset']['history_policy']=protocol['history_policy']
    task['analysis']['primary']=PRIMARY_LABEL
    task['analysis']['selection_policy']=validate_policy(protocol['selection_policy'])
    count=len(task['knowledge']['industry']['members'])
    task['knowledge']['industry']['delivery']=f'{count} UniRank comparison-method cards, available on demand through tools.py. Inclusion is external knowledge, not another baseline or a local result.'
    task['knowledge']['agent']['delivery']='This topic\'s receipts and explicit imported controls only; old campaigns are not local evidence.'
    return task


def knowledge_resources(root,state,evidence,protocol):
    resources=_original_resources(root,state,evidence,protocol)
    upstream=Path(read(root/'setup.json')['upstream_path'])
    for name in [
        'unirank/features.py','unirank/pytorch/models/rank_model.py','unirank/pytorch/models/multitask_model.py',
        'unirank/pytorch/layers/embeddings/feature_embedding.py',
        'unirank/pytorch/layers/activations.py','unirank/pytorch/layers/attentions/target_attention.py',
        'unirank/pytorch/dataloaders/unirank_dataloader.py'
    ]:
        path=upstream/name
        if path.is_file():
            resources.append({'id':'framework:'+name,'aliases':[name],
                'metadata':{'source':'primary upstream implementation','path':name,'version':read(root/'setup.json')['upstream_commit'],
                            'content_sha256':sha(path),'scope':'read-only API reference; does not prescribe a model'},
                'text_path':str(path)})
    return resources


def train_job(data,source,config,out,seed,gpu,protocol,deadline,calibration=0):
    root=Path(protocol['campaign_root']);out.mkdir(parents=True,exist_ok=True)
    frozen=out/'model.py'
    if not frozen.exists():shutil.copyfile(source,frozen)
    write(out/'model-config.json',config)
    command=[sys.executable,'-u',str(HERE/'train.py'),'--campaign',str(root),'--data',str(data),
             '--model',str(frozen),'--config',str(out/'model-config.json'),'--output',str(out),'--seed',str(seed)]
    if calibration:command+=['--calibration-steps',str(calibration)]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    result=bounded(command,out,min(protocol.get('trial_cap_seconds',21600),max(1,deadline-time.time())),env)
    receipt={'seed':seed,'gpu':gpu,'source_sha256':sha(frozen),'config':config,'output':str(out),**result}
    if result['status']=='success' and (out/'result.json').is_file():receipt['metrics']=read(out/'result.json')
    else:receipt.update(status='failed',error=read(out/'failure.json') if (out/'failure.json').is_file() else 'Inspect process.log')
    write(out/'receipt.json',receipt);return receipt


# These original functions resolve globals in legacy_controller, including from
# inside researcher(). Only task-specific adapter functions change.
legacy.phase_contract=phase_contract
legacy.make_task=make_task
legacy.knowledge_resources=knowledge_resources
legacy.train_job=train_job
legacy.best_trial=best_trial
legacy.trial_score=trial_score
legacy.trial_logloss=trial_logloss
legacy.task_deltas=task_deltas
