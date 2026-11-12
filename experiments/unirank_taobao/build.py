"""Prepare a new task snapshot without launching research or choosing visibility."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml
from common import read,write,sha
from selection import POLICY,PRIMARY_LABEL

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]


def build(root):
    root=root.resolve();setup=read(root/'setup.json')
    if (root/'topic-board.json').exists():raise ValueError('Do not modify a running or completed campaign')
    upstream=root/'upstream';scope=read(HERE/'research_scope.json')
    setup.update(upstream_path=str(upstream),baseline='din',research_mode='auto_gpu_topics',decision_every=6,
                 primary=PRIMARY_LABEL,selection_policy=POLICY,personal_enabled=False,
                 scope='Unified modeling research on UniRank Taobao; DIN is an initial empirical reference; whole-model architecture exploration per GPU',
                 research_objective=scope['objective'],baseline_role=scope['baseline_role'])
    write(root/'setup.json',setup)
    source=root/'source';source.mkdir(exist_ok=True)
    for name in ['common.py','baseline.py','train.py','visibility.py','controller.py','campaign.py','data.py','prepare_only.py','update_knowledge.py','budget.py','selection.py']:
        shutil.copyfile(HERE/name,source/name)
    shutil.copytree(HERE/'industry',source/'industry',dirs_exist_ok=True)
    shutil.copyfile(HERE/'research_scope.json',source/'research_scope.json')
    shutil.copyfile(HERE/'selection-policy.json',source/'selection-policy.json')
    shutil.copyfile(HERE/'selection-policy.json',root/'selection-policy.json')
    # An explicit per-campaign budget belongs to the user; rebuilding cannot
    # silently replace a requested twelve-hour rerun with the default48h.
    if not (root/'budget-policy.json').exists():
        shutil.copyfile(HERE/'budget-policy.json',root/'budget-policy.json')
    shutil.copyfile(root/'budget-policy.json',source/'budget-policy.json')
    write(root/'research-scope.json',scope)
    original=PROJECT/'experiments/taobao_ctr_dual'
    for name in ['agent.py','agent_tools.py','knowledge.py','references.py','topics.py']:
        shutil.copyfile(original/name,source/name)
    shutil.copyfile(original/'controller.py',source/'legacy_controller.py')
    design=source/'task-design'
    if design.exists():shutil.rmtree(design)
    shutil.copytree(PROJECT/'task_designs/autonomous_research_v1',design,ignore=shutil.ignore_patterns('__pycache__'))
    taskpath=design/'tasks/taobao_display_ctr/task.json';task=read(taskpath)
    task.update(id=scope['task_id'],objective=scope['objective'],status='prepared_external_task_contract',
                baseline_role=scope['baseline_role'],architecture_search=scope['architecture_search'])
    task['dataset']={
        'name':'UniRank released Taobao_Action benchmark',
        'official_url':'https://huggingface.co/datasets/salmon1802/Taobao',
        'field_reference':'https://github.com/salmon1802/UniRank',
        'source_commit':setup['upstream_commit'],'dataset_revision':read(root/'dataset-source-manifest.json')['revision'],
        'tasks':['is_click','cart','fav','buy'],
        'prediction_unit':'ad impression; binary probabilities for four tasks',
        'history':'Previous ad exposures, real advertisement/category/campaign/customer/brand/price fields and feedback action tokens; fixed visibility protocol.',
        'auxiliary_label_semantics':'Any cart/fav/buy of the same user/category/brand within 24h after impression; not advertisement-specific conversion attribution.',
        'split':'Official 2017-05-06..11 train / May12 validation / May13 held out, Asia/Shanghai',
        'limitations':['Official whole-window filtering/vocabularies and publisher static snapshots are inherited benchmark assumptions. Their production point-in-time validity is not established.']
    }
    task['research_flow']['baseline_requirements']=['A reproducible DIN local-activation reference adapted to actual fields and four labels.']
    task['research_flow']['reference_candidates_not_prescribed']=['DIN establishes the initial empirical reference. Unified candidates may replace the whole model; later controls should include the closest architectural alternatives.']
    task['research_flow']['nearest_work_check']='Check the nearest unified architecture and mechanism, record inherited components and differences, and verify primary sources before originality claims.'
    task['analysis']['primary']=PRIMARY_LABEL
    task['analysis']['selection_policy']=POLICY
    task['analysis']['secondary']=['Macro Logloss','All four task AUC/Logloss and changes against reference/entry','Parameters, training cost, inference cost']
    task['implementation_boundary']['agent_editable']=['candidate model modules','candidate training-loss hook','prospective experiment plans','evidence-linked agent knowledge']
    task['final_assessment']['test']='Closed during autonomous research. A final suite must be frozen before any held-out scoring; no automatic test access from research loop.'
    task['readiness_gates']=['Verify pinned released parquet digests','Audit actual loader fields/labels/prefix semantics','Resolve and freeze visibility policy','Pass forward/backward/masking tests','Measure DIN full-unit cost','Freeze protocol before research']
    write(taskpath,task)
    prior=Path(read(root/'setup.json')['previous_campaign'])/'knowledge-assets-v2'
    if not (root/'knowledge-assets-v2').exists():shutil.copytree(prior,root/'knowledge-assets-v2')
    from update_knowledge import update
    update(root)
    (root/'config').mkdir(exist_ok=True)
    ds=yaml.safe_load((upstream/'config/dataset_config.yaml').read_text())['Taobao_Action']
    ds['data_root']=str(root/'data')
    for split in ['train','valid','test']:
        for suffix,folder in [('data','data'),('user_info','user_info'),('item_info','item_info')]:
            ds[split+'_'+suffix]=str(root/'data/Taobao_Action'/split/folder)
    ds['block_cache_size']=2
    (root/'config/dataset_config.yaml').write_text(yaml.safe_dump({'Taobao_Action':ds},sort_keys=False))
    official=yaml.safe_load((upstream/'config/model_config.yaml').read_text())
    base=official['Base'];base.update(enable_torch_compile=False,group_id=None,num_workers=0)
    model={key:value for key,value in official['HeMix_Taobao_Action'].items() if key in [
        'dataset_id','loss','metrics','task','batch_size','embedding_dim','tower_hidden_units','num_tasks',
        'epochs','shuffle','seed','monitor','monitor_mode','max_len','accumulation_steps']}
    model.update(model='DIN',attention_hidden_units=[64,32])
    (root/'config/model_config.yaml').write_text(yaml.safe_dump({'Base':base,'DIN_Taobao_Action':model},sort_keys=False))
    prepared=root/'prepared'
    if not prepared.exists():prepared.symlink_to('data/Taobao_Action',target_is_directory=True)
    write(root/'topics.json',{'topics':scope['topics']})
    from budget import refresh
    refresh(root)
    write(root/'source-snapshot.json',{'created_unix':time.time(),'sources':{p.name:sha(p) for p in source.glob('*.py')},
        'generic_instructions_sha256':sha(design/'agent/instructions.txt'),
        'research_scope_sha256':sha(source/'research_scope.json'),
        'budget_policy_sha256':sha(source/'budget-policy.json'),
        'selection_policy_sha256':sha(source/'selection-policy.json'),
        'knowledge_bundle_identity':read(root/'knowledge-update.json')['bundle_identity'],
        'original_generic_instructions_sha256':sha(PROJECT/'task_designs/autonomous_research_v1/agent/instructions.txt')})
    with (root/'environment-freeze.txt').open('w') as stream:
        subprocess.run([sys.executable,'-m','pip','freeze'],stdout=stream,check=True)
    write(root/'environment.json',{'python':sys.executable,'python_version':sys.version,
        'deviations':['Internal mirror torch 2.8.0+cu128; verified A800 CUDA execution.',
                      'Optional heavyball omitted: pinned heavyball 3.2 requires numpy>=2.3.4, incompatible with repository Python3.10. DIN uses AdamW/Adagrad.',
                      'No torch.compile; BF16 autocast; TF32 disabled; one GPU per model. Full validation retained including partial batches.'],
        'tests':'Four contract tests passed: forward/backward without labels, padding, strict visibility, official/strict counterexample.'})
    print(json.dumps({'root':str(root),'baseline':'din','phase':'prepared_policy_pending'},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();build(args.root)
