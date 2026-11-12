from pathlib import Path
import copy,io,subprocess,sys,tarfile
repo=Path('/home/liuzhishan/lhr/rec-harness-v5')
root=repo.parent/'rec-harness-runs/amazon-autoresearch-next-v9'
assert not root.exists()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
assert commit.startswith('358b2b1')
root.mkdir();frozen=root/'controller-source';frozen.mkdir()
with tarfile.open(fileobj=io.BytesIO(subprocess.check_output(['git','archive',commit,'rec_harness','examples','pyproject.toml','tests','README.md','docs/implementation'],cwd=repo))) as archive:archive.extractall(frozen)
sys.path.insert(0,str(frozen))
from rec_harness.core import Artifacts,digest,file_hash,read_json,write_json
from rec_harness.store import Store
from rec_harness.cost_audit import audit_cost
from rec_harness.amazon_tool_online import create_study
from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
from rec_harness.adapters.amazon_memory import validate_memory
prior=root.parent/'amazon-autoresearch-next-v8'
old=read_json(root.parent/'amazon-input-offset-recovery-v1/config.json')
from rec_harness.adapters.amazon_controls import load_config
source=load_config(prior/'study')
base=read_json(prior/'result.json')['state']['incumbent']
assert base=='amazon-next-v8-round-1-control-1'
model=Path(source['model_versions'][base]['path'])
assert file_hash(model)==source['model_versions'][base]['sha256']
assert digest(source['protocol'])==source['protocol_hash']
lineage={'source_commit':commit,'base_model':base,'base_source_hash':file_hash(model),'predecessor':str(prior),'scope':'Model-only autoresearch continuation from the v8 promoted model; historical development evidence only; no tool/Harness evolution or final test. Fresh single-seed study with retained historical costs; old policies are not rewritten.'}
for name in ['study','research']:
 store=Store(prior/name/'memory',read_only=True)
 try:
  cost=audit_cost(store,store.budget()['limit']);assert not cost['pending'];lineage[name]=cost
 finally:store.close()
review=root.parent/'amazon-autoresearch-next-v5-repair-v1/review'
assert read_json(review/'review-result.json')['status']=='completed'
objects=Artifacts(root/'objects')
study=copy.deepcopy(source)
for key in ['model_versions','worker','predecessor','continuation_required']:study.pop(key,None)
study.update(study_id='amazon-model-autoresearch-v9',development_seeds=[42],confirmation_seeds=[],selection_policy='baseline-single-seed-v1',gpu_ids=[4,5,6,7],budget_gpu_seconds=1200000)
worker=frozen/'rec_harness/adapters/amazon_worker.py';study['worker_sha256']=file_hash(worker)
for name,sha in study['data_hashes'].items():assert file_hash(Path(study['data'])/name)==sha
case={'id':'model-v9','study':study,'models':{base:objects.put(model)},'worker':objects.put(worker),'execution_recipe':old['recipe']}
execution=create_study(case,root/'study',objects)
store=Store(root/'study/memory')
try:
 amount=lineage['study']['charged_or_reserved'];store.reserve('predecessor-consumed',amount);store.settle('predecessor-consumed',amount,'retained')
finally:store.close()
packs=[validate_memory(read_json(prior/name),study['protocol_hash'],[42]) for name in ['development-memory.json','new-development-memory.json']]
assert all(pack['schema']=='amazon-development-memory-v1' for pack in packs)
entries=[entry for pack in packs for entry in pack['entries']]
assert len({entry['round'] for entry in entries})==len(entries)
body={'schema':'amazon-development-memory-v1','protocol_hash':study['protocol_hash'],'entries':entries}
memory=validate_memory({**body,'hash':digest(body)},study['protocol_hash'],[42])
write_json(root/'development-memory.json',memory)
config={'adapter':'amazon-nextitnet-v1','study':str(root/'study'),'task_id':'amazon-next-v9','max_rounds':20,
 'base_model_id':base,'execution_id':execution,'role_call_limit':int(lineage['research']['charged_or_reserved'])+400,'role_timeout':600,
 'review_revision_limit':2,'role_layout':'compact-four-v1','hypothesis_graph':True,'selection_policy':'baseline-single-seed-v1',
 'generated_control_count':1,'history_diagnostic':True,'minimum_round_gpu_seconds':5000,
 'development_memory_files':[str(root/'development-memory.json'),str(review.parent/'recovered-development-memory.json')],
 'source_history_studies':[str(prior/'study'),str(root.parent/'amazon-autoresearch-next-v5/study'),str(root.parent/'amazon-autoresearch-next-v4/study')],
 'research_policy':{'retrieval_limit':24},
 'semantic_encoder':{'model_path':'/root/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3','query':'Improve Amazon NextItNet next-item ranking beyond the current selected model. Retrieve prior mechanisms, failed alternatives, development evidence and discriminating controls.','max_length':2048,'batch_size':4,'threads':2},
 'rescore_recipe':old['recipe'],'uid_registry':str(root/'cpu-uids'),
 'agent_backend':{'kind':'jailed_codex','model':'gpt-6-sol','reasoning_effort':'medium','template':str(root.parent/'codex-jail-v1'),'runtime_root':'/tmp/rec-harness-amazon-next-v9-jails','web_search':{'researcher':'live'}}}
write_json(root/'config.json',config);write_json(root/'lineage.json',lineage)
AmazonAutoresearch.init(root/'research',config)
controller=AmazonAutoresearch(root/'research',load_agent=False)
try:
 amount=lineage['research']['charged_or_reserved'];controller.store.reserve('predecessor-consumed',amount);controller.store.settle('predecessor-consumed',amount,'retained')
 controller.store.event('next_campaign_lineage',{'lineage_hash':digest(lineage),'artifact':controller.artifacts.put(root/'lineage.json')})
finally:controller.close()
write_json(root/'preparation.json',{'status':'ready','source_commit':commit,'execution_id':execution,'gpu_ids':study['gpu_ids'],'development_seeds':study['development_seeds'],'confirmation_seeds':study['confirmation_seeds'],'new_role_call_limit':400,'rounds':20,'model_hash':file_hash(model),'config_hash':digest(read_json(root/'research/amazon-auto.json')),'kernel_hashes':{str(p.relative_to(frozen)):file_hash(p) for p in (frozen/'rec_harness').rglob('*.py')},'scope':lineage['scope']})
print({'status':'ready','root':str(root),'source_commit':commit,'history_entries':len(memory['entries']),'new_codex_call_limit':400})

import shutil
shutil.copyfile(__file__,root/'prepare.py')
