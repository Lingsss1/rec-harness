"""Prepare the matched baseline, then run only the frozen model autoresearch."""
from pathlib import Path
import sys,time,traceback
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'controller-source'))
from rec_harness.core import read_json,write_json,file_hash,digest,ContractError
from rec_harness.autonomous import autoresearch
from rec_harness.adapters.amazon_study import run_study
from rec_harness.adapters.amazon_autonomous import AmazonAutoresearch
from rec_harness.adapters.amazon_memory import export_development_memory

def progress(stage):
 write_json(root/'progress.json',{'stage':stage,'updated_at':time.time()})
 print(stage,flush=True)
try:
 preparation=read_json(root/'preparation.json')
 for name,sha in preparation['kernel_hashes'].items():
  if file_hash(root/'controller-source'/name)!=sha:raise ContractError('frozen controller changed')
 cfg=read_json(root/'research/amazon-auto.json')
 assert digest(cfg)==preparation['config_hash']
 assert cfg['adapter']=='amazon-nextitnet-v1' and 'measurement_discovery' not in cfg
 progress('baseline_development_training')
 run_study(root/'study',model_versions=['selected_v3'],seeds=[42],execution_id=preparation['execution_id'])
 controller=AmazonAutoresearch(root/'research',load_agent=False)
 try:
  with controller.lock():
   progress('baseline_independent_scoring')
   controller._rescore(root/'baseline','selected_v3',[42])
 finally:controller.close()
 progress('autoresearch_running')
 from rec_harness.jailed_agent import JailedCodexAgent
 from sol_medium_profile import prepare_profile
 profile=prepare_profile(cfg)
 backend=cfg['agent_backend']
 agent=JailedCodexAgent(backend['template'],backend['runtime_root'],model=backend['model'],
     timeout=cfg.get('role_timeout',300),auth_root=profile,web_search=backend.get('web_search'),
     role_skills=backend.get('role_skills'),prompts=cfg.get('research_policy',{}).get('prompts'))
 controller=AmazonAutoresearch(root/'research',agent=agent)
 try:result=controller.run()
 finally:controller.close()
 write_json(root/'result.json',result)
 controller=AmazonAutoresearch(root/'research',load_agent=False)
 try:write_json(root/'audit.json',controller.verify())
 finally:controller.close()
 export_development_memory(root/'research',root/'new-development-memory.json')
 progress('completed' if not result['failed_rounds'] else 'completed_with_failed_rounds')
except BaseException as error:
 write_json(root/'runner-error.json',{'error_kind':type(error).__name__,'error':str(error)})
 progress('failed');traceback.print_exc();raise
