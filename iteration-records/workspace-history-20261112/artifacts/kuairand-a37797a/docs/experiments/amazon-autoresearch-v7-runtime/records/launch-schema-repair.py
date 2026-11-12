from pathlib import Path
import os,subprocess,sys
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'controller-source'))
from rec_harness.core import write_json
from rec_harness.process_identity import identity
assert not (root/'launch-schema-repair.json').exists(),'already launched; inspect recorded process'
environment=os.environ.copy()
environment.update(PYTHONPATH=str(root/'controller-source'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
assert environment.get('https_proxy'),'validated service network setup is required'
with (root/'controller-schema-repair.log').open('x') as log:
 process=subprocess.Popen([sys.executable,str(root/'controller-schema-repair.py')],cwd=root,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write_json(root/'launch-schema-repair.json',identity(process.pid))
print({'pid':process.pid,'campaign':str(root/'research'),'mode':'model-only-autoresearch','rounds':10})
