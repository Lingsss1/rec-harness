import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path('experiments/taobao_ctr_dual').absolute()))
from controller import knowledge_resources,DESIGN
from agent import run
from common import read,write
from knowledge import scoped_cards
root=Path('artifacts/taobao-ctr-dual-20261104').absolute()
protocol=read(root/'protocol.json');state=read(root/'enhanced/state.json')
industry=[json.loads(line) for line in (DESIGN/'tasks/taobao_display_ctr/industry/cards.jsonl').read_text().splitlines()]
papers=read(DESIGN/'tasks/taobao_display_ctr/paper_sources.json')
evidence=scoped_cards(industry,papers,state['memory'],protocol['data_identity'],protocol['identity'])
runtime={'phase':'knowledge_preflight','expected_files':['ready.json'],
'action':'Validate this public evidence transport. Resolve source_registry_ref and queryformer-paper-v1; read queryformer-paper-v1 --find attention --lines 8; search --source industry query attention compression; read unidac-detailed-report to check the labeled fallback. Write ready.json containing available references, source version, text hash, returned line range, search IDs, fallback reference, and any failures. Do not propose models or evaluate any scores.'}
workspace={'source_registry_ref':'industry/sources.json',
'api':'python3 /workspace/tools.py resolve REF; python3 /workspace/tools.py read REF [--find TEXT] [--lines N]; python3 /workspace/tools.py search [--source industry|paper|agent] TERMS',
'personal':'disabled'}
receipt=run(root/'knowledge-preflight',{'kind':'evidence_transport_verification','training':'not authorized'},runtime,workspace,evidence,240,resources=knowledge_resources(root,state,evidence,protocol))
write(root/'knowledge-preflight-verification.json',{'receipt':receipt,'result':read(root/'knowledge-preflight/ready.json'),'budget_counted':True,'deadline_unix':read(root/'budget.json')['deadline_unix']})
print(json.dumps({'status':receipt['status'],'wall_seconds':receipt['wall_seconds'],'fixed_instructions_sha256':receipt['fixed_instructions_sha256']}))
