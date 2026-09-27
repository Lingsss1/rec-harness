"""Automatic source-contract and numerical admission, without per-candidate edits."""
import fcntl
from pathlib import Path
import uuid

from rec_harness.core import Artifacts,ContractError,file_hash,read_json
from rec_harness.store import Store
from .contract_gate import run_gate
from .model_source_contract import review_source
from .training_smoke import run_training_smoke


def apply_registration(config,record,artifacts):
    if record['protocol_hash']!=config['protocol_hash']:raise ContractError('model admission protocol mismatch')
    base=config['model_versions'][record['baseline']]
    if file_hash(base['path'])!=base['sha256']:raise ContractError('admission baseline changed')
    for name,item in record['versions'].items():
        if name in config['model_versions']:raise ContractError('model already admitted')
        source=item['source'];archived=artifacts.verify(item['source_ref'])
        if file_hash(source['path'])!=source['sha256'] or file_hash(archived)!=source['sha256']:
            raise ContractError('admitted model source changed')
        review_source(Path(base['path']).read_text(),Path(source['path']).read_text())
        artifacts.verify(item['probe_ref'])
        if 'training_smoke' in item:artifacts.verify(item['training_smoke']['report_ref'])
        config['model_versions'][name]=source


def admit_plan(root,proposal_id,*,recipe,uid_registry):
    from .amazon_controls import load_config
    root=Path(root).resolve()
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        store=Store(root/'memory')
        try:
            config=load_config(root,store);events=store.events()
            prior=[e['payload'] for e in events if e['kind']=='amazon_model_candidate_admitted' and e['payload']['proposal_id']==proposal_id]
            if prior:return prior[0]
            plans=[e['payload'] for e in events if e['kind']=='amazon_research_plan_registered' and e['payload']['proposal_id']==proposal_id]
            if len(plans)!=1:raise ContractError('unique preregistered plan required')
            plan=plans[0];base=plan['baseline'];version=plan['versions']['candidate'];name=version['model_version']
            controls=plan['versions']['controls'];names=[v['model_version'] for v in controls]
            reused=[v for v in names if v in config['model_versions']]
            if base not in names or plan.get('reused_control_versions')!=reused:
                raise ContractError('automatic admission requires incumbent and exact registered control identities')
            for control in controls:
                if control['model_version'] not in reused:continue
                registered=config['model_versions'][control['model_version']]
                if registered['sha256']!=control['source']['sha256'] or file_hash(registered['path'])!=registered['sha256']:
                    raise ContractError('automatic admission control source changed')
            artifacts=Artifacts(root/'artifacts/objects')
            pending=[version,*[v for v in controls if v['model_version'] not in reused]]
            reviewed=[]
            for item in pending:
                source=artifacts.verify(item['source'])
                review=review_source(Path(config['model_versions'][base]['path']).read_text(),source.read_text())
                reviewed.append((item,source,review))
            directory=root/'model-admissions'/proposal_id/uuid.uuid4().hex;directory.mkdir(parents=True)
            admitted={}
            for item,source,review in reviewed:
                model_id=item['model_version']
                target=directory if model_id==name else directory/model_id
                target.mkdir(parents=True,exist_ok=True)
                path=target/'model.py';path.write_bytes(source.read_bytes());ref={'path':str(path),'sha256':file_hash(path)}
                gate=run_gate(config['python'],ref,model_id,target/'contract',{},60)
                if gate['status']!='success':raise ContractError('model numerical contract failed; attempt preserved at '+str(target))
                probe=read_json(target/'contract/report.json');gradients=probe['checks']['parameter_gradients']
                if not gradients or not all(g.get('finite') is True for g in gradients.values()):
                    raise ContractError('model gradients nonfinite')
                smoke=run_training_smoke(config['python'],ref,target/'training-smoke',recipe,artifacts,uid_registry)
                admitted[model_id]={'source':ref,'source_ref':artifacts.put(path),
                    'probe_ref':artifacts.put(target/'contract/report.json'),'gate':gate,
                    'training_smoke':smoke,'source_contract':review}
            record={'proposal_id':proposal_id,'plan_handoff_hash':plan['handoff_hash'],'baseline':base,
                'protocol_hash':config['protocol_hash'],'source_contract':reviewed[0][2],
                'versions':admitted,
                'scope':'Atomic admission of all new planned model versions under source and CPU numerical/training contracts; no unrestricted Python admission'}
            store.event('amazon_model_candidate_admitted',record);return record
        finally:store.close()
