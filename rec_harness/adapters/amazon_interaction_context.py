"""Freeze development-only factorial block measurements for research roles."""
from pathlib import Path
from rec_harness.core import Artifacts,ContractError,digest,file_hash,read_json,write_json,number
from rec_harness.diagnostics import validate_measurement
from .amazon_controls import load_config
from .amazon_report import audit_job
from .block_interaction import summarize_interaction
from . import block_interaction


def freeze_interactions(groups,study,artifacts):
    records=[];seen=set()
    for group in groups:
        if set(group)!={'study','directories'}:raise ContractError('interaction group requires study and four directories')
        source_root=Path(group['study']);source=load_config(source_root)
        if len(group['directories'])!=4:raise ContractError('four interaction conditions required')
        for directory in map(Path,group['directories']):
            metadata=read_json(directory/'acceptance.json')
            if metadata['seed'] not in study['development_seeds'] or metadata['dataset'] not in study['datasets']:
                raise ContractError('interaction must match frozen development conditions')
        report=summarize_interaction(group['directories']);binding=report['bindings']
        model= binding['model_version'];dataset=binding['dataset'];seed=binding['seed']
        if (binding['protocol_hash']!=study['protocol_hash'] or source['protocol_hash']!=study['protocol_hash']
            or source['data_hashes']!=study['data_hashes'] or dataset not in study['datasets']
            or seed not in study['development_seeds']):
            raise ContractError('interaction must match frozen development conditions')
        if model not in study['model_versions'] or study['model_versions'][model]['sha256']!=binding['source_hash']:
            raise ContractError('interaction model source differs')
        identity=(model,dataset,seed,tuple(report['blocks']))
        if identity in seen:raise ContractError('duplicate interaction group')
        seen.add(identity)
        baseline,_=audit_job(source_root,source,model,dataset,seed)
        if (baseline['source_hash'],baseline['checkpoint_sha256'],baseline['candidates_hash'])!=(binding['source_hash'],binding['checkpoint_hash'],binding['candidates_hash']):
            raise ContractError('interaction checkpoint or candidates differ from audited baseline')
        baseline_ranks=read_json(source_root/'jobs'/f'{model}-{dataset}-{seed}'/'validation_ranks.json')
        objects=Artifacts(source_root/'artifacts/objects');costs={};condition_refs=[]
        for directory in map(Path,group['directories']):
            value=read_json(directory/'acceptance.json');receipt=read_json(directory/'inference/receipt.json')
            submitted=read_json(directory/'inference/request.json')
            if receipt['process']['status']!='success':raise ContractError('interaction inference was not successful')
            scores=objects.verify(receipt['collection']['artifacts']['scores.json'])
            if file_hash(scores)!=file_hash(directory/'inference/results/scores.json'):
                raise ContractError('interaction local scores differ from collected artifact')
            for name,expected in [('model.py',binding['source_hash']),('checkpoint.pt',binding['checkpoint_hash']),
                                  ('worker.py',binding['score_worker_hash']),('request.json',file_hash(directory/'inference-request.json'))]:
                if submitted['inputs'][name]['sha256']!=expected:raise ContractError('interaction inference input binding differs')
            if binding['score_worker_hash']!=file_hash(Path(__file__).with_name('score_worker.py')):
                raise ContractError('interaction worker differs from reviewed implementation')
            blocks=value.get('bypass_blocks',[value['bypass_block']] if 'bypass_block' in value else [])
            arguments=submitted['arguments'];actual=[int(arguments[i+1]) for i,item in enumerate(arguments) if item=='--bypass-block']
            if sorted(actual)!=sorted(blocks):raise ContractError('interaction execution arguments differ')
            if not blocks:
                ranked=read_json(directory/'host-ranking.json')
                if (ranked['hash']!=value['host_report_hash'] or ranked['hash']!=digest({k:v for k,v in ranked.items() if k!='hash'})
                    or ranked['user_ids']!=baseline_ranks['user_ids'] or ranked['ranks']!=baseline_ranks['ranks']):
                    raise ContractError('interaction reference differs from audited ranks')
            if receipt['wall_seconds'] is not None:number(receipt['wall_seconds'],'interaction inference time',0)
            costs[digest(receipt)]=receipt['wall_seconds']
            condition_refs.append(artifacts.put(directory/'acceptance.json'))
        measurements={dataset+'.users':binding['users']}
        prefix=dataset+'.blocks-'+'-'.join(map(str,report['blocks']))+'.'
        for metric,values in report['metrics'].items():
            measurements.update({prefix+metric+'.'+kind:value for kind,value in values.items()})
        public={'observation_id':'block-interaction-'+report['hash'][:32],'model_version':model,'dataset':dataset,'seed':seed,
            'measurements':measurements,'limitations':[report['qualification'],
                'Exploratory fixed-checkpoint diagnostic; not a utility-promoted autonomous tool or new model-training evidence.']}
        validate_measurement({key:public[key] for key in ['measurements','limitations']})
        path=artifacts.root.parent/'block-interactions'/(digest(public)+'.json');write_json(path,public)
        records.append({'model_version':model,'protocol_hash':study['protocol_hash'],'source_hash':binding['source_hash'],
                        'public_ref':artifacts.put(path),'source_receipts':condition_refs,'inference_costs':costs,
                        'diagnostic_kind':'block_interaction','implementation_hash':digest({
                            'worker':binding['score_worker_hash'],'scorer':binding['host_scorer_hash'],
                            'contrast':file_hash(block_interaction.__file__),'projection':file_hash(__file__)}),
                        'report_hash':report['hash']})
    return records
