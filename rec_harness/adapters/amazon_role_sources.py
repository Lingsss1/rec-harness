"""Check every source promised by a generated-control role before reuse."""
from rec_harness.core import ContractError,file_hash,safe_file


def verify_generated_sources(directory,role,context,accepted):
    generated=context.get('generated_controls',[]) if role=='experiment' else []
    hashes=accepted.get('model_source_hashes')
    if not generated and hashes is None:return
    expected=['model.py',*[item['file'] for item in generated]]
    if not isinstance(hashes,dict) or len(set(expected))!=len(expected) or set(hashes)!=set(expected):
        raise ContractError('incomplete generated model source bindings')
    if hashes['model.py']!=accepted.get('source_hash'):
        raise ContractError('generated candidate source binding differs')
    for name,sha in hashes.items():
        if file_hash(safe_file(directory/'model',name))!=sha:
            raise ContractError('model source changed: '+name)
