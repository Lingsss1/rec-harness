"""Append-only, mechanically constrained controls for the frozen Amazon study.

This is not arbitrary candidate admission: only a reviewed one-line removal of
the incumbent's prefix mixing is supported while GPU candidate isolation is pending.
"""
import fcntl
from pathlib import Path

from rec_harness.core import Artifacts, ContractError, digest, file_hash, read_json, write_json
from rec_harness.store import Store
from .amazon_identity import model_sources, record_version

CONTROL = 'legacy_no_prefix'
OLD = '        blended = 0.85 * x + 0.15 * context'
NEW = '        blended = x'


def control_source(source):
    if source.count(OLD) != 1:
        raise ContractError('incumbent does not match the reviewed prefix control transformation')
    return source.replace(OLD, NEW)


def load_config(root, store=None):
    root = Path(root)
    config = read_json(root / 'study.json')
    if digest(config) != read_json(root / 'study-lock.json')['config_hash']:
        raise ContractError('study configuration changed')
    config['model_versions'] = dict(model_sources(config))
    config.pop('arms', None)
    own_store = store is None
    store = store or Store(root / 'memory')
    try:
        store.verify()
        for event in store.events():
            if event['kind']=='amazon_model_candidate_admitted':
                from .amazon_admission import apply_registration
                apply_registration(config,event['payload'],Artifacts(root/'artifacts/objects'))
                continue
            if event['kind'] == 'amazon_prefix_candidate_admitted':
                from .amazon_prefix_candidate import apply_registration
                apply_registration(config, event['payload'], Artifacts(root / 'artifacts/objects'))
                continue
            if event['kind'] != 'amazon_control_registered': continue
            item = event['payload']
            if record_version(item) != CONTROL or item['protocol_hash'] != config['protocol_hash']:
                raise ContractError('unsupported control registration')
            if CONTROL in config['model_versions']: raise ContractError('duplicate control registration')
            expected = control_source(Path(config['model_versions']['legacy_incumbent']['path']).read_text())
            if Path(item['source']['path']).read_text() != expected or file_hash(item['source']['path']) != item['source']['sha256']:
                raise ContractError('registered control source changed')
            artifacts = Artifacts(root / 'artifacts/objects')
            artifacts.verify(item['probe_ref']); artifacts.verify(item['source_ref'])
            config['model_versions'][CONTROL] = item['source']
        return config
    finally:
        if own_store: store.close()


def prepare_control(root):
    root = Path(root)
    config = load_config(root)
    if CONTROL in config['model_versions']: return Path(config['model_versions'][CONTROL]['path'])
    baseline = config['model_versions']['legacy_incumbent']
    if file_hash(baseline['path']) != baseline['sha256']: raise ContractError('incumbent source changed')
    output = root / 'frozen-controls' / (CONTROL + '.py')
    contents = control_source(Path(baseline['path']).read_text())
    output.parent.mkdir(exist_ok=True)
    if output.exists() and output.read_text() != contents: raise ContractError('control file already differs')
    if not output.exists(): output.write_text(contents)
    return output


def register_control(root, probe_path):
    root = Path(root)
    with (root / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store = Store(root / 'memory')
        try:
            config = load_config(root, store)
            if CONTROL in config['model_versions']: return {'already_registered': True}
            source = root / 'frozen-controls' / (CONTROL + '.py')
            expected = control_source(Path(config['model_versions']['legacy_incumbent']['path']).read_text())
            if source.read_text() != expected: raise ContractError('control exceeds reviewed transformation')
            probe = read_json(probe_path)
            if digest({k: v for k, v in probe.items() if k != 'hash'}) != probe['hash'] or probe['source_hash'] != file_hash(source):
                raise ContractError('control probe/source identity mismatch')
            if not all(probe['checks'].get(k) is True for k in ['causal_input_gradient', 'tail_readout', 'empty_history_finite', 'padding_gradient_zero']):
                raise ContractError('control correctness gate failed')
            gradients = probe['checks']['parameter_gradients']
            if not gradients or not all(g.get('finite') is True for g in gradients.values()):
                raise ContractError('control gradient gate failed')
            if list((root / 'jobs').glob(CONTROL + '-*/completed.json')):
                raise ContractError('control outcomes exist before preregistration')
            artifacts = Artifacts(root / 'artifacts/objects')
            record = {'model_version': CONTROL, 'kind': 'mechanism_control', 'base_model_version': 'legacy_incumbent',
                'source': {'path': str(source.resolve()), 'sha256': file_hash(source)},
                'source_ref': artifacts.put(source), 'probe_ref': artifacts.put(probe_path),
                'protocol_hash': config['protocol_hash'], 'seeds': config['development_seeds'] + config['confirmation_seeds'],
                'contrast': 'legacy_incumbent minus legacy_no_prefix; all other source bytes identical',
                'selection_role': 'attribution_only_not_a_new_Codex_candidate',
                'prior_exposure': 'baseline/incumbent validation outcomes for all listed seeds already exist',
                'cost_policy': 'same study GPU budget; no reset or free control training',
                'confirmation_scope': 'new control runs with fixed contrast; not fresh held-out users or a new independent benchmark'}
            store.event('amazon_control_registered', record)
            write_json(root / 'control-registration.json', record)
            return record
        finally:
            store.close()
