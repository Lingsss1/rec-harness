"""Model-version identities; read legacy field names without rewriting archives."""
from rec_harness.core import ContractError


def model_sources(config):
    if 'model_versions' in config and 'arms' in config and config['model_versions'] != config['arms']:
        raise ContractError('conflicting model-version registry fields')
    return config['model_versions'] if 'model_versions' in config else config['arms']


def record_version(record):
    if 'model_version' in record and 'arm' in record and record['model_version'] != record['arm']:
        raise ContractError('conflicting model-version receipt identity')
    return record['model_version'] if 'model_version' in record else record['arm']
