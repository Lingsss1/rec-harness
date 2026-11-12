"""Bind observed Codex search events to role archives without claiming source truth."""
import json
from pathlib import Path
from .core import ContractError,digest,file_hash,read_json


def receipt(log,mode,response=None):
    log=Path(log)
    events=[]
    if log.exists():
        for line in log.read_text().splitlines():
            try:event=json.loads(line)
            except ValueError:continue
            if (isinstance(event,dict) and isinstance(event.get('item'),dict)
                and event['item'].get('type') in {'web_search','web_search_call'}):events.append(event)
    return {'schema':'codex-literature-search-v2','mode':mode,
        'observed_search_events':events,'search_observed':bool(events),
        'reported_sources':(response or {}).get('method',{}).get('sources',[]),
        'sources_status':'agent_reported_unverified',
        'response_hash':digest(response) if response is not None else None,
        'stdout_sha256':file_hash(log) if log.exists() else None,
        'qualification':'CLI events and agent-reported citations; not source-content or mechanism verification'}


def from_archive(archive,artifacts):
    ref=archive.get('literature-search.json')
    if ref is None:return None
    value=read_json(artifacts.verify(ref))
    if value.get('schema')=='codex-literature-search-v1':return None  # Preserve historical trace hashes.
    if value.get('schema')!='codex-literature-search-v2':raise ContractError('unknown literature receipt schema')
    if value.get('mode') not in {'live','cached','disabled'}:raise ContractError('invalid archived search mode')
    if 'stdout.jsonl' not in archive:raise ContractError('literature receipt lacks archived CLI log')
    log=artifacts.verify(archive['stdout.jsonl'])
    response=None
    if value.get('response_hash') is not None:
        if 'response.json' not in archive:raise ContractError('literature receipt lacks bound response')
        response=read_json(artifacts.verify(archive['response.json']))
    if value!=receipt(log,value['mode'],response):raise ContractError('literature receipt differs from archived CLI evidence')
    return {**value,'receipt_ref':ref,'stdout_ref':archive['stdout.jsonl'],
        'response_ref':archive.get('response.json') if response is not None else None,
        'observation_scope':'Observed during this role call, not part of the pre-call visible input. Search event presence does not prove successful retrieval or support for a citation.'}
