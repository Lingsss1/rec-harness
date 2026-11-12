"""Content-addressed primary-source acquisition with explicit retrieval failures."""
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .core import ContractError, digest, read_json, write_json

SEEDS = [
    {'url': 'https://arxiv.org/html/2402.17152v1', 'version': '2402.17152v1', 'locator': 'HSTU architecture', 'implementation': 'models.py:HSTUEncoder'},
    {'url': 'https://arxiv.org/html/2602.16986v1', 'version': '2602.16986v1', 'locator': 'SLA and sequence truncation', 'implementation': 'models.py:HSTUEncoder'},
    {'url': 'https://arxiv.org/html/2410.02604v1', 'version': '2410.02604v1', 'locator': 'decoupled attention and representation embeddings', 'implementation': 'models.py:DAREEncoder'},
    {'url': 'https://arxiv.org/html/2607.28070v1', 'version': '2607.28070v1', 'locator': 'Sections 3.1–3.4', 'implementation': 'ccformer.py:CCFormerEncoder'},
    {'url': 'https://arxiv.org/html/2602.23671v1', 'version': '2602.23671v1', 'locator': 'Section 4 and Appendix A', 'implementation': 'configs/benchmark.json'},
]


def acquire(request, directory, artifacts):
    url = request['url']; parts = urlsplit(url)
    if parts.scheme != 'https' or parts.hostname not in {'arxiv.org', 'export.arxiv.org', 'github.com', 'raw.githubusercontent.com', 'openreview.net'}:
        raise ContractError('only primary-source HTTPS hosts may be acquired')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    receipt = directory/'receipt.json'
    if receipt.exists():
        result = read_json(receipt)
        if result['request'] != request: raise ContractError('source request drift')
        if result.get('content_ref'): artifacts.verify(result['content_ref'])
        return result
    result = {'id': 'source-'+digest(request)[:24], 'request': request,
        'retrieved_at': datetime.now(timezone.utc).isoformat(), 'verification_status': 'not_retrieved',
        'content_ref': None, 'qualification': 'Content acquisition is not verification of a scientific claim or reproduction.'}
    try:
        with urlopen(Request(url, headers={'User-Agent': 'rec-harness-kuairand/1.0'}), timeout=60) as response:
            body = response.read(32*1024*1024 + 1)
            if len(body) > 32*1024*1024: raise ContractError('source exceeds acquisition size limit')
            content_type = response.headers.get('Content-Type', '')
        text = body.decode('utf-8', errors='replace')
        if 'text/html' in content_type and ('arxiv' not in text.lower() and parts.hostname == 'arxiv.org'):
            raise ContractError('source returned an unrecognized HTML body')
        (directory/'content').write_bytes(body)
        result.update(verification_status='content_retrieved', content_ref=artifacts.put(directory/'content'),
            content_type=content_type, locator_status='requested_locator_not_independently_verified')
    except (OSError, ValueError) as error:
        result['error'] = str(error)
    write_json(receipt, result)
    return result
