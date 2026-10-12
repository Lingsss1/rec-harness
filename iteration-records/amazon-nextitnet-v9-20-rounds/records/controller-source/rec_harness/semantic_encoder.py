"""Optional, offline Qwen3 embeddings for already-authorized public memory.

No model download, remote code, GPU, or Codex call occurs here. Campaigns pin
the local model files and runtime; historical replay uses recorded scores.
"""
import importlib.metadata
import copy
import time
from pathlib import Path

from .core import ContractError, digest, file_hash
from .semantic import documents, receipt, validate


def _runtime():
    try:
        return {name: importlib.metadata.version(name)
                for name in ('torch', 'transformers', 'tokenizers', 'safetensors')}
    except importlib.metadata.PackageNotFoundError as error:
        raise ContractError('semantic retrieval requires the local semantic optional dependencies') from error


def _files(directory):
    root = Path(directory)
    if not root.is_dir():
        raise ContractError('semantic model must be an existing local directory')
    files = {str(path.relative_to(root)): file_hash(path)
             for path in sorted(root.rglob('*')) if path.is_file()}
    if not {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors'} <= files.keys():
        raise ContractError('semantic model requires local config, tokenizer and single safetensors weights')
    return files


def freeze(spec):
    if not isinstance(spec, dict) or set(spec) - {
            'model_path', 'query', 'instruction', 'max_length', 'batch_size', 'threads', 'expected_hash'}:
        raise ContractError('invalid semantic encoder configuration')
    if not isinstance(spec.get('model_path'), str) or not spec['model_path']:
        raise ContractError('semantic model_path required')
    if not isinstance(spec.get('query'), str) or not spec['query'].strip():
        raise ContractError('semantic retrieval requires an explicit research query')
    settings = {'max_length': spec.get('max_length', 2048),
                'batch_size': spec.get('batch_size', 4), 'threads': spec.get('threads', 2),
                'instruction': spec.get('instruction', 'Retrieve research evidence relevant to the query.'),
                'pooling': 'last_nonpadding_token', 'normalization': 'l2',
                'device': 'cpu', 'dtype': 'float32', 'attention': 'eager'}
    for key, maximum in [('max_length', 8192), ('batch_size', 32), ('threads', 16)]:
        if type(settings[key]) is not int or not 1 <= settings[key] <= maximum:
            raise ContractError('invalid semantic encoder ' + key)
    if not isinstance(settings['instruction'], str) or not settings['instruction'].strip():
        raise ContractError('semantic instruction required')
    root = Path(spec['model_path']).resolve()
    from .core import read_json
    if not (root / 'config.json').is_file() or read_json(root / 'config.json').get('model_type') != 'qwen3':
        raise ContractError('semantic encoder currently supports local Qwen3 embedding models')
    files = _files(root)
    result = {'model_path': str(root), 'query': spec['query'], 'files': files,
            'model_hash': digest(files), 'runtime': _runtime(), 'settings': settings,
            'implementation_hash': file_hash(Path(__file__))}
    if 'expected_hash' in spec and spec['expected_hash'] != digest(result):
        raise ContractError('semantic encoder differs from frozen outer configuration')
    return result


def pin_options(config):
    """Freeze an outer project's/suite's identity before creating child campaigns."""
    if 'semantic_retrieval' in config:
        raise ContractError('semantic_retrieval is controller-owned; supply semantic_encoder')
    if 'semantic_encoder' in config:
        frozen = freeze(config['semantic_encoder'])
        config['semantic_encoder'] = {**config['semantic_encoder'],
            'model_path': frozen['model_path'], 'expected_hash': digest(frozen)}


def verify_options(original, frozen):
    expected = freeze(original['semantic_encoder']) if 'semantic_encoder' in original else None
    if frozen.get('semantic_retrieval') != expected:
        raise ContractError('paired semantic encoder changed')


def freeze_options(config):
    if 'semantic_retrieval' in config:
        raise ContractError('semantic_retrieval is controller-owned; supply semantic_encoder')
    if 'semantic_encoder' in config:
        config['semantic_retrieval'] = freeze(config.pop('semantic_encoder'))


class LocalEncoder:
    def __init__(self, frozen, cache_dir=None):
        self.frozen = frozen
        self.model = self.tokenizer = None
        self.cache = {}
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    def verify(self):
        value = self.frozen
        if (value['implementation_hash'] != file_hash(Path(__file__)) or value['runtime'] != _runtime()
                or digest(value['files']) != value['model_hash'] or _files(value['model_path']) != value['files']):
            raise ContractError('frozen semantic encoder files or runtime changed')

    def __call__(self, nodes, query):
        from .core import public_only
        public_only(nodes)
        if not isinstance(query, str) or not query.strip():
            raise ContractError('semantic query required')
        self.verify()
        key = digest({'nodes': nodes, 'query': query, 'encoder': self.frozen})
        identity = {name: self.frozen[name] for name in
                    ('model_hash', 'implementation_hash', 'runtime', 'settings')}
        path = self.cache_dir / (key + '.json') if self.cache_dir is not None else None
        if path is not None and path.exists():
            from .core import read_json
            result = read_json(path)
            validate(result, nodes, query)
            if result['encoder'] != identity or 'execution' not in result:
                raise ContractError('cached semantic encoder identity changed')
            return result
        if key in self.cache:
            return copy.deepcopy(self.cache[key])
        start = time.monotonic()
        cpu_start = time.process_time()
        import torch
        from transformers import AutoModel, AutoTokenizer
        settings = self.frozen['settings']
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(settings['threads'])
        try:
            if self.model is None:
                model_path = self.frozen['model_path']
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
                    trust_remote_code=False, padding_side='left')
                self.model = AutoModel.from_pretrained(model_path, local_files_only=True,
                    trust_remote_code=False, use_safetensors=True, torch_dtype=torch.float32,
                    attn_implementation='eager').to('cpu').eval()
            texts = documents(nodes)
            ids = list(texts)
            inputs = ['Instruct: ' + settings['instruction'] + '\nQuery:' + query, *texts.values()]
            vectors = []
            truncated = []
            with torch.inference_mode():
                for offset in range(0, len(inputs), settings['batch_size']):
                    batch = inputs[offset:offset + settings['batch_size']]
                    lengths = [len(self.tokenizer.encode(text, truncation=False)) for text in batch]
                    truncated.extend(offset + i for i, length in enumerate(lengths) if length > settings['max_length'])
                    encoded = self.tokenizer(batch, padding=True, truncation=True,
                        max_length=settings['max_length'], return_tensors='pt')
                    hidden = self.model(**encoded).last_hidden_state[:, -1]
                    vectors.append(torch.nn.functional.normalize(hidden, p=2, dim=1))
                vectors = torch.cat(vectors)
                scores = (vectors[1:] @ vectors[0]).tolist()
            result = receipt(nodes, query, identity, dict(zip(ids, scores)),
                execution={'elapsed_seconds': time.monotonic() - start,
                           'cpu_seconds': time.process_time() - cpu_start,
                           'document_count': len(ids), 'query_truncated': 0 in truncated,
                           'truncated_documents': [ids[i - 1] for i in truncated if i > 0]})
            if path is not None:
                from .core import write_json
                write_json(path, result)
            self.cache = {key: copy.deepcopy(result)}
            return result
        finally:
            torch.set_num_threads(previous_threads)


def configured(controller, role):
    """Construct lazily; old campaigns never import the optional ML runtime."""
    frozen = controller.config.get('semantic_retrieval')
    if not frozen:
        return '', None
    if not hasattr(controller, '_semantic_encoder'):
        controller._semantic_encoder = LocalEncoder(frozen, controller.root / 'semantic-retrieval')
    return frozen['query'] + '\nResearch role: ' + role, controller._semantic_encoder
