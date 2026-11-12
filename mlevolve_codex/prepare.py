"""Pin and fetch the upstream hybrid-memory embedding model on CPU."""
import json
import os
from pathlib import Path
import subprocess

from huggingface_hub import HfApi, snapshot_download

from mlevolve_codex.run import ROOT
from mle_star_codex.run import save, sha


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    manifest = ROOT / 'embedding-model.json'
    if manifest.exists():
        revision = json.loads(manifest.read_text())['revision']
    else:
        revision = HfApi().model_info('BAAI/bge-base-en-v1.5').sha
        save(manifest, {'model': 'BAAI/bge-base-en-v1.5', 'revision': revision, 'device': 'cpu'})
    directory = Path(snapshot_download('BAAI/bge-base-en-v1.5', revision=revision,
        local_dir=ROOT / 'embedding-model',
        allow_patterns=['*.json', '*.txt', 'model.safetensors', '1_Pooling/*']))
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(4)
    model = SentenceTransformer(str(directory), device='cpu')
    result = model.encode(['frozen NextItNet recommendation', 'causal convolution memory'])
    assert result.shape == (2, 768)
    save(ROOT / 'embedding-acceptance.json', {'shape': list(result.shape),
        'files': {str(p.relative_to(directory)): sha(p) for p in directory.rglob('*')
                  if p.is_file() and '.cache' not in p.parts}})
    environment = subprocess.check_output([os.sys.executable, '-m', 'pip', 'freeze'], text=True)
    (ROOT / 'environment.txt').write_text(environment)
    print('UPSTREAM_MEMORY_EMBEDDING_READY', flush=True)


if __name__ == '__main__':
    main()
