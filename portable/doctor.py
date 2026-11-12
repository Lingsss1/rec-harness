import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys

root = Path(__file__).resolve().parents[1]
report = {'root': str(root), 'python': sys.version, 'platform': platform.platform(), 'packages': {}}
for name in ['torch', 'numpy', 'pandas', 'scipy', 'pydantic', 'omegaconf', 'jsonschema', 'agents', 'openai']:
    module = importlib.import_module(name)
    path = str(Path(module.__file__).resolve())
    if not path.startswith(str(root)+os.sep):
        raise RuntimeError('Dependency escaped portable folder: '+name+' '+path)
    report['packages'][name] = path
import torch
report.update(cuda_available=torch.cuda.is_available(), torch_version=torch.__version__,
              torch_cuda_version=torch.version.cuda, codex_binary=shutil.which('codex'))
if torch.cuda.is_available():
    report['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(json.dumps(report, ensure_ascii=False, indent=2))
