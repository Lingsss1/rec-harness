"""Prepare and run a NEW family/method cell on compatible prepared datasets."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=Path, required=True)
parser.add_argument('--datasets', required=True)
parser.add_argument('--model', choices=['gru4rec', 'nextitnet', 'bert4rec', 'hstu'], required=True)
parser.add_argument('--method', choices=['mlestar', 'self_evolverec'], required=True)
parser.add_argument('--gpus', required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--budget', type=int, default=43200)
parser.add_argument('--run', action='store_true')
args = parser.parse_args()
if os.environ.get('REC_HARNESS_DATASETS') != args.datasets:
    env = {**os.environ, 'REC_HARNESS_DATASETS': args.datasets}
    raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=env))
sys.path.insert(0, str(root))
from amazon_family_compare.bootstrap import bootstrap
bootstrap()
from amazon_family_compare.core import read, save
from amazon_family_compare.prepare import sha, DATASETS
gpus = [int(g) for g in args.gpus.split(',')]
if len(gpus) != 4 or len(set(gpus)) != 4 or min(gpus) < 0:
    parser.error('Use four distinct NVIDIA GPU indexes')
data = args.data.resolve()
output = args.output.resolve()
if output.exists():
    parser.error('Output must be a fresh directory; archived jobs are never silently resumed')
for name in DATASETS:
    for relative in ['trainval/'+name+'_train.txt', 'trainval/'+name+'_valid.txt', 'test/'+name+'_test.txt']:
        if not (data/relative).is_file():
            parser.error('Missing prepared split: '+str(data/relative))
    if args.method == 'self_evolverec' and not (data/'trainval'/(name+'_text_name_dict.json.gz')).is_file():
        parser.error('Self-EvolveRec needs metadata pickle: '+name+'_text_name_dict.json.gz')
template = root/'artifacts/model-family-comparison-20261029/runs'/(args.model+'-'+args.method)/'config.json'
config = read(template)
old = '/home/liuzhishan/lhr/rec-harness'
def relocate(value):
    if isinstance(value, str):
        return str(root)+value[len(old):] if value == old or value.startswith(old+'/') else value
    if isinstance(value, dict):
        return {k: relocate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v) for v in value]
    return value
config = relocate(config)
config.update(root=str(output), data_dir=str(data), gpus=gpus,
    cpu_cores=[c for g in gpus for c in range(16*g,16*g+16)],
    controller_python=str(root/'portable/bin/python-controller'), worker_python=str(root/'portable/bin/python-worker'),
    search_budget_gpu_seconds=args.budget, llm_slot_directory=str(output/'llm-slots'))
config['alignment']['datasets'] = DATASETS
config['replication'] = {'repetition':1,'fresh_baseline_and_llm_calls':True,
    'cross_run_response_or_memory_reuse':False,'search_gpu_seconds':args.budget}
save(output/'config.json', config)
save(output/'registration.json', {'source_root':str(root),'datasets':DATASETS,
    'baseline_sha256':sha(config['baseline']['source']), 'budget_gpu_seconds':args.budget,
    'files':{str(p.relative_to(data)):sha(p) for n in DATASETS for p in
             [data/'trainval'/(n+'_train.txt'),data/'trainval'/(n+'_valid.txt'),data/'test'/(n+'_test.txt')]}})
print(json.dumps({'prepared':str(output/'config.json'),'run':args.run}), flush=True)
if args.run:
    import time
    from amazon_family_compare.launch import inventory, idle
    from amazon_family_compare.controller import FamilyController
    while not idle(inventory(),gpus):
        save(output/'status.json', {'status':'waiting_for_idle_gpus','gpus':gpus})
        time.sleep(15)
    os.sched_setaffinity(0,set(config['cpu_cores']))
    module = __import__('amazon_family_compare.'+('mlestar' if args.method=='mlestar' else 'self_evolverec'),fromlist=['run'])
    module.Controller = FamilyController
    save(output/'status.json', {'status':'running','pid':os.getpid()})
    result = module.run(config)
    save(output/'status.json', {'status':'complete','test_hr10':result['test']['score']})
