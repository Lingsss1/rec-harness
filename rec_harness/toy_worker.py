"""Untrusted model execution worker: no validation targets or scoring authority."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();request=json.loads(Path(args.request).read_text())
    sys.path.insert(0,'/inputs/model')
    spec=importlib.util.spec_from_file_location('candidate','/inputs/model/model.py')
    model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
    weights=model.train(request['training'],request['seed'])
    predictions=[model.predict(weights,row) for row in request['queries']]
    Path(args.output).write_text(json.dumps({'weights':weights,'predictions':predictions},allow_nan=False))


if __name__=='__main__':main()
