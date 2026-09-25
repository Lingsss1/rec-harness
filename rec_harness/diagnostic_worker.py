"""Runs only inside the CPU jail. Returned measurements remain untrusted."""
import importlib.util
import json
from pathlib import Path


def main():
    spec=importlib.util.spec_from_file_location('candidate_diagnostic','/inputs/tool/diagnostic.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result=module.measure(json.loads(Path('/inputs/request.json').read_text()))
    Path('/output/measurement.json').write_text(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
