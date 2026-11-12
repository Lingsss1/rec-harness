"""Runs only inside the CPU jail. Returned measurements remain untrusted."""
import importlib.util
import json
import sys
from pathlib import Path


def main(input_root=Path('/inputs'),output_root=Path('/output')):
    tool=input_root/'tool'
    sys.path.insert(0,str(tool))
    spec=importlib.util.spec_from_file_location('candidate_diagnostic',tool/'diagnostic.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    result=module.measure(json.loads((input_root/'request.json').read_text()))
    (output_root/'measurement.json').write_text(json.dumps(result,allow_nan=False))


if __name__=='__main__':main()
