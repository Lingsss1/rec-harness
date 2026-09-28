"""Real worker subprocesses; process isolation itself is tested by launcher suites."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from rec_harness import diagnostic_worker


class DiagnosticWorkerTests(unittest.TestCase):
    def test_multifile_tool_imports_and_registered_module(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);inputs=root/'inputs';output=root/'output'
            tool=inputs/'tool';tool.mkdir(parents=True);output.mkdir()
            (tool/'helper.py').write_text('def spread(values): return max(values)-min(values)\n')
            (tool/'diagnostic.py').write_text('''from __future__ import annotations
from dataclasses import dataclass
from helper import spread
@dataclass
class Result:
    value: float
def measure(request):
    result=Result(spread([row['metrics']['quality'] for row in request['runs']]))
    return {'measurements': {'spread': result.value}, 'limitations': ['aggregate only']}
''')
            (inputs/'request.json').write_text(json.dumps({'runs':[{'metrics':{'quality':2}},{'metrics':{'quality':5}}]}))
            # Import this exact worker file, including when testing an installed wheel.
            program='''import importlib.util,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('worker',sys.argv[1])
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.main(Path(sys.argv[2]),Path(sys.argv[3]))
'''
            result=subprocess.run([sys.executable,'-I','-c',program,str(Path(diagnostic_worker.__file__).resolve()),str(inputs),str(output)],
                cwd=root,text=True,capture_output=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads((output/'measurement.json').read_text()),
                {'measurements':{'spread':3},'limitations':['aggregate only']})
