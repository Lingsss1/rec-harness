"""Task-scoped agent transport; frozen training/scoring kernel is untouched."""
import importlib.util
from pathlib import Path
from rec_harness import bound_output
from rec_harness.agents import validate_schema
from rec_harness.jailed_agent import JailedCodexAgent
from rec_harness.core import file_hash,read_json

class RepairedSchemaAgent(JailedCodexAgent):
    def __init__(self,*args,repair_root,**kwargs):
        super().__init__(*args,**kwargs)
        root=Path(repair_root);record=read_json(root/'transport-lock.json')
        assert file_hash(Path(__file__))==record['adapter_sha256']
        source=root/'fixed-bound-output.py'
        assert file_hash(source)==record['bound_output_sha256']
        spec=importlib.util.spec_from_file_location('rec_harness.repaired_bound_output',source)
        self.fixed=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.fixed)
    def run(self,role,context,directory):
        host=self.fixed.bound_schema(role,context)
        original=bound_output.bound_schema
        try:
            bound_output.bound_schema=lambda role,context:self.fixed.service_schema(self.fixed.bound_schema(role,context))
            response=super().run(role,context,directory)
        finally:
            bound_output.bound_schema=original
        validate_schema(response,host)
        return response
