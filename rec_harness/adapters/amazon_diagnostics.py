"""Diagnostic input consists only of already audited development aggregates."""
from types import SimpleNamespace
from rec_harness.core import ContractError,digest
from rec_harness.diagnostics import measure
from rec_harness.measurement_gaps import MeasurementDeferred,triage
from rec_harness.runtime import node
from rec_harness.store import Store


class DiagnosticLedger:
    def __init__(self,graph,costs):self.graph,self.costs=graph,costs
    def __getattr__(self,name):return getattr(self.graph,name)
    def reserve(self,*args):return self.costs.reserve(*args)
    def settle(self,*args):return self.costs.settle(*args)


def diagnostic_context(controller,base,rows):
    config=controller.config
    if not config.get('diagnostic_tool') and not config.get('measurement_requirements') and not config.get('measurement_discovery'):return {}
    task=config['task_id'];protocol=config['protocol_hash'];known={n['id']:n for n in controller.store.snapshot()['nodes']}
    nodes=[];runs=[]
    if base not in known:nodes.append(node(base,'ModelVersion',{'scope':'registered study incumbent'},task,protocol))
    for row in rows:
        run_id='dev-aggregate-'+digest([base,protocol,row['dataset'],row['seed']])[:32]
        run={**row,'run_id':run_id,'valid':True};runs.append(run)
        observed=node(run_id,'Experiment',{
            **run,'model_version':base,'scope':'audited development aggregate, no raw interactions or checkpoint'},task,protocol)
        if run_id not in known:nodes.append(observed)
        elif known[run_id]!=observed:raise ContractError('previously observed development aggregate changed')
    if nodes:controller.store.patch({'base_version':controller.store.version,'nodes':nodes,'edges':[]})
    costs=Store(controller.root/'diagnostic-budget')
    try:
        costs.configure_budget(config.get('diagnostic_budget',0));costs.recover()
        view=SimpleNamespace(root=controller.root,config=config,task=task,protocol_hash=protocol,
            store=DiagnosticLedger(controller.store,costs),artifacts=controller.artifacts)
        state={'incumbent':base,'baseline':{'runs':runs}}
        diagnostic=measure(view,state)
        try:gaps=triage(view,state,diagnostic);failure=None
        except MeasurementDeferred as error:gaps=[];failure=str(error)
        return {'diagnostic':diagnostic,'measurement_gaps':gaps,'measurement_failure':failure,'measurement_baseline':{'runs':runs}}
    finally:costs.close()
