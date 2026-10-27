"""Real upstream draft/review and hybrid-memory acceptance, no model training."""
from pathlib import Path
import sys

from mlevolve_codex.run import ROOT, UPSTREAM, WORKSPACE, NativeController, build_agent
from mle_star_codex.run import save, source_parts


def main():
    root = WORKSPACE / 'artifacts/mlevolve-native-smoke'
    root.mkdir(parents=True, exist_ok=False)
    (root / 'embedding-model').symlink_to(ROOT / 'embedding-model', target_is_directory=True)
    sys.path.insert(0, str(UPSTREAM))
    sys.path.insert(0, str(WORKSPACE / 'staged-autoresearch'))
    c = NativeController({'run_dir': str(root), 'repo': str(WORKSPACE / 'artifacts/RecHarness-6lyc'),
        'model': 'gpt-6.1-sol', 'effort': 'high'})
    agent = build_agent(c)
    node = agent.step(node=None, exec_callback=None, execute_immediately=False)
    source_parts(node.code)
    assert node.stage == 'draft' and node.pending_execution
    assert agent.global_memory is not None
    (root / 'draft.py').write_text(node.code)
    from utils.metric import MetricValue
    # Synthetic value is confined to interface acceptance memory; never formal evidence.
    node.metric = MetricValue(0.1, maximize=True)
    node.is_buggy, node.is_valid = False, True
    node.code_summary = 'Interface acceptance only: causal convolution NextItNet draft.'
    node.analysis = 'Synthetic memory persistence probe, no training or measured score.'
    assert agent.global_memory.save_node(node, node.parent)
    assert agent.global_memory.retriever.vector_index.ntotal == 1
    retrieved = agent.global_memory.retrieve_similar_records('causal convolution NextItNet', top_k=1)
    assert retrieved and retrieved[0][0].record_id == 'node_' + node.id
    from agents.planner.base_planner import PLANNING_JSON_SCHEMA
    plan = c.query_native('Transport acceptance only: choose model_design and propose changing residual branch scaling. '
        'Return reason, module=[model_design], plan={model_design: a concrete description}.', PLANNING_JSON_SCHEMA)
    assert plan['module'] == ['model_design'] and set(plan['plan']) == {'model_design'}
    save(root / 'acceptance.json', {'upstream_draft_review': True, 'source_contract': True,
        'native_planner_schema': True, 'hybrid_memory_initialized': True,
        'hybrid_memory_save_retrieve': True, 'training_calls': 0,
        'effort': 'high', 'memory_dimension': agent.global_memory.embedding_model.dimension})
    print('MLEVOLVE_NATIVE_DRAFT_REVIEW_MEMORY_SCHEMA_PASSED', flush=True)


if __name__ == '__main__':
    main()
