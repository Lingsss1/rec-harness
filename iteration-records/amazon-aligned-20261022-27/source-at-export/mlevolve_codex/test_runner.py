"""Exercise upstream search and the 44-slot boundary without LLM or GPU cost."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from mlevolve_codex.run import (NativeController, UPSTREAM, WORKSPACE, build_agent,
                               search, transport_schema)

sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(WORKSPACE / 'staged-autoresearch'))


class Memory:
    def __init__(self, **kwargs):
        self.records = []
    def save_node(self, node, parent):
        self.records.append(node.id)
        return True
    def retrieve_similar_records(self, *args, **kwargs):
        return []


class Tests(unittest.TestCase):
    def test_native_schema_preserves_module_plan_and_optional_fields(self):
        from agents.planner.base_planner import PLANNING_JSON_SCHEMA
        from staged_research.codex import service_schema, remove_optional_nulls
        bounded = transport_schema(PLANNING_JSON_SCHEMA)
        strict = service_schema(bounded)
        self.assertFalse(strict['properties']['plan']['additionalProperties'])
        value = {'reason': 'synthetic', 'module': ['model_design'], 'plan': {
            'model_design': 'synthetic convolution change',
            'training_evaluation': None, 'data_processing_and_feature_engineering': None}}
        clean = remove_optional_nulls(value, bounded)
        self.assertEqual(clean['plan'], {'model_design': 'synthetic convolution change'})
        import jsonschema
        jsonschema.validate(clean, PLANNING_JSON_SCHEMA)

    def test_upstream_graph_finishes_exactly_44_with_memory_and_one_failed_trial(self):
        from engine.search_node import SearchNode
        from agents import draft_agent, improve_agent, debug_agent, evolution_agent, fusion_agent, aggregation_agent, code_review_agent
        from agents.triggers import register_node
        from engine.executor import ExecutionResult
        with tempfile.TemporaryDirectory() as tmp:
            c = NativeController({'run_dir': tmp, 'model': 'synthetic', 'effort': 'high',
                'repo': str(WORKSPACE / 'artifacts/RecHarness-6lyc'), 'gpus': [0, 1, 2, 3]})
            c.event = lambda *args, **kwargs: None
            c.query_native = lambda *args, **kwargs: {'analysis': 'synthetic observation', 'code_summary': 'synthetic model'}
            with patch('agents.memory.global_memory.GlobalMemoryLayer', Memory):
                agent = build_agent(c)
            serial = [0]
            def generate(a, parent_node=None, **kwargs):
                parent = parent_node or a.virtual_root
                parent.add_expected_child_count()
                node = SearchNode(code='SYNTHETIC', plan='synthetic', parent=parent,
                    stage='draft' if parent.stage == 'root' else 'improve', local_best_node=parent.local_best_node or parent)
                register_node(a, node, 'synthetic', parent_node=parent, new_branch=parent.stage == 'root')
                return node
            def worker(controller, gpu, code, node_id, reset=True):
                serial[0] += 1
                n = serial[0]
                receipt = {'status': 'crash' if n == 7 else 'success', 'allocated_gpu_seconds': 0,
                    'metrics': {'HR@10': 0.3 + n / 10000}}
                return ExecutionResult([json.dumps(receipt)], 0, None)
            patches = [patch.object(module, 'run', generate) for module in (
                draft_agent, improve_agent, debug_agent, evolution_agent, fusion_agent, aggregation_agent)]
            for p in patches:
                p.start()
            try:
                with patch.object(code_review_agent, 'run', lambda a, n: n.code), patch('mlevolve_codex.run.execute', worker):
                    completed = search(agent, c)
            finally:
                for p in patches:
                    p.stop()
            self.assertEqual(completed, 44)
            self.assertEqual(serial[0], 44)
            self.assertEqual(len(agent.journal.nodes), 45)
            self.assertEqual(len(agent.global_memory.records), 43)
            self.assertEqual(sum(n.is_buggy is True for n in agent.journal.nodes), 1)
            self.assertGreaterEqual(len(agent.branch_all_nodes), 4)
            self.assertGreater(agent.best_metric, 0.3)
            self.assertEqual(json.loads((Path(tmp) / 'progress.json').read_text())['completed_candidates'], 44)


if __name__ == '__main__':
    unittest.main()
