"""Validate reference-only startup and task-owned contracts; does not launch work."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def validate_launch(payload, root=ROOT):
    schema = read(Path(root) / 'agent/launch.schema.json')
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)


def check_fixed_context(root, terms):
    paths = sorted((Path(root) / 'agent').glob('*'))
    for path in paths:
        if not path.is_file():
            continue
        content = path.read_text()
        for term in terms:
            pattern = r'(?<![A-Za-z0-9_])' + re.escape(term) + r'(?![A-Za-z0-9_])'
            if re.search(pattern, content, flags=re.IGNORECASE):
                raise ValueError(f'Task-specific term in fixed agent artifact {path.name}: {term}')


def validate(root=ROOT):
    root = Path(root).resolve()
    for path in root.rglob('*.json'):
        if 'source_cache' not in path.relative_to(root).parts and 'workspace' not in path.relative_to(root).parts:
            read(path)
    launch = read(root / 'launch.example.json')
    validate_launch(launch, root)
    catalog = read(root / 'catalog.json')['entries']
    for ref in launch.values():
        if ref not in catalog:
            raise ValueError('Unresolved startup reference: ' + ref)
        resolved = (root / catalog[ref]['path']).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError('Design catalog path escapes its root')
        if catalog[ref]['kind'] != 'workspace' and not resolved.is_file():
            raise ValueError('Missing contract: ' + str(resolved))
    task_path = root / catalog[launch['task_ref']]['path']
    task = read(task_path)
    check_fixed_context(root, task['fixed_agent_boundary_audit']['forbidden_terms'])
    for path in (root / 'agent').glob('*.schema.json'):
        Draft202012Validator.check_schema(read(path))
    validator_path = (task_path.parent / task['design_validator_ref']).resolve()
    if not validator_path.is_relative_to(task_path.parent):
        raise ValueError('Task validator escapes its own package')
    spec = importlib.util.spec_from_file_location('task_design_validation', validator_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task_result = module.validate_task(task_path.parent, task)
    agent_artifacts = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((root / 'agent').glob('*')) if path.is_file()}
    return {'status': 'design_validation_passed', 'fixed_agent_artifacts_sha256': agent_artifacts, 'startup_fields': sorted(launch), 'task_checks': task_result, 'runnable_campaign': False, 'remaining_execution_work': ['data acquisition and schema/time audit', 'task adapter and enforced tool integration', 'causal feature and scorer acceptance', 'complete-unit cost calibration', 'protocol and allocation freeze']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate(args.root)
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
