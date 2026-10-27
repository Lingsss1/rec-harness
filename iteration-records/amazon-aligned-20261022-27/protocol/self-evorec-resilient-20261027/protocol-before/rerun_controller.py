"""Single-model selection and constrained transport for the fresh reruns."""
from pathlib import Path
import subprocess

from .core import Controller, read, save
from .prepare import WORKSPACE, sha
from .protocol_fixes import SINGLE_MODEL_CONTRACT


class SingleSelectionMixin:
    def llm(self, instruction, output_schema=None, search=False, key=None):
        if self.c.get('single_model_search'):
            instruction = SINGLE_MODEL_CONTRACT + '\n' + instruction + '\n' + SINGLE_MODEL_CONTRACT
        return super().llm(instruction, output_schema, search, key)

    def finish(self):
        records = [read(path) for path in self.root.glob('trials/*/receipt.json')]
        records = [r for r in records if r['status'] == 'success']
        records.sort(key=lambda r: (-r['score'], r['id']))
        reviewed = []
        for record in records:
            trial = Path(record['output'])
            audit_path = trial / 'single-model-audit.json'
            if not audit_path.exists():
                command = [self.c['worker_python'], '-m', 'amazon_aligned.single_model',
                           '--audit', str(trial), '--data', self.c['data_dir'], '--output', str(audit_path)]
                with (trial / 'single-model-audit.log').open('w') as log:
                    subprocess.run(command, cwd=WORKSPACE, env=self.environment(''),
                                   stdout=log, stderr=subprocess.STDOUT, timeout=300, check=True)
            audit = read(audit_path)
            if audit['predictor_sha256'] != sha(trial / 'working/predict.py'):
                raise RuntimeError('Candidate changed since single-model audit')
            if any(sha(Path(path)) != expected for path, expected in audit.get('checkpoint_sha256', {}).items()):
                raise RuntimeError('Checkpoint changed since single-model audit')
            reviewed.append({'id': record['id'], 'validation_hr10': record['score'],
                             'eligible': audit['eligible'], 'reason': audit['reason'],
                             'audit': str(audit_path)})
            if audit['eligible'] is True:
                self.best = record
                save(self.root / 'single-model-selection.json', {
                    'selection_metric': 'highest host validation HR@10 among verified single models',
                    'reviewed': reviewed, 'selected': self.brief(record),
                    'policy': 'unrecognized predictors are ineligible; held-out metrics never used to select'})
                return super().finish()
        raise RuntimeError('No verified single-model candidate; do not evaluate an ensemble')


class FreshSelfController(SingleSelectionMixin, Controller):
    pass
