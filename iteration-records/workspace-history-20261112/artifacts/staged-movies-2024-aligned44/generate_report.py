"""Report-only recovery; registered code and all experiment receipts stay intact."""
import hashlib
import json
from pathlib import Path
import sys
import types

prefix = Path(__file__).resolve().parent
workspace = prefix.parents[1]
sys.path.insert(0, str(workspace / 'staged-autoresearch'))
from staged_research.storage import read, write
source_path = workspace / 'staged-autoresearch/staged_research/report.py'
source = source_path.read_text()
assert source.count('for rep in range(1, 6):') == 1
corrected = source.replace('for rep in range(1, 6):', 'for rep in range(1, registration["config"]["protocol"]["replicates"] + 1):')
corrected = corrected.replace('Five repetitions give imprecise exploratory bootstrap intervals.', 'One replicate per arm: descriptive results only; no statistical significance or bootstrap interval.')
corrected = corrected.replace('item["complete_pairs"], "unavailable"', 'item["complete_pairs"], "unavailable"')
namespace = types.ModuleType('staged_research.report_postprocess')
namespace.__package__ = 'staged_research'
exec(compile(corrected, str(source_path) + ':report-only-correction', 'exec'), namespace.__dict__)
registration = read(prefix / 'study/registration.json')
summary = namespace.report(prefix / 'study', registration, prefix / 'report')
# The display denominator belongs to this one-repeat comparison.
f = prefix / 'report/report.md'
f.write_text(f.read_text().replace(' / 5 |', ' / 1 |'))
write(prefix / 'report/recovery.json', {'reason': 'Original report used fixed five-repeat comparison loop after run budget was changed to one repeat', 'training_calls_added': 0, 'agent_calls_added': 0, 'test_calls_added': 0, 'registered_code_modified': False, 'original_report_sha256': hashlib.sha256(source.encode()).hexdigest(), 'registered_id': registration['registration_id']})
print(json.dumps({'completed': summary['completed_trajectories'], 'expected': summary['expected_trajectories'], 'report': str(prefix / 'report/summary.json')}))
