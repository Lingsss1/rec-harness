import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest

from rec_harness.core import ContractError, read_json
from rec_harness.replay import build_world
from rec_harness.replay_policy import run_policy, validate_action
from rec_harness.policy_sandbox import PolicySandbox

ROOT = Path(__file__).resolve().parents[1]


def world():
    return build_world([{'node_id': 'later', 'primary_parent_id': 'root', 'creation_seq': 1,
                        'dependencies': [], 'observation': {'score': 0.9, 'status': 'success'},
                        'protocol_hash': 'p', 'cost': 2}],
                       root_observation={'score': 0.1, 'status': 'success'}, initial_cost=1)


class PolicyProtocolTests(unittest.TestCase):
    def test_future_selection_and_unbounded_memory_rejected(self):
        action = {'parents': [], 'action_kind': 'continue', 'select': 'later', 'memory': None, 'rationale': ''}
        with self.assertRaisesRegex(ContractError, 'outside visible'):
            validate_action(action, {'observations': {'root': {}}})
        action.update(select=None, memory='x' * 65537)
        with self.assertRaisesRegex(ContractError, '64 KiB'):
            validate_action(action, {'observations': {}})

    def test_driver_only_supplies_prefix_and_selects_visible_model(self):
        class ProtocolFixture:  # Explicit test double; OS isolation tested separately.
            def decide(self, policy, visible, directory):
                observations = visible['view']['observations']
                if visible['step'] == 0:
                    assert 'later' not in json.dumps(visible)
                    assert '0.9' not in json.dumps(visible)
                    parents, selected = ['root'], 'root'
                else:
                    assert observations['later']['score'] == 0.9
                    parents, selected = [], 'later'
                return {'parents': parents, 'select': selected, 'action_kind': 'continue', 'memory': None, 'rationale': 'fixture'}
        with tempfile.TemporaryDirectory() as directory:
            result = run_policy(world(), 'unused', ProtocolFixture(), Path(directory) / 'run', 3)
            self.assertEqual(result['report']['selected_score'], 0.9)
            self.assertEqual(result['report']['historical_cost'], 3)
            self.assertFalse(result['online_utility_evidence'])


@unittest.skipUnless(os.environ.get('REC_HARNESS_OS_TESTS') == '1', 'explicit Linux/root OS boundary test')
class PolicyIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.template = cls.root / 'runtime'
        # Deployment prepares the minimal public runtime using system Python;
        # the controller may itself run inside the training virtual environment.
        subprocess.run(['/usr/bin/python3', '-m', 'rec_harness.policy_sandbox',
                        'prepare', str(cls.template)], cwd=ROOT, check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def sandbox(self, timeout=10):
        return PolicySandbox(self.template, self.root / 'jails', timeout=timeout)

    def test_host_proc_network_fork_and_environment_are_unavailable(self):
        canary = self.root / 'future.json'; canary.write_text('future outcome')
        policy = self.root / 'attack.py'
        policy.write_text('import json, os, socket\nfrom pathlib import Path\n'
                          'blocked = {}\n'
                          f'for p in [{str(canary)!r}, "/proc/1/root", "/root/.codex/auth.json"]:\n'
                          '    blocked[p] = not Path(p).exists()\n'
                          'for name, fn in [("network",socket.socket),("fork",os.fork)]:\n'
                          '    try: fn(); blocked[name] = False\n'
                          '    except PermissionError: blocked[name] = True\n'
                          'blocked["nonroot"] = os.getuid() != 0\n'
                          'blocked["environment"] = "REC_HARNESS_TEST_SECRET" not in os.environ\n'
                          'print(json.dumps(blocked))\n')
        os.environ['REC_HARNESS_TEST_SECRET'] = 'synthetic canary'
        try:
            result = self.sandbox().decide(policy, {}, self.root / 'attack-run')
        finally:
            del os.environ['REC_HARNESS_TEST_SECRET']
        self.assertTrue(all(result.values()), result)

    def test_same_world_policy_seed_replays_identically(self):
        reports = [run_policy(world(), ROOT / 'examples/toy/replay_policy.py', self.sandbox(),
                              self.root / name, 3, seed=42) for name in ['replay-a', 'replay-b']]
        self.assertEqual(reports[0], reports[1])
        self.assertEqual(reports[0]['report']['selected_score'], 0.9)
        first = read_json(self.root / 'replay-a/decision-0000/request.json')
        self.assertNotIn('later', json.dumps(first))

    def test_timeout_and_invalid_json_remain_failures(self):
        for name, source in [('timeout', 'while True: pass'), ('malformed', 'print("not json")')]:
            policy = self.root / f'{name}.py'; policy.write_text(source)
            with self.assertRaises((ContractError, ValueError)):
                run_policy(world(), policy, self.sandbox(timeout=0.2), self.root / name, 3)
            self.assertTrue((self.root / name / 'failure.json').exists())

    def test_filter_synchronizes_a_preexisting_thread(self):
        program = '''
import json, socket, threading
from rec_harness.policy_sandbox import restrict_syscalls
ready, release = threading.Event(), threading.Event()
result = {}
def existing_thread():
    ready.set(); release.wait()
    try:
        sock = socket.socket(); sock.close(); result['blocked'] = False
    except PermissionError:
        result['blocked'] = True
t = threading.Thread(target=existing_thread); t.start(); ready.wait()
restrict_syscalls(sync_threads=True)
release.set(); t.join(timeout=3)
assert result.get('blocked') is True, result
print(json.dumps(result))
'''
        process = subprocess.run(['/usr/bin/python3', '-c', program], cwd=ROOT,
                                 capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertTrue(json.loads(process.stdout)['blocked'])


if __name__ == '__main__':
    unittest.main()
