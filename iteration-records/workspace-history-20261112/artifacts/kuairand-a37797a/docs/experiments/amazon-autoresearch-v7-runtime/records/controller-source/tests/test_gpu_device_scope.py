import unittest
from rec_harness.core import ContractError
from rec_harness.isolation import gpu_device_paths, prepare


class GPUDeviceScopeTests(unittest.TestCase):
    def test_explicit_devices_only_no_host_wildcard(self):
        self.assertEqual(gpu_device_paths([3, 0]),
                         ['/dev/nvidiactl', '/dev/nvidia-uvm', '/dev/nvidia0', '/dev/nvidia3'])

    def test_ambiguous_or_cpu_only_device_requests_fail_before_preparation(self):
        for ids in [None, [], [True], [-1], [0, 0], ['0']]:
            with self.assertRaises(ContractError):
                gpu_device_paths(ids)
        with self.assertRaisesRegex(ContractError, 'requires include_gpu'):
            prepare('/unused', [], [], gpu_devices=[0])
