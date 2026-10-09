import os
import unittest

from rec_harness.core import ContractError
from rec_harness.gpu_descriptors import select_descriptors


class DescriptorTests(unittest.TestCase):
    def fixture(self):
        return {3: {'kind': 'device', 'target': '/dev/nvidia0'},
                4: {'kind': 'device', 'target': '/dev/nvidia1'},
                5: {'kind': 'socket', 'target': 'socket:[100]'},
                6: {'kind': 'pipe', 'target': 'pipe:[200]', 'inode': 200, 'filesystem_device': 1, 'access': os.O_RDONLY},
                7: {'kind': 'pipe', 'target': 'pipe:[200]', 'inode': 200, 'filesystem_device': 1, 'access': os.O_WRONLY},
                8: {'kind': 'other', 'target': '/host/directory'},
                9: {'kind': 'eventfd', 'target': 'anon_inode:[eventfd]'}}

    def test_only_selected_device_local_pipes_and_eventfd_retained(self):
        result = select_descriptors({}, self.fixture(), [0])
        self.assertEqual(result['retain'], [3, 6, 7, 9])
        self.assertEqual(result['close'], [4, 5, 8])
        self.assertEqual(result['pipe_pairs'], 1)

    def test_inherited_handles_and_incomplete_pipe_rejected(self):
        with self.assertRaisesRegex(ContractError, 'inherited'):
            select_descriptors({3: {}}, self.fixture(), [0])
        missing = self.fixture(); del missing[7]
        with self.assertRaisesRegex(ContractError, 'endpoint'):
            select_descriptors({}, missing, [0])
        duplicate = self.fixture(); duplicate[10] = duplicate[7]
        with self.assertRaisesRegex(ContractError, 'endpoint'):
            select_descriptors({}, duplicate, [0])
