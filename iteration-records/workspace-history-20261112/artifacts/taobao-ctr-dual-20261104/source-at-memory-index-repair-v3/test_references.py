from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from references import observed_trial_refs
import controller
from common import write


class ReferenceRecoveryTests(unittest.TestCase):
    def test_descriptive_reference_binds_to_real_trial(self):
        refs=['run://r-001 context.history baseline-din: validation AUC observed',
              'run://r-001 context.history: baseline-static, baseline-pooled, baseline-din; validation only']
        self.assertEqual(observed_trial_refs(refs,{'baseline-din','baseline-static','baseline-pooled'}),
                         ['baseline-din','baseline-pooled','baseline-static'])

    def test_exact_existing_reference_is_preserved(self):
        self.assertEqual(observed_trial_refs(['baseline-din'],{'baseline-din'}),['baseline-din'])

    def test_fabricated_suffix_or_prefix_does_not_match(self):
        for ref in ['baseline-din-invented','fakebaseline-din','baseline-din_v2','unknown-trial']:
            with self.subTest(ref=ref),self.assertRaises(ValueError):observed_trial_refs([ref],{'baseline-din'})

    def test_empty_references_are_rejected(self):
        with self.assertRaises(ValueError):observed_trial_refs([],{'baseline-din'})

    def test_runtime_error_is_returned_to_the_researcher(self):
        state={'trials':[],'last_runtime_error':{'error':'Cite an observed ID'}}
        contract=controller.phase_contract('propose','auto',state,{'identity':'p'},1000)
        self.assertEqual(contract['context']['previous_runtime_error'],state['last_runtime_error'])

    def test_recovery_cannot_fall_through_to_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);write(root/'auto-validation-only.json',{'enabled':True})
            with patch.object(controller,'finish_validation',return_value='validation-only') as finish:
                result=controller.final_assessment(root,root/'auto',{}, {},None,[],0)
            self.assertEqual(result,'validation-only');finish.assert_called_once()


if __name__=='__main__':unittest.main()
