"""The budget starts on verified data, and rebuilding preserves its deadline."""
import unittest

from budget import budget_state


class BudgetClock(unittest.TestCase):
    policy={'duration_seconds':172800,'start_event':'dataset_download_and_checksum_verification_completed'}

    def test_download_wait_does_not_start_research_clock(self):
        state=budget_state(self.policy,{'created_unix':100},{'phase':'downloading_official_lfs'}, {})
        self.assertIsNone(state['start_unix'])
        self.assertIsNone(state['deadline_unix'])
        self.assertEqual(state['clock_phase'],'waiting_for_verified_dataset')

    def test_verified_timestamp_replaces_old_clock_and_survives_rebuilds(self):
        old={'start_unix':100,'deadline_unix':86500,'single_experiment_limit_seconds':900}
        state=budget_state(self.policy,{'created_unix':100},{'phase':'downloaded_and_verified'}, {'verified_unix':400},old)
        self.assertEqual(state['start_unix'],400)
        self.assertEqual(state['deadline_unix'],173200)
        self.assertEqual(state['seconds'],48*3600)
        self.assertEqual(state['pre_budget']['wall_seconds'],300)
        self.assertEqual(state['single_experiment_limit_seconds'],900)
        self.assertEqual(state,budget_state(self.policy,{'created_unix':100},{'phase':'downloaded_and_verified'}, {'verified_unix':400},state))

    def test_verification_without_a_recorded_time_cannot_reset_clock_to_now(self):
        with self.assertRaises(ValueError):
            budget_state(self.policy,{'created_unix':100},{'phase':'downloaded_and_verified'}, {})

    def test_rechecking_or_downloading_again_does_not_pause_or_extend_started_clock(self):
        state=budget_state(self.policy,{'created_unix':100},{'phase':'downloaded_and_verified'}, {'verified_unix':400})
        for phase in ['downloaded_and_verified','listing_official_taobao_release']:
            repeated=budget_state(self.policy,{'created_unix':100},{'phase':phase},{'verified_unix':9000},state)
            self.assertEqual(repeated['start_unix'],400)
            self.assertEqual(repeated['deadline_unix'],173200)
            self.assertEqual(repeated['clock_phase'],'running')

    def test_explicit_cached_data_rerun_has_a_new_twelve_hour_anchor(self):
        policy={'duration_seconds':43200,'start_event':'new_campaign_started_with_verified_dataset',
                'campaign_start_unix':1000}
        state=budget_state(policy,{'created_unix':990},{'phase':'downloaded_and_verified'}, {'verified_unix':400})
        self.assertEqual(state['start_unix'],1000)
        self.assertEqual(state['deadline_unix'],44200)
        self.assertEqual(state['seconds'],43200)
        changed={**policy,'campaign_start_unix':9000}
        again=budget_state(changed,{'created_unix':990},{'phase':'downloaded_and_verified'}, {'verified_unix':400},state)
        self.assertEqual(again['start_unix'],1000)
        self.assertEqual(again['deadline_unix'],44200)

    def test_cached_data_rerun_requires_readiness_and_explicit_valid_start(self):
        policy={'duration_seconds':43200,'start_event':'new_campaign_started_with_verified_dataset',
                'campaign_start_unix':1000}
        waiting=budget_state(policy,{'created_unix':990},{'phase':'downloading'}, {'verified_unix':400})
        self.assertIsNone(waiting['start_unix'])
        for invalid in [None,False,200]:
            with self.assertRaises(ValueError):
                budget_state({**policy,'campaign_start_unix':invalid},{'created_unix':990},
                             {'phase':'downloaded_and_verified'}, {'verified_unix':400})


if __name__=='__main__':unittest.main()
