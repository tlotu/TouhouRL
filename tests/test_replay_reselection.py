"""Checks for screened forecast selection and first-hold validity."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.evaluate_replay_reselection import choose


class ReselectionTests(unittest.TestCase):
    def test_only_shortlisted_actions_are_forecast_and_nominal_wins_ties(self):
        env = SimpleNamespace(choices=[SimpleNamespace(route=SimpleNamespace(actions=(a,))) for a in range(9)])
        with patch('scripts.evaluate_replay_reselection.forecast',
                   return_value={'branches': [{'frames': 60}, {'frames': 60}]}) as mock:
            selected, scores = choose(env, None, None, 8, 60, [0, 2, 5, 8])
        self.assertEqual(selected, 8)
        self.assertEqual(set(scores), {0, 2, 5, 8})
        self.assertEqual([call.args[1] for call in mock.call_args_list], [0, 2, 5, 8])

    def test_candidate_failing_one_first_hold_is_excluded(self):
        def score(env, action, worker, horizon, policy):
            return {'branches': [{'frames': 60}, {'frames': 3 if action == 2 else 4}]}
        with patch('scripts.evaluate_replay_reselection.forecast', side_effect=score):
            selected, scores = choose(None, None, None, 0, 60, [0, 2])
        self.assertEqual(selected, 0)
        self.assertEqual(set(scores), {0})


if __name__ == '__main__':
    unittest.main()
