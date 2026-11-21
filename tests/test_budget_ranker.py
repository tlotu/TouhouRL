"""Check ranked imitation metrics and finite features without visible hazards."""
import unittest

import numpy as np
import torch

from agent.budget_ranker import DirectionRanker, direction_features
from agent.direction_buffer import DirectionBuffer
from agent.forward_model import Scene
from scripts.train_budget_ranker import metrics
from scripts.train_budget_groups import different_scores


class RankerTests(unittest.TestCase):
    def test_tied_scores_and_singleton_are_evaluated_separately(self):
        offered = np.array([[True, True, True], [True, False, False]])
        labels = np.array([2, 0])
        scores = np.array([[4, 60, 60], [-1, -1, -1]])
        # The actual action space has nine directions.
        offered = np.pad(offered, ((0, 0), (0, 6)))
        scores = np.pad(scores, ((0, 0), (0, 6)), constant_values=-1)
        result = metrics(np.array([[1, 0], [1, 0]]), offered, labels, scores)
        self.assertEqual((result['top1'], result['top2'], result['teacher_top1']), (1, 2, 0))
        self.assertEqual((result['unique_verified'], result['unique_top2'], result['events']), (1, 1, 1))

    def test_scorer_accepts_unbatched_and_batched_current_features(self):
        features = direction_features(Scene((20., 20.)), {'enemies': np.zeros((2, 8))},
                                      (DirectionBuffer(),), np.zeros(9))
        self.assertTrue(np.isfinite(features).all())
        x = torch.from_numpy(features)
        model = DirectionRanker(features.shape[-1])
        torch.testing.assert_close(model(x), model(x[None])[0])

    def test_long_preferences_require_two_distinct_measured_scores(self):
        for scores, expected in [([-1, -1], False), ([-1, 60], False),
                                 ([60, 60], False), ([32, 60], True)]:
            row = (None, None, None, None, np.array(scores), None)
            self.assertEqual(different_scores(row), expected)
