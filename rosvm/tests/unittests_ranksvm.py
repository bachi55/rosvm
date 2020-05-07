####
#
# The MIT License (MIT)
#
# Copyright 2018, 2019 Eric Bach <eric.bach@aalto.fi>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
####

import unittest
import numpy as np
import itertools as it

from ranksvm.rank_svm_cls import KernelRankSVC, get_pairwise_labels
from ranksvm.pair_utils import get_pairs_single_dataset


class TestPredictionSpeedup(unittest.TestCase):
    def test_compare_with_slow_version(self):
        wtX1 = np.random.RandomState(919).rand(34, 1)

        # Implementation using broadcasting
        Y_11 = np.sign(- wtX1 + wtX1.T)

        # Implementation using loops
        Y_11_slow = np.zeros((len(wtX1), len(wtX1)))
        for i, j in it.combinations(range(len(wtX1)), 2):
            Y_11_slow[i, j] = np.sign(wtX1[j] - wtX1[i])
            Y_11_slow[j, i] = -Y_11_slow[i, j]  # -0 == 0

        np.testing.assert_equal(Y_11, Y_11_slow)

        # -------------------------------------------
        wtX2 = np.random.RandomState(919).rand(199, 1)
        # Implementation using broadcasting
        Y_12 = np.sign(- wtX1 + wtX2.T)

        # Implementation using loops
        Y_12_slow = np.zeros((len(wtX1), len(wtX2)))
        for i, j in it.product(range(len(wtX1)), range(len(wtX2))):
            Y_12_slow[i, j] = np.sign(wtX2[j] - wtX1[i])

        np.testing.assert_equal(Y_12, Y_12_slow)


class TestAMatrixConstruction(unittest.TestCase):
    def test_that_class_labels_are_correctly_flipped(self):
        np.random.seed(90)

        for _ in range(100):
            rsvm = KernelRankSVC()

            # Create some random kernel matrix:
            X = np.random.rand(5, 2)
            rsvm._KX_fit = X @ X.T

            pairs = [(0, 1), (1, 2), (1, 3), (0, 3), (4, 1)]
            # Mol_0 elutes before Mol_1
            # Mol_1 elutes before Mol_2
            # Mol_1 elutes before Mol_3
            # Mol_0 elutes before Mol_3
            # Mol_4 elutes before Mol_1
            y = np.array([1, 1, 1, -1, -1])
            # Mol_0 elutes before Mol_3
            # Mol_4 elutes before Mol_1
            pairs[3] = (pairs[3][1], pairs[3][0])
            pairs[4] = (pairs[4][1], pairs[4][0])

            A_no_y = rsvm._build_A_matrix(pairs, n_samples=len(X), y=None)
            A_with_y_ref = np.diag(y) * A_no_y

            A_with_y = rsvm._build_A_matrix(pairs, n_samples=len(X), y=y)

            self.assertEqual(A_no_y.shape, (5, 5))
            self.assertEqual(A_with_y.shape, (5, 5))
            self.assertTrue((A_with_y_ref == A_with_y).all())

            # Pairwise kernel implemented directly implemented according to the formula.
            KX_pair_ref = np.multiply(np.outer(y, y), (A_no_y * rsvm._KX_fit * A_no_y.T))

            self.assertTrue((KX_pair_ref == A_with_y @ rsvm._KX_fit @ A_with_y.T).all())
            self.assertTrue((KX_pair_ref == A_with_y_ref @ rsvm._KX_fit @ A_with_y_ref.T).all())


class TestGetPairwiseLabels(unittest.TestCase):
    def test_without_class_balancing(self):
        pairs = [(0, 1), (1, 2), (1, 3), (0, 3), (4, 1)]
        pairs_out, y = get_pairwise_labels(pairs, balance_classes=False)

        self.assertListEqual(pairs, pairs_out)
        np.testing.assert_array_equal(y, np.ones((len(pairs), 1), dtype="int"))

    def test_with_class_balancing(self):
        np.random.seed(777)

        # odd number of pairs
        pairs = [(0, 1), (1, 2), (1, 3), (0, 3), (4, 1)]
        pairs_out, y = get_pairwise_labels(pairs, balance_classes=True)

        self.assertEqual(np.sum(y == 1), 3)
        self.assertEqual(np.sum(y == -1), 2)

        for idx, _ in enumerate(pairs):
            if y[idx] == 1:
                self.assertEqual(pairs[idx], pairs_out[idx])
            else:
                self.assertEqual(pairs[idx], (pairs_out[idx][1], pairs_out[idx][0]))

        # even number of pairs
        pairs = [(0, 1), (1, 2), (1, 3), (0, 3), (4, 1), (2, 1)]
        pairs_out, y = get_pairwise_labels(pairs, balance_classes=True)

        self.assertEqual(np.sum(y == 1), 3)
        self.assertEqual(np.sum(y == -1), 3)

        for idx, _ in enumerate(pairs):
            if y[idx] == 1:
                self.assertEqual(pairs[idx], pairs_out[idx])
            else:
                self.assertEqual(pairs[idx], (pairs_out[idx][1], pairs_out[idx][0]))

    def test_random_seed(self):
        pairs = [(0, 1), (1, 2), (1, 3), (0, 3), (4, 1), (2, 1)]

        pairs_out_1, _ = get_pairwise_labels(pairs, balance_classes=True, random_state=330)
        pairs_out_2, _ = get_pairwise_labels(pairs, balance_classes=True, random_state=787)
        pairs_out_3, _ = get_pairwise_labels(pairs, balance_classes=True, random_state=787)

        self.assertFalse((pairs_out_1 == pairs_out_2))
        self.assertFalse((pairs_out_1 == pairs_out_3))
        self.assertTrue((pairs_out_2 == pairs_out_3))


class Test_x_AKAt_y(unittest.TestCase):
    def init_example(self):
        rsvm = KernelRankSVC()

        # Create some random kernel matrix:
        X = np.random.rand(10, 3)
        rsvm._KX_fit = X @ X.T

        # Create A matrix
        pairs = [(0, 1), (1, 2), (1, 3), (0, 3), (4, 1), (5, 2), (7, 3)]
        y = np.array([1, 1, 1, -1, -1, 1, 1])
        pairs[3] = (pairs[3][1], pairs[3][0])
        pairs[4] = (pairs[4][1], pairs[4][0])
        rsvm._A = rsvm._build_A_matrix(pairs, n_samples=len(X), y=y)

        return rsvm, pairs

    def test_dimensions(self):
        np.random.seed(737)

        rsvm, pairs = self.init_example()

        n_pairs = len(pairs)
        x = np.random.uniform(0, 10, (n_pairs, 1))
        y = np.random.uniform(0, 10, (n_pairs, 1))

        self.assertEqual(rsvm._x_AKAt_y(None, None).shape, (n_pairs, n_pairs))
        self.assertEqual(rsvm._x_AKAt_y(None, y).shape, (n_pairs, 1))
        self.assertEqual(rsvm._x_AKAt_y(x, None).shape, (1, n_pairs))
        self.assertTrue(np.isscalar(rsvm._x_AKAt_y(x, y)))

    def test_values(self):
        np.random.seed(747)

        for _ in range(100):
            rsvm, pairs = self.init_example()
            n_pairs = len(pairs)

            x = np.random.uniform(0, 10, (n_pairs, 1))
            y = np.random.uniform(0, 10, (n_pairs, 1))

            # None, None
            AKAt = rsvm._A @ rsvm._KX_fit @ rsvm._A.T
            np.testing.assert_array_equal(AKAt, rsvm._x_AKAt_y(None, None))

            # None, y
            AKAty = rsvm._A @ rsvm._KX_fit @ rsvm._A.T @ y
            np.testing.assert_allclose(AKAty, rsvm._x_AKAt_y(None, y), rtol=1e-12)

            # x, None
            xtAKAt = x.T @ rsvm._A @ rsvm._KX_fit @ rsvm._A.T
            np.testing.assert_allclose(xtAKAt, rsvm._x_AKAt_y(x, None), rtol=1e-12)

            # x, y
            xtAKAty = (x.T @ rsvm._A @ rsvm._KX_fit @ rsvm._A.T @ y).item()
            np.testing.assert_allclose(xtAKAty, rsvm._x_AKAt_y(x, y), rtol=1e-12)


def slow_cindex(Y, P):
    """
    Function taken from the rlscore package: https://github.com/aatapa/RLScore as reference.
    """
    correct = Y
    predictions = P
    assert len(correct) == len(predictions)
    disagreement = 0.
    decisions = 0.
    for i in range(len(correct)):
        for j in range(len(correct)):
            if correct[i] > correct[j]:
                decisions += 1.
                if predictions[i] < predictions[j]:
                    disagreement += 1.
                elif predictions[i] == predictions[j]:
                    disagreement += 0.5
    # Disagreement error is not defined for cases where there
    # are no disagreeing pairs
    disagreement /= decisions
    return 1. - disagreement


class TestPointwiseScoringFunction(unittest.TestCase):
    def test_random_prediction(self):
        np.random.seed(767)

        for _ in range(25):
            y = np.random.random(100)
            y_pred = np.random.random(100)  # Predicted pseudo-scores

            perf = KernelRankSVC().score_pointwise_using_predictions(y_pred, y)
            perf_ref = slow_cindex(y, y_pred)

            np.testing.assert_allclose(perf, perf_ref)

    def test_corner_cases(self):
        y = np.array([1, 2, 3, 3, 4])

        y_pred = np.array([-4, 1, 5, 5, 7])
        perf = KernelRankSVC().score_pointwise_using_predictions(y_pred, y)
        self.assertEqual(perf, 1.0)

        y_pred = np.array([-4, 1, 8, 5, 7])
        perf = KernelRankSVC().score_pointwise_using_predictions(y_pred, y)
        self.assertEqual(perf, 8.0 / 9.0)

        y_pred = np.array([-4, 1, 8, 7, 7])
        perf = KernelRankSVC().score_pointwise_using_predictions(y_pred, y)
        self.assertEqual(perf, 7.5 / 9.0)

        y_pred = np.array([10, 9, 8, 7, 6])
        perf = KernelRankSVC().score_pointwise_using_predictions(y_pred, y)
        self.assertEqual(perf, 0.0)

        y_pred = np.array([10, 10, 10, 10, 10])
        perf = KernelRankSVC().score_pointwise_using_predictions(y_pred, y)
        self.assertEqual(perf, 0.5)


class TestPairwisePrediction(unittest.TestCase):
    def test_correctness(self):
        wtX1 = np.random.rand(193, 1)
        wtX2 = np.random.rand(78, 1)

        Y1 = np.sign(- wtX1 + wtX2.T)
        Y2 = - np.sign(wtX1 - wtX2.T)
        Y3 = np.sign(wtX2 - wtX1.T).T
        np.testing.assert_equal(Y1, Y2)
        np.testing.assert_equal(Y2, Y3)

        for (i, j) in it.product(range(10), range(5)):
            np.testing.assert_equal(Y1[i, j], np.sign(wtX2[j] - wtX1[i]))
            np.testing.assert_equal(Y2[i, j], np.sign(wtX2[j] - wtX1[i]))


class TestPairwiseScoringFunction(unittest.TestCase):
    def test_random_prediction(self):
        np.random.seed(767)

        def get_pairwise_perf(y, y_pred):
            pairs, _ = get_pairs_single_dataset(y, 0, np.inf)

            # Prediction matrix
            Y = np.zeros((len(y), len(y)))
            for i, j in pairs:
                Y[i, j] = np.sign(y_pred[j] - y_pred[i])
                Y[j, i] = -Y[i, j]

            perf = KernelRankSVC().score_pairwise_using_prediction(Y, pairs)

            return perf

        for _ in range(25):
            y = np.random.random(100)
            y_pred = np.random.random(100)  # Predicted pseudo-scores

            perf = get_pairwise_perf(y, y_pred)
            perf_ref = slow_cindex(y, y_pred)

            np.testing.assert_allclose(perf, perf_ref)

    def test_corner_cases(self):
        def get_pairwise_perf(y, y_pred):
            pairs, _ = get_pairs_single_dataset(y, 0, np.inf)

            # Prediction matrix
            Y = np.zeros((len(y), len(y)))
            for i, j in pairs:
                Y[i, j] = np.sign(y_pred[j] - y_pred[i])
                Y[j, i] = -Y[i, j]

            perf = KernelRankSVC().score_pairwise_using_prediction(Y, pairs)

            return perf

        y = np.array([1, 2, 3, 3, 4])

        y_pred = np.array([-4, 1, 5, 5, 7])
        perf = get_pairwise_perf(y, y_pred)
        self.assertEqual(perf, 1.0)

        y_pred = np.array([-4, 1, 8, 5, 7])
        perf = get_pairwise_perf(y, y_pred)
        self.assertEqual(perf, 8.0/9.0)

        y_pred = np.array([-4, 1, 8, 7, 7])
        perf = get_pairwise_perf(y, y_pred)
        self.assertEqual(perf, 7.5/9.0)

        y_pred = np.array([10, 9, 8, 7, 6])
        perf = get_pairwise_perf(y, y_pred)
        self.assertEqual(perf, 0.0)

        y_pred = np.array([10, 10, 10, 10, 10])
        perf = get_pairwise_perf(y, y_pred)
        self.assertEqual(perf, 0.5)


if __name__ == '__main__':
    unittest.main()
