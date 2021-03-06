####
#
# The MIT License (MIT)
#
# Copyright 2018-2020 Eric Bach <eric.bach@aalto.fi>
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
import pickle

from sklearn.model_selection import check_cv
from sklearn.model_selection._split import type_of_target, _CVIterableWrapper, _safe_indexing

from rosvm.ranksvm.rank_svm_cls import KernelRankSVC, Labels
from rosvm.ranksvm.tests.gradient_of_exterior_features import fAndG


def cindex_rlscore_reference(Y, P):
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


class TestPMatrixConstruction(unittest.TestCase):
    def test_small_example(self):
        pairs = [(0, 1), (1, 2), (1, 3), (0, 3), (5, 1)]
        n_samples = 6
        n_pairs = len(pairs)

        P_0, P_1 = KernelRankSVC._build_P_matrices(pairs, n_samples)

        self.assertEqual((n_pairs, n_samples), P_0.shape)
        self.assertEqual((n_pairs, n_samples), P_1.shape)

        np.testing.assert_equal(
            np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1]], dtype="float"),
            P_0.todense()
        )

        np.testing.assert_equal(
            np.array([[0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 1, 0, 0, 0, 0]], dtype="float"),
            P_1.todense()
        )


class TestAMatrixConstruction(unittest.TestCase):
    def test_small_example(self):
        # -----------------------------------------------
        pairs = [(0, 1), (1, 2), (1, 3), (0, 3), (5, 1)]
        y = [1, 1, 1, -1, -1]
        n_pairs = len(pairs)

        n_samples = 6
        A = KernelRankSVC._build_A_matrix(pairs, y, n_samples=n_samples)

        self.assertEqual((n_pairs, n_samples), A.shape)
        np.testing.assert_equal(
            np.array([[1, -1, 0, 0, 0, 0],
                      [0, 1, -1, 0, 0, 0],
                      [0, 1, 0, -1, 0, 0],
                      [-1, 0, 0, 1, 0, 0],
                      [0, 1, 0, 0, 0, -1]], dtype="float"),
            A.todense()
        )

        # -----------------------------------------------
        y = [1, 1, 1, 1, 1]
        A = KernelRankSVC._build_A_matrix(pairs, y, n_samples=n_samples)
        np.testing.assert_equal(
            np.array([[1, -1, 0, 0, 0, 0],
                      [0, 1, -1, 0, 0, 0],
                      [0, 1, 0, -1, 0, 0],
                      [1, 0, 0, -1, 0, 0],
                      [0, -1, 0, 0, 0, 1]], dtype="float"),
            A.todense()
        )

    def test_row_labels_are_correctly_applied(self):
        rs = np.random.RandomState(120)
        for _ in range(10):
            pairs = [(0, 1), (1, 2), (1, 3), (0, 3), (6, 1), (4, 3)]
            y = [np.sign(rs.randint(-1, 1)) for _ in range(len(pairs))]
            A = KernelRankSVC._build_A_matrix(pairs, y, 10)

            # Apply row labels in an alternative way to the matrix A
            y_all_pos = [1 for _ in range(len(pairs))]
            A_all_pos = KernelRankSVC._build_A_matrix(pairs, y_all_pos, 10)
            A_all_pos = np.diag(y) @ A_all_pos

            np.testing.assert_equal(A_all_pos, A.todense())

    def test_pairwise_labels_are_correct(self):
        """
        Here we want to test, whether we it is correct to pull in the labels into the A matrix.
        In the RankSVM dual we need to evaluate the following expression:

              y @ y.T * A @ K @ A.T
            = diag(y) @ A @ K @ A.T @ diag(y.T)    # see [1] Eq. 2.11, here u = v
            = (Dy @ A) @ K @ (A.T @ Dy)

        Here, "@" is the a matrix or dot-product, and "*" refers to the pointwise matrix or
        vector product (also called Hadamard product), and ".T" is the transposed.

        [1] "Hadamard products and multivariate statistical analysis", Styan (1973)
        """
        rs = np.random.RandomState(90)

        for _ in range(10):
            # Create some random kernel matrix:
            X = np.random.rand(rs.randint(7, 20), 3)
            K = X @ X.T

            pairs = [(0, 1), (1, 2), (1, 3), (0, 3), (6, 1), (4, 3)]
            y = [np.sign(rs.randint(-1, 1)) for _ in range(len(pairs))]
            A = KernelRankSVC._build_A_matrix(pairs, y, len(X))

            self.assertEqual((len(pairs), len(X)), A.shape)

            # Apply row labels in an alternative way to the matrix A
            y_all_pos = [1 for _ in range(len(pairs))]
            A_all_pos = KernelRankSVC._build_A_matrix(pairs, y_all_pos, len(X))

            np.testing.assert_equal(np.outer(y, y) * (A_all_pos @ K @ A_all_pos.T), A @ K @ A.T)


class TestPointwiseScoringFunction(unittest.TestCase):
    def test_random_prediction(self):
        rs = np.random.RandomState(767)

        for i in range(25):
            y = rs.random(100)
            y_pred = rs.random(100)  # Predicted pseudo-scores

            perf, _ = KernelRankSVC().score_pointwise_using_predictions(y, y_pred)
            perf_ref = cindex_rlscore_reference(y, y_pred)

            np.testing.assert_allclose(perf, perf_ref)

    def test_corner_cases(self):
        y = np.array([1, 3, 3, 4, 2])

        y_pred = np.array([-4, 5, 5, 7, 1])
        perf, n_test_pairs = KernelRankSVC().score_pointwise_using_predictions(y, y_pred)
        self.assertEqual(perf, 1.0)
        self.assertEqual(n_test_pairs, 9)

        y_pred = np.array([-4, 8, 5, 7, 1])
        perf, n_test_pairs = KernelRankSVC().score_pointwise_using_predictions(y, y_pred)
        self.assertEqual(perf, 8.0 / 9.0)

        y_pred = np.array([-4, 8, 7, 7, 1])
        perf, _ = KernelRankSVC().score_pointwise_using_predictions(y, y_pred)
        self.assertEqual(perf, 7.5 / 9.0)

        y_pred = np.array([10, 8, 7, 6, 9])
        perf, _ = KernelRankSVC().score_pointwise_using_predictions(y, y_pred)
        self.assertEqual(perf, 0.0)

        y_pred = np.array([10, 10, 10, 10, 10])
        perf, _ = KernelRankSVC().score_pointwise_using_predictions(y, y_pred)
        self.assertEqual(perf, 0.5)


class TestPairwiseScoringFunction(unittest.TestCase):
    @staticmethod
    def make_Y_matrix(y):
        return y[:, np.newaxis] - y[np.newaxis, :]

    def test_random_prediction_against_pointwise_scoring(self):
        rs = np.random.RandomState(767)

        for i in range(25):
            y = rs.random(100)
            y_pred = rs.random(100)  # Predicted pseudo-scores

            Y = self.make_Y_matrix(y)
            Y_pred = self.make_Y_matrix(y_pred)

            perf_pair, _ = KernelRankSVC().score_pairwise_using_prediction_SLOW(Y, Y_pred)
            perf_point, _ = KernelRankSVC().score_pointwise_using_predictions(y, y_pred)

            np.testing.assert_allclose(perf_pair, perf_point)

    def test_corner_cases(self):
        Y = self.make_Y_matrix(np.array([1, 3, 3, 4, 2]))

        Y_pred = self.make_Y_matrix(np.array([-4, 5, 5, 7, 1]))
        perf, n_test_pairs = KernelRankSVC().score_pairwise_using_prediction_SLOW(Y, Y_pred)
        self.assertEqual(perf, 1.0)
        self.assertEqual(n_test_pairs, 9)

        Y_pred = self.make_Y_matrix(np.array([-4, 8, 5, 7, 1]))
        perf, n_test_pairs = KernelRankSVC().score_pairwise_using_prediction_SLOW(Y, Y_pred)
        self.assertEqual(perf, 8.0 / 9.0)

        Y_pred = self.make_Y_matrix(np.array([-4, 8, 7, 7, 1]))
        perf, _ = KernelRankSVC().score_pairwise_using_prediction_SLOW(Y, Y_pred)
        self.assertEqual(perf, 7.5 / 9.0)

        Y_pred = self.make_Y_matrix(np.array([10, 8, 7, 6, 9]))
        perf, _ = KernelRankSVC().score_pairwise_using_prediction_SLOW(Y, Y_pred)
        self.assertEqual(perf, 0.0)

        Y_pred = self.make_Y_matrix(np.array([10, 10, 10, 10, 10]))
        perf, _ = KernelRankSVC().score_pairwise_using_prediction_SLOW(Y, Y_pred)
        self.assertEqual(perf, 0.5)


class TestVectorizedPairwiseScoringFunction(unittest.TestCase):
    @staticmethod
    def make_Y_matrix(y):
        return y[:, np.newaxis] - y[np.newaxis, :]

    def test_random_prediction_against_pointwise_scoring(self):
        rs = np.random.RandomState(767)

        for i in range(25):
            y = rs.random(100)
            y_pred = rs.random(100)  # Predicted pseudo-scores

            Y = self.make_Y_matrix(y)
            Y_pred = self.make_Y_matrix(y_pred)

            perf_pair, _ = KernelRankSVC().score_pairwise_using_prediction(Y, Y_pred)
            perf_point, _ = KernelRankSVC().score_pointwise_using_predictions(y, y_pred)

            np.testing.assert_allclose(perf_pair, perf_point)

    def test_corner_cases(self):
        Y = self.make_Y_matrix(np.array([1, 3, 3, 4, 2]))

        Y_pred = self.make_Y_matrix(np.array([-4, 5, 5, 7, 1]))
        perf, n_test_pairs = KernelRankSVC().score_pairwise_using_prediction(Y, Y_pred)
        self.assertEqual(perf, 1.0)
        self.assertEqual(n_test_pairs, 9)

        Y_pred = self.make_Y_matrix(np.array([-4, 8, 5, 7, 1]))
        perf, n_test_pairs = KernelRankSVC().score_pairwise_using_prediction(Y, Y_pred)
        self.assertEqual(perf, 8.0 / 9.0)

        Y_pred = self.make_Y_matrix(np.array([-4, 8, 7, 7, 1]))
        perf, _ = KernelRankSVC().score_pairwise_using_prediction(Y, Y_pred)
        self.assertEqual(perf, 7.5 / 9.0)

        Y_pred = self.make_Y_matrix(np.array([10, 8, 7, 6, 9]))
        perf, _ = KernelRankSVC().score_pairwise_using_prediction(Y, Y_pred)
        self.assertEqual(perf, 0.0)

        Y_pred = self.make_Y_matrix(np.array([10, 10, 10, 10, 10]))
        perf, _ = KernelRankSVC().score_pairwise_using_prediction(Y, Y_pred)
        self.assertEqual(perf, 0.5)


class TestDualGradients(unittest.TestCase):
    def test_exterior_features(self):
        """
        Adaptation of the gradient test code provided by: www.matrixcalculus.org.
        """
        # Set up data
        rs = np.random.RandomState(1929)

        for _ in range(10):
            n_samples = rs.randint(4, 10)
            K = rs.randn(n_samples, n_samples)
            K = 0.5 * (K + K.T)  # make it symmetric

            pairs = []
            y = []
            for _ in range(rs.randint(3, 10)):
                pairs.append(tuple(rs.choice(np.arange(n_samples), 2, replace=False)))
                y.append(rs.choice([1, -1], 1).item())

            P_0, P_1 = KernelRankSVC._build_P_matrices(pairs, n_samples)
            n_pairs = P_0.shape[0]

            alpha = rs.rand(n_pairs)
            assert np.all(alpha > 0)  # Dual variables are always positive

            # Check the gradient
            rsvm = KernelRankSVC()
            rsvm.KX_train_ = K
            rsvm.P_0_ = P_0
            rsvm.P_1_ = P_1
            rsvm.py_train_ = y

            # t = 1e-6
            # delta = rs.rand(n_pairs)
            # f1, _ = fAndG(K=K, P=P_0, Q=P_1, alpha=alpha + t * delta, y=y)
            # f2, _ = fAndG(K=K, P=P_0, Q=P_1, alpha=alpha - t * delta, y=y)

            _, g_ref = fAndG(K=K, P=P_0.toarray(), Q=P_1.toarray(), alpha=alpha, y=np.array(y))
            g = rsvm._grad_exterior_feat(alpha)  # own gradient implementation

            np.testing.assert_allclose(g_ref, g)


class TestAlphaThreshold(unittest.TestCase):
    def test_correctness(self):
        alpha_bnd = KernelRankSVC._bound_alpha(np.array([0.001, 0.01, 0.1, 0.99, 0.999, 1.0]), 0.01, 0, 1)
        np.testing.assert_equal(alpha_bnd, np.array([0, 0, 0.1, 1.0, 1.0, 1.0]))

        alpha_bnd = KernelRankSVC._bound_alpha(np.array([0.001, 0.01, 0.1, 0.99, 0.999, 1.0]), 0.001, 0, 1)
        np.testing.assert_equal(alpha_bnd, np.array([0, 0.01, 0.1, 0.99, 1.0, 1.0]))

        alpha_bnd = KernelRankSVC._bound_alpha(np.array([0.001001, 0.01, 0.1, 0.99, 0.99899, 1.0]), 0.001, 0, 1)
        np.testing.assert_equal(alpha_bnd, np.array([0.001001, 0.01, 0.1, 0.99, 0.99899, 1.0]))

        alpha_bnd = KernelRankSVC._bound_alpha(np.array([0.001001, 0.01, 0.1, 0.99, 0.99899, 1.0]), 0.0001, 0, 1)
        np.testing.assert_equal(alpha_bnd, np.array([0.001001, 0.01, 0.1, 0.99, 0.99899, 1.0]))


class TestLabelsClass(unittest.TestCase):
    def test_passes_sklearn_test_for_y(self):
        rts = [1, 2.1, 3, 9, 18, 16, 1, 2, 0]
        dss_int = [1, 1, 2, 2, 2, 1, 1, 1, 2]
        dss_str = ["A", "A", "C", "C", "A", "A", "A", "C", "C"]

        # type_of_target
        self.assertIsInstance(type_of_target(Labels(rts, dss_int)), str)
        self.assertIsInstance(type_of_target(Labels(rts, dss_str)), str)

        # check_cv
        self.assertIsInstance(check_cv(Labels(rts, dss_int)), _CVIterableWrapper)
        self.assertIsInstance(check_cv(Labels(rts, dss_str)), _CVIterableWrapper)

        # _safe_indexing
        self.assertIsInstance(_safe_indexing(Labels(rts, dss_int), [1, 7, 2]), Labels)
        self.assertIsInstance(_safe_indexing(Labels(rts, dss_str), [1, 7, 2]), Labels)
        self.assertIsInstance(_safe_indexing(Labels(rts, dss_int),
                                             [True, True, False, False, False, True, False, True, True]), Labels)
        self.assertIsInstance(_safe_indexing(Labels(rts, dss_str),
                                             [True, True, False, False, False, True, False, True, True]), Labels)
        self.assertIsInstance(_safe_indexing(Labels(rts, dss_int), np.array([1, 7, 2])), Labels)
        self.assertIsInstance(_safe_indexing(
            Labels(rts, dss_str), np.array([True, True, False, False, False, True, False, True, True])), Labels)
        self.assertIsInstance(_safe_indexing(Labels(rts, dss_int), slice(4, 7)), Labels)

    def test_equality_check(self):
        rts_A = [1, 9, 8, 0]
        dss_A = ["A", "B", "A", "B"]

        rts_B = [1, 7, 8, 0]
        dss_B = ["A", "A", "B", "B"]

        rts_C = [1, 9, 8]
        dss_C = ["A", "B", "A"]

        self.assertEqual(Labels(rts_A, dss_A), Labels(rts_A, dss_A))
        self.assertNotEqual(Labels(rts_A, dss_A), Labels(rts_A, dss_B))
        self.assertNotEqual(Labels(rts_A, dss_A), Labels(rts_B, dss_A))
        self.assertNotEqual(Labels(rts_A, dss_A), Labels(rts_B, dss_B))
        self.assertNotEqual(Labels(rts_C, dss_C), Labels(rts_A, dss_A))

    def test_subsetting(self):
        rts = [1, 2, 3, 9, 18, 16, 1, 2, 0]
        dss = ["A", "A", "C", "C", "A", "A", "A", "C", "C"]

        y = Labels(rts, dss)

        # Integer indexing
        for i, rt_ds in enumerate(list(zip(rts, dss))):
            self.assertEqual(rt_ds, y[i])

        # Slice indexing
        self.assertEqual(Labels(rts[3:7], dss[3:7]), y[3:7])
        self.assertNotEqual(Labels(rts[3:7], dss[3:7]), y[1:7])

        # Indexing using integer list
        self.assertEqual(Labels([rts[i] for i in [1, 7, 2]], [dss[i] for i in [1, 7, 2]]), y[[1, 7, 2]])
        self.assertNotEqual(Labels([rts[i] for i in [1, 7, 2]], [dss[i] for i in [1, 7, 2]]), y[[1, 2, 7]])

        # Indexing using numpy ndarray
        self.assertEqual(Labels([rts[i] for i in [1, 7, 2]], [dss[i] for i in [1, 7, 2]]),
                         y[np.array([1, 7, 2])])

        # Indexing using boolean ndarray
        b = np.zeros_like(rts, dtype=bool)
        b[[1, 7, 2]] = True
        self.assertEqual(Labels([rts[i] for i in [1, 2, 7]], [dss[i] for i in [1, 2, 7]]), y[b])
        self.assertEqual(Labels([rts[i] for i in [1, 2, 7]], [dss[i] for i in [1, 2, 7]]), y[b.tolist()])

    def test_iterator(self):
        rts = [1, 2, 3, 9, 18, 16, 1, 2, 0]
        dss = ["A", "A", "C", "C", "A", "A", "A", "C", "C"]
        y = Labels(rts, dss)

        # Integer indexing
        rts_dss = list(zip(rts, dss))
        for i, y_i in enumerate(y):
            self.assertEqual(rts_dss[i], y_i)

    def test_is_pickleable(self):
        rts = [1, 2, 3, 9, 18, 16, 1, 2, 0]
        dss = ["A", "A", "C", "C", "A", "A", "A", "C", "C"]
        y = Labels(rts, dss)

        self.assertEqual(pickle.loads(pickle.dumps(y)), y)

    def test_getting_ds_indices(self):
        rts = [1, 2, 3, 9, 18, 16, 1, 2, 0, 3.1, 2.1, 0.3]
        dss = ["A", "A", "C", "C", "A", "A", "A", "C", "C", "A", "B", "B"]
        y = Labels(rts, dss)

        self.assertEqual([0, 1, 4, 5, 6, 9], y.get_idc_for_ds("A"))
        self.assertEqual([10, 11], y.get_idc_for_ds("B"))
        self.assertEqual([2, 3, 7, 8], y.get_idc_for_ds("C"))

        with self.assertRaises(KeyError):
            y.get_idc_for_ds("D")

        self.assertEqual([], y.get_idc_for_ds("D", on_missing_raise=False))


if __name__ == '__main__':
    unittest.main()

