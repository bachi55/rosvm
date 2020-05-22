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

from rank_svm_cls_2 import KernelRankSVC


class TestAMatrixConstruction(unittest.TestCase):
    def test_small_example(self):
        # -----------------------------------------------
        pairs = [(0, 1), (1, 2), (1, 3), (0, 3), (5, 1)]
        y = [1, 1, 1, -1, -1]

        n_samples = 6
        A = KernelRankSVC._build_A_matrix(pairs, y, n_samples=n_samples)

        self.assertEqual((5, n_samples), A.shape)
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


if __name__ == '__main__':
    unittest.main()
