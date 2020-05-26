####
#
# The MIT License (MIT)
#
# Copyright 2019, 2020 Eric Bach <eric.bach@aalto.fi>
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

from ranksvm.mkl_utils import frobenius_product, kernel_alignment, LinearMKLer

from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import make_classification
from sklearn.metrics.pairwise import linear_kernel


class TestFrobeniusProduct(unittest.TestCase):
    def test_correctness(self):
        for _ in range(100):
            A = np.random.uniform(-10, 10, (25, 60))
            B = np.random.uniform(-5, 5, (25, 60))

            # <A, A>_F
            AA_F = frobenius_product(A)
            np.testing.assert_allclose(AA_F, np.trace(A.T @ A))

            # <A, B>_F
            AB_F = frobenius_product(A, B)
            np.testing.assert_allclose(AB_F, np.trace(A.T @ B))


class TestKernelAlignment(unittest.TestCase):
    def test_corner_cases(self):
        A = np.random.uniform(-10, 10, (25, 60))
        KA = A @ A.T

        # Kernel aligned with it self
        np.testing.assert_equal(kernel_alignment(KA, KA, False), 1.0)
        np.testing.assert_equal(kernel_alignment(KA, KA, True), 1.0)

    def test_correctness(self):
        self.skipTest("How to test?")


class TestLinearMKLer(unittest.TestCase):
    def test_unimkl(self):
        # Set up list of features
        X = [
            np.random.RandomState(1932).randn(102, 738),
            np.random.RandomState(455).randn(102, 12),
            np.random.RandomState(212).randn(102, 231),
            np.random.RandomState(32).randn(102, 324)
        ]

        # Set up list of kernels
        Kx = [
            X[0] @ X[0].T,
            X[1] @ X[1].T,
            X[2] @ X[2].T,
            X[3] @ X[3].T
        ]

        # Split training and test
        train, test = next(ShuffleSplit(random_state=102).split(Kx[0]))

        # Fit the transformer
        trans = LinearMKLer(method="unimkl").fit([Kx_k[np.ix_(train, train)] for Kx_k in Kx])
        assert (isinstance(trans, LinearMKLer)), "Fit function must return a 'LinearMKLer'."

        # Check the kernel weights
        np.testing.assert_equal(trans._kernel_weights, np.ones((4,)) / 4)

        # Transform (combine) the kernels
        np.testing.assert_equal(
            trans.transform([Kx_k[np.ix_(train, train)] for Kx_k in Kx]),
            np.mean(np.array([Kx_k[np.ix_(train, train)] for Kx_k in Kx]), axis=0))

        np.testing.assert_equal(
            trans.transform([Kx_k[np.ix_(test, train)] for Kx_k in Kx]),
            np.mean(np.array([Kx_k[np.ix_(test, train)] for Kx_k in Kx]), axis=0))

    def test_align(self):
        # Create simple classification dataset
        n_features = 6
        n_informative = 2
        n_redundant = 2
        X, y = make_classification(n_redundant=n_redundant, n_features=n_features, n_informative=n_informative,
                                   n_samples=200, random_state=192, shuffle=False, n_clusters_per_class=1)
        y[y == 0] = -1

        # Calculate kernels for the different types features and the output
        n_noise = n_features - n_informative - n_redundant
        Kx_inf = linear_kernel(X[:, :n_informative])
        Kx_red = linear_kernel(X[:, n_informative:(n_informative + n_redundant)])
        Kx_noise = linear_kernel(X[:, -n_noise:])
        Ky = np.outer(y, y)

        # --------------------------------------
        # Fit the transformer
        trans = LinearMKLer(method="align").fit([Kx_inf, Kx_red, Kx_noise, Kx_noise], Ky)
        assert (isinstance(trans, LinearMKLer)), "Fit function must return a 'LinearMKLer'."

        # Check the kernel weights
        np.testing.assert_allclose(np.round(trans._kernel_weights, 6),
                                   np.array([0.719934, 0.670636, 0.000601, 0.000601]))

        # Transform (combine) the kernels
        Kx_mkl = trans.transform([Kx_inf, Kx_red, Kx_noise, Kx_noise])
        self.assertTrue(kernel_alignment(Kx_mkl, Ky, True) > 0.7)

        # --------------------------------------
        # Fit the transformer
        trans = LinearMKLer(method="align", center_before_combine=True).fit([Kx_inf, Kx_red, Kx_noise, Kx_noise], Ky)
        assert (isinstance(trans, LinearMKLer)), "Fit function must return a 'LinearMKLer'."

        # Check the kernel weights
        np.testing.assert_allclose(np.round(trans._kernel_weights, 6),
                                   np.array([0.719934, 0.670636, 0.000601, 0.000601]))

        # Transform (combine) the kernels
        Kx_mkl = trans.transform([Kx_inf, Kx_red, Kx_noise, Kx_noise])
        self.assertTrue(kernel_alignment(Kx_mkl, Ky, True) > 0.7)
        np.testing.assert_allclose(np.sum(Kx_mkl, axis=0), np.zeros((Kx_mkl.shape[0],)), atol=1e-12)

    def test_alignf(self):
        # Create simple classification dataset
        n_features = 6
        n_informative = 2
        n_redundant = 2
        X, y = make_classification(n_redundant=n_redundant, n_features=n_features, n_informative=n_informative,
                                   n_samples=200, random_state=192, shuffle=False, n_clusters_per_class=1)
        y[y == 0] = -1

        # Calculate kernels for the different types features and the output
        n_noise = n_features - n_informative - n_redundant
        Kx_inf = linear_kernel(X[:, :n_informative])
        Kx_red = linear_kernel(X[:, n_informative:(n_informative + n_redundant)])
        Kx_noise = linear_kernel(X[:, -n_noise:])
        Ky = np.outer(y, y)

        # --------------------------------------
        # Fit the transformer
        Kx_l = [Kx_red, Kx_noise, Kx_noise, Kx_inf]
        trans = LinearMKLer(method="alignf").fit(Kx_l, Ky)

        # Check the kernel weights
        np.testing.assert_equal(trans._kernel_weights, np.array([0, 0, 0, 1]))
        self.assertEqual(np.linalg.norm(trans._kernel_weights), 1.)

        # Transform (combine) the kernels
        Kx_mkl = trans.transform(Kx_l)
        np.testing.assert_equal(kernel_alignment(Kx_mkl, Ky), kernel_alignment(Kx_inf, Ky))

        # --------------------------------------
        # Fit the transformer
        Kx_l = [Kx_red, Kx_noise, Kx_noise, Kx_inf]
        trans = LinearMKLer(method="alignf", center_before_combine=True).fit(Kx_l, Ky)

        # Check the kernel weights
        np.testing.assert_equal(trans._kernel_weights, np.array([0, 0, 0, 1]))
        self.assertEqual(np.linalg.norm(trans._kernel_weights), 1.)

        # Transform (combine) the kernels
        Kx_mkl = trans.transform(Kx_l)
        np.testing.assert_equal(kernel_alignment(Kx_mkl, Ky), kernel_alignment(Kx_inf, Ky, centered=True))
        np.testing.assert_allclose(np.sum(Kx_mkl, axis=0), np.zeros((Kx_mkl.shape[0],)), atol=1e-12)

        # --------------------------------------
        # Fit the transformer
        Kx_l = [Kx_red, Kx_noise, Kx_noise]
        trans = LinearMKLer(method="alignf").fit(Kx_l, Ky)

        # Check the kernel weights
        np.testing.assert_equal(trans._kernel_weights, np.array([1, 0, 0]))
        self.assertEqual(np.linalg.norm(trans._kernel_weights), 1.)

        # Transform (combine) the kernels
        Kx_mkl = trans.transform(Kx_l)
        np.testing.assert_equal(kernel_alignment(Kx_mkl, Ky), kernel_alignment(Kx_red, Ky))

        # --------------------------------------
        # Fit the transformer
        Kx_l = [Kx_red, Kx_inf, Kx_noise, Kx_inf]
        trans = LinearMKLer(method="alignf").fit(Kx_l, Ky)

        # Check the kernel weights
        np.testing.assert_allclose(trans._kernel_weights, np.array([0, 0.70710678, 0, 0.70710678]), atol=1e-8)
        self.assertEqual(np.linalg.norm(trans._kernel_weights), 1.)

        # Transform (combine) the kernels
        Kx_mkl = trans.transform(Kx_l)
        np.testing.assert_allclose(Kx_mkl, 2 * 0.70710678 * Kx_inf, atol=1e-8)

        # --------------------------------------
        # Fit the transformer
        Kx_l = [Kx_red, Kx_red, Kx_noise, Kx_noise]
        trans = LinearMKLer(method="alignf").fit(Kx_l, Ky)

        # Check the kernel weights
        np.testing.assert_allclose(trans._kernel_weights, np.array([0.70710678, 0.70710678, 0, 0]), atol=1e-8)
        self.assertEqual(np.linalg.norm(trans._kernel_weights), 1.)

        # Transform (combine) the kernels
        Kx_mkl = trans.transform(Kx_l)
        np.testing.assert_allclose(Kx_mkl, 2 * 0.70710678 * Kx_red, atol=1e-8)
