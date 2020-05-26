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
import numpy as np
import itertools as it
import cvxpy as cp

from sklearn.preprocessing import KernelCenterer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def frobenius_product(A, B=None):
    """
    Calculate the Frobenius product: <A, B>_F = Tr[A' * B]

    :param A: array-like, shape=(N, D), matrix A
    :param B: array-like, shape=(N, D), matrix B

    :return: scalar, Frobenius product
    """
    if B is None:
        B = A

    if A.shape != B.shape:
        raise ValueError("A and B matrix must have the same shape")

    return np.sum(np.multiply(A, B))


def kernel_alignment(K1, K2, centered=False):
    """
    Calculate (centered) kernel alignment score between the two kernels Kx and Ky:

        A(Kx, Ky) = <Kx, Ky>_F / (|Kx|_F * |Ky|_F)
                  = <Kx, Ky>_F / sqrt(<Kx, Kx>_F * <Ky, Ky>_F)

        |A|_F = sqrt(<A, A>_F)

    :param K1: array-like, shape = (n_samples, n_samples)
    :param K2: array-like, shape = (n_samples, n_samples)
    :param centered: boolean, indicating whether the centered kernel alignment should be
        calculated.

    :return: scalar, (centered) kernel alignment score
    """
    if K1.shape != K2.shape:
        raise ValueError("Matrices must have same shape.")

    if centered:
        K1_c = KernelCenterer().fit_transform(K1)
        K2_c = KernelCenterer().fit_transform(K2)
    else:
        K1_c = K1
        K2_c = K2

    # Calculate alignment
    fprod_12 = frobenius_product(K1_c, K2_c)
    fprod_11 = frobenius_product(K1_c)
    fprod_22 = frobenius_product(K2_c)

    return fprod_12 / np.sqrt(fprod_11 * fprod_22)


class LinearMKLer(BaseEstimator, TransformerMixin):
    def __init__(self, method="unimkl", center_before_combine=False):
        """
        Class to linearly combine a set of kernels:

            K = a_1 * K_1 + a_2 * K_2 + ... + a_m * K_m,

        whereby the weight vector a = (a_1, ..., a_m) can be learning using different methods.

        :param method: string, method to learn the weight vector. (default = 'unimkl')
        :param center_before_combine: boolean, indicating whether the input kernels should be
            centered before they are combined. (default = False)
        """
        if method not in ["unimkl", "align", "alignf"]:
            raise ValueError("Invalid MKL method: %s" % method)

        # Method to learn the MKL weights
        self.method = method

        # Should the input kernels be centered before they are combined?
        self.center_before_combine = center_before_combine

        self._kernel_weights = None  # stores the kernel weights
        self._kernel_centerer = None

    def fit(self, Kx, Ky=None, **fit_params):
        """
        Learn the function MKL weights from training data.

        :param Kx: list of array-like, length = n_kernel, K_i with shape = (n_samples_train, n_samples_train),
            input kernel matrices.
        :param Ky: array-like, shape = (n_samples, n_samples), output kernel matrix (default = None)
        :return: self
        """
        if (self.method != "unimkl") and (Ky is None):
            raise ValueError("Alignment based kernel weights require an output kernel Ky.")

        if not isinstance(Kx, list):
            Kx = [Kx]  # we can accept a single kernel

        # Number of input kernels
        n_kernel = len(Kx)

        # Learn weights
        if self.method == "unimkl":

            self._kernel_weights = np.ones((n_kernel,), dtype="float") / n_kernel

        elif self.method == "align":

            self._kernel_weights = np.array([kernel_alignment(Kx_k, Ky, centered=True) for Kx_k in Kx])

        elif self.method == "alignf":
            # Center input kernels
            Kx_c = [KernelCenterer().fit_transform(Kx_k) for Kx_k in Kx]

            # Compute Frobenius products between the input kernels
            M = np.full((n_kernel, n_kernel), fill_value=np.nan)
            for i, j in it.combinations_with_replacement(range(n_kernel), 2):
                M[i, j] = frobenius_product(Kx_c[i], Kx_c[j])
                M[j, i] = M[i, j]
            assert (not np.any(np.isnan(M)))

            # Compute Frobenius products between input and output
            a = np.array([frobenius_product(Kx_c_k, Ky) for Kx_c_k in Kx_c])  # shape = (n_kernel, )

            # Learn the weights
            x = cp.Variable((n_kernel,), nonneg=True)  # x >= 0
            prob = cp.Problem(cp.Minimize(cp.quad_form(x, M) - 2 * a @ x), constraints=[x >= 0])
            prob.solve()

            # Extract optimal weights and normalize
            self._kernel_weights = x.value / np.linalg.norm(x.value)

            np.allclose(np.linalg.norm(self._kernel_weights), 1.0)  # For alignf the kernel weights must be normalized

        # Fit kernel centerer
        if self.center_before_combine:
            self._kernel_centerer = [KernelCenterer().fit(Kx_k) for Kx_k in Kx]
        else:
            self._kernel_centerer = None

        return self

    def transform(self, Kx, Ky=None):
        """
        Take a set of kernels and linearly combine them using the trained weights.

        :param Kx: list of array-like, length = n_kernel, K_i with shape = (n_samples_A, n_samples_B),
            input kernel matrices. Note: n_samples_B = n_samples_train, i.e. we generally assume that
            the test kernel has shape = (n_test, n_train). That is to be aligned with the sklearn
            framework.
        :param Ky: unused argument

        :return: array-like, shape = (n_samples_A, n_samples_B), combined kernel.
        """
        check_is_fitted(self, ["_kernel_weights", "_kernel_centerer"])

        if not isinstance(Kx, list):
            raise ValueError("Input kernel(s) must be provided as list.")

        if len(self._kernel_weights) != len(Kx):
            raise ValueError("Number of input kernels does not match the number of kernel weights.")

        # Combined kernel matrix
        K_out = np.zeros_like(Kx[0])

        for k in range(len(self._kernel_weights)):
            if self.center_before_combine:
                Kx_k = self._kernel_centerer[k].transform(Kx[k])
            else:
                Kx_k = Kx[k]

            K_out += (self._kernel_weights[k] * Kx_k)

        return K_out
