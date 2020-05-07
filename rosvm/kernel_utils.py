####
#
# The MIT License (MIT)
#
# Copyright 2019 Eric Bach <eric.bach@aalto.fi>
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
import scipy.sparse as sp
import itertools as it

from sklearn.metrics.pairwise import manhattan_distances


def check_input(X, Y, datatype=None, shallow=False):
    """
    Function to check whether the two input sets A and B are compatible.

    :param X: array-like, shape = (n_samples_A, n_features), examples A
    :param Y: array-like, shape = (n_samples_B, n_features), examples B
    :param datatype: string, used to specify constraints on the input data (type)

    :return: X, Y, is_sparse. X is simply passed through. If Y is None, than it
        will be equal X otherwise it is also just passed through. is_sparse is
        a boolean indicating whether X and Y are sparse matrices
    """
    if Y is None:
        Y = X

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Number of features for set A and B must match: %d vs %d." % (
            X.shape[1], Y.shape[1]))

    if sp.issparse(X) != sp.issparse(Y):
        raise ValueError("X and Y must either both be sparse or dense. Mixed types are not allowed.")

    is_sparse = sp.issparse(X)

    if not shallow:
        if datatype == "binary":
            if is_sparse:
                val_X = np.unique(X.data)
                val_Y = np.unique(Y.data)
            else:
                val_X = np.unique(X)
                val_Y = np.unique(Y)

            if not np.all(np.in1d(val_X, [0, 1])) or not np.all(np.in1d(val_Y, [0, 1])):
                raise ValueError("Input data must be binary.")
        elif datatype == "positive":
            if is_sparse:
                all_pos_X = (X.data >= 0).all()
                all_pos_Y = (Y.data >= 0).all()
            else:
                all_pos_X = (X >= 0).all()
                all_pos_Y = (Y >= 0).all()

            if not all_pos_X and not all_pos_Y:
                raise ValueError("Input data must be positive.")
        elif datatype == "real":
            pass

    return X, Y, is_sparse


def minmax_kernel(X, Y=None, shallow_input_check=False):
    """
    Calculates the minmax kernel value for two sets of examples
    represented by their feature vectors.

    :param X: array-like, shape = (n_samples_A, n_features), examples A
    :param Y: array-like, shape = (n_samples_B, n_features), examples B

    :return: array-like, shape = (n_samples_A, n_samples_B), kernel matrix
             with minmax kernel values:

                K[i,j] = k_mm(A_i, B_j)

    :source: https://github.com/gmum/pykernels/blob/master/pykernels/regular.py
    """
    X, Y, is_sparse = check_input(X, Y, datatype="positive", shallow=shallow_input_check)  # Handle for example Y = None

    n_A, n_B = X.shape[0], Y.shape[0]

    min_K = np.zeros((n_A, n_B))
    max_K = np.zeros((n_A, n_B))

    if not is_sparse:
        # Dense Matrix Implementation

        # TODO: Check loop-less version for speed and memory consumption.
        # X = X[:, None, :]
        # Y = Y[None, :, :]
        # min_K = np.sum(np.minimum(X, Y), axis=2)
        # max_K = np.sum(np.maximum(X, Y), axis=2)

        for s in range(X.shape[1]):  # loop if the feature dimensions
            c_s_A = X[:, s].reshape(-1, 1)
            c_s_B = Y[:, s].reshape(-1, 1)

            # Check for empty features dimension
            if np.all(c_s_A == 0) and np.all(c_s_B == 0):
                continue

            min_K += np.minimum(c_s_A, c_s_B.T)
            max_K += np.maximum(c_s_A, c_s_B.T)
    else:
        # Sparse Matrix Implementation
        for s in range(X.shape[1]):  # loop if the feature dimensions
            c_s_A = X[:, s]
            c_s_B = Y[:, s]

            # Check for empty feature dimension
            if len(c_s_A.indices) == 0 and len(c_s_B.indices) == 0:
                continue

            # Update maximums
            maxs = np.maximum(c_s_A.toarray().reshape(-1, 1), c_s_B.toarray().reshape(1, -1))
            max_K += maxs

            # Get indices to update minimums
            idc = tuple(zip(*it.product(c_s_A.indices, c_s_B.indices)))
            if len(idc) > 0:
                mins = np.minimum(c_s_A.data.reshape(-1, 1), c_s_B.data.reshape(1, -1))
                min_K[idc] = min_K[idc] + np.ravel(mins)

    K_mm = min_K / max_K

    return K_mm


def tanimoto_kernel(X, Y=None, shallow_input_check=False):
    """
    Tanimoto kernel function

    :param X: array-like, shape=(n_samples_A, n_features), binary feature matrix of set A
    :param Y: array-like, shape=(n_samples_B, n_features), binary feature matrix of set B
        or None, than Y = X

    :return array-like, shape=(n_samples_A, n_samples_B), tanimoto kernel matrix
    """
    X, Y, is_sparse = check_input(X, Y, datatype="binary", shallow=shallow_input_check)

    XY = X @ Y.T
    XX = X.sum(axis=1).reshape(-1, 1)
    YY = Y.sum(axis=1).reshape(-1, 1)

    K_tan = XY / (XX + YY.T - XY)

    assert (not sp.issparse(K_tan)), "Kernel matrix should not be sparse."

    return K_tan


def generalized_tanimoto_kernel(X, Y=None, shallow_input_check=False):
    """
    Generalized tanimoto kernel function

    :param X:
    :param Y:
    :return:
    """
    X, Y, is_sparse = check_input(X, Y, datatype="real", shallow=shallow_input_check)

    if is_sparse:
        raise NotImplementedError("Sparse matrices not supported.")

    XL1 = np.sum(np.abs(X), axis=1)[:, np.newaxis]
    YL1 = np.sum(np.abs(Y), axis=1)[:, np.newaxis]

    XmYL1 = manhattan_distances(X, Y)

    K_gtan = (XL1 + YL1.T - XmYL1) / (XL1 + YL1.T + XmYL1)

    return K_gtan


def get_centering_matrix(n):
    """
    Function to return a centering matrix (operator):

        C = I - (11^T) / n

    :param n: scalar, number of samples
    :return: array-like, shape=(n, n), centering matrix
    """
    return np.identity(n) - np.full((n, n), fill_value=(1. / n))
