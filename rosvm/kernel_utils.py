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
import scipy.sparse as sp
import itertools as it

from sklearn.metrics.pairwise import manhattan_distances
from joblib import delayed, Parallel

"""
Kernel functions here are optimized to work on matrix inputs. 
"""


def check_input(X, Y, datatype=None, shallow=False):
    """
    Function to check whether the two input sets A and B are compatible.

    :param X: array-like, shape = (n_samples_A, n_features), examples A
    :param Y: array-like, shape = (n_samples_B, n_features), examples B
    :param datatype: string, used to specify constraints on the input data (type)
    :param shallow: boolean, indicating whether checks regarding features values, e.g. >= 0, should be skipped.

    :return: X, Y, is_sparse. X is simply passed through. If Y is None, than it
        will be equal X otherwise it is also just passed through. is_sparse is
        a boolean indicating whether X and Y are sparse matrices
    """
    if Y is None:
        Y = X

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Number of features for set A and B must match: %d vs %d." % (
            X.shape[1], Y.shape[1]))

    if isinstance(X, np.ndarray):
        if not isinstance(Y, np.ndarray):
            raise ValueError("Input matrices must be of same type.")
        is_sparse = False
    elif isinstance(X, sp.csr_matrix):
        if not isinstance(Y, sp.csr_matrix):
            raise ValueError("Input matrices must be of same type.")
        is_sparse = True
    else:
        raise ValueError("Input matrices only allowed to be of type 'np.ndarray' or 'scipy.sparse.csr_matrix'.")

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
                any_neg_X = (X.data < 0).any()
                any_neg_Y = (Y.data < 0).any()
            else:
                any_neg_X = (X < 0).any()
                any_neg_Y = (Y < 0).any()

            if any_neg_X or any_neg_Y:
                raise ValueError("Input data must be positive.")
        elif datatype == "real":
            pass

    return X, Y, is_sparse


def minmax_kernel(X, Y=None, shallow_input_check=False, n_jobs=4):
    """
    Calculates the minmax kernel value for two sets of examples
    represented by their feature vectors.

    :param X: array-like, shape = (n_samples_A, n_features), examples A
    :param Y: array-like, shape = (n_samples_B, n_features), examples B
    :param shallow_input_check: boolean, indicating whether checks regarding features values, e.g. >= 0, should be
        skipped.
    :param n_jobs: scalar, number of jobs used for the kernel calculation from sparse input

    :return: array-like, shape = (n_samples_A, n_samples_B), kernel matrix
             with minmax kernel values:

                K[i,j] = k_mm(A_i, B_j)

    :source: https://github.com/gmum/pykernels/blob/master/pykernels/regular.py
    """
    X, Y, is_sparse = check_input(X, Y, datatype="positive", shallow=shallow_input_check)  # Handle for example Y = None

    if is_sparse:
        K_mm = _min_max_sparse_csr(X, Y, n_jobs=n_jobs)
    else:
        K_mm = _min_max_dense(X, Y)

    return K_mm


def _min_max_dense(X, Y):
    """
    MinMax-Kernel implementation for dense feature vectors.
    """
    n_A, n_B = X.shape[0], Y.shape[0]

    min_K = np.zeros((n_A, n_B))
    max_K = np.zeros((n_A, n_B))

    for s in range(X.shape[1]):  # loop if the feature dimensions
        c_s_A = X[:, s].reshape(-1, 1)
        c_s_B = Y[:, s].reshape(-1, 1)

        # Check for empty features dimension
        if np.all(c_s_A == 0) and np.all(c_s_B == 0):
            continue

        min_K += np.minimum(c_s_A, c_s_B.T)
        max_K += np.maximum(c_s_A, c_s_B.T)

    return min_K / max_K


@delayed
def _min_max_sparse_csr_single_element(x_i, y_j, nonz_idc_x_i, nonz_idc_y_j):
    min_k = 0
    max_k = 0

    # In the indices intersection we need to check min and max
    for s in nonz_idc_x_i & nonz_idc_y_j:
        max_k += np.maximum(x_i[0, s], y_j[0, s])
        min_k += np.minimum(x_i[0, s], y_j[0, s])

    # Indices that appear only in X[i]: minimum is zero, maximum comes from X[i]
    for s in nonz_idc_x_i - nonz_idc_y_j:
        max_k += x_i[0, s]

    # Indices that appear only in Y[j]: minimum is zero, maximum comes from Y[j]
    for s in nonz_idc_y_j - nonz_idc_x_i:
        max_k += y_j[0, s]

    return np.sum(min_k), np.sum(max_k)


def _min_max_sparse_csr(X, Y, n_jobs=1):
    """
    MinMax-Kernel implementation for sparse feature vectors.
    """
    # Find the non-zero indices for each row and put them into set-objects
    n_x, n_y = X.shape[0], Y.shape[0]
    nonz_idc_x = [set() for _ in range(n_x)]
    nonz_idc_y = [set() for _ in range(n_y)]

    for i in range(n_x):
        nonz_idc_x[i].update(X.indices[X.indptr[i]:X.indptr[i + 1]])  # non-zero indices of matrix X in row

    for i in range(n_y):
        nonz_idc_y[i].update(Y.indices[Y.indptr[i]:Y.indptr[i + 1]])  # non-zero indices of matrix X in row

    # Calculate kernel values
    res = Parallel(n_jobs=n_jobs)(_min_max_sparse_csr_single_element(X[i], Y[j], nonz_idc_x[i], nonz_idc_y[j])
                                  for i, j in it.product(range(n_x), range(n_y)))

    min_k, max_k = zip(*res)
    min_k = np.array(min_k).reshape((n_x, n_y))
    max_k = np.array(max_k).reshape((n_x, n_y))

    return min_k / max_k


def tanimoto_kernel(X, Y=None, shallow_input_check=False):
    """
    Tanimoto kernel function

    :param X: array-like, shape=(n_samples_A, n_features), binary feature matrix of set A
    :param Y: array-like, shape=(n_samples_B, n_features), binary feature matrix of set B
        or None, than Y = X
    :param shallow_input_check: boolean, indicating whether checks regarding features values, e.g. >= 0, should be
        skipped.

    :return array-like, shape=(n_samples_A, n_samples_B), tanimoto kernel matrix
    """
    X, Y, is_sparse = check_input(X, Y, datatype="binary", shallow=shallow_input_check)

    if is_sparse:
        raise NotImplementedError("Tanimoto: Sparse matrices not supported.")

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


if __name__ == "__main__":
    import timeit
    from featurizer_cls import CircularFPFeaturizer

    # Performance evaluation of MinMax kernel calculation for sparse matrices returned by the featurizer
    smis = ["CC(=O)C1=CC2=C(OC(C)(C)[C@@H](O)[C@@H]2O)C=C1",
            "C1COC2=CC=CC=C2C1",
            "O=C(CCc1ccc(O)cc1)c1c(O)cc(O)c(C2OC(CO)C(O)C(O)C2O)c1O",
            "O=c1c(OC2OC(CO)C(O)C(O)C2O)c(-c2ccc(OC3OC(CO)C(O)C(O)C3O)c(O)c2)oc2cc(O)cc(O)c12",
            "O=C(O)C1OC(Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)C(O)C(O)C1O",
            "Oc1cc(O)c2c(c1)OC1(c3ccc(O)c(O)c3)Oc3cc(O)c4c(c3C2C1O)OC(c1ccc(O)c(O)c1)C(O)C4",
            "COc1cc(O)c2c(=O)c(O)c(-c3ccc(O)c(O)c3)oc2c1",
            "CC1OC(O)C(O)C(O)C1O",
            "Cc1cc2nc3c(O)nc(=O)nc-3n(CC(O)C(O)C(O)CO)c2cc1C",
            "O=C(C=Cc1ccc(O)c(O)c1)OC(Cc1ccc(O)c(O)c1)C(=O)O",
            "COc1cc(O)c2c(c1)OC(c1ccc(O)cc1)CC2=O",
            "C=CC(C)(O)CCC1C(C)(O)CCC2C(C)(C)CCCC21C",
            "COc1cc2ccc(=O)oc2cc1O",
            "NCCc1c[nH]c2ccc(O)cc12",
            "COc1cc(C=NN=Cc2cc(OC)c(O)c(OC)c2)cc(OC)c1O",
            "COc1cc(C=O)cc(OC)c1O",
            "COc1cc(-c2oc3cc(O)cc(O)c3c(=O)c2O)cc(OC)c1O",
            "CC(CCC(O)=NCCS(=O)(=O)O)C1CCC2C3C(O)CC4CC(O)CCC4(C)C3CC(O)C12C",
            "CC(C)(C)c1cc(O)ccc1O",
            "Cn1cnc2c1c(O)nc(=O)n2C",
            "Cn1c(=O)c2nc[nH]c2n(C)c1=O",
            "CC1=C(CCO)S[CH]N1Cc1cnc(C)[nH]c1=N",
            "O=C(O)C(O)C(O)CO",
            "CC1(O)CCC(C(C)(C)O)CC1",
            "C[n+]1cccc(C(=O)[O-])c1",
            "OCCc1c[nH]c2ccccc12",
            "NCCc1ccc(O)cc1",
            "OCCc1ccc(O)cc1",
            "O=c1ccc2ccc(O)cc2o1",
            "Oc1ccnc(O)n1",
            "CC1CCC2(C(=O)O)CCC3(C)C(=CCC4C5(C)CCC(O)C(C)(C)C5CCC43C)C2C1C",
            "COc1cc(C(=O)O)ccc1O",
            "COc1cc(C=O)ccc1O",
            "CC(=CCC1=C(C)C(=O)c2ccccc2C1=O)CCCC(C)CCCC(C)CCCC(C)C",
            "Oc1nc(O)c2nc[nH]c2n1",
            "OC1COC(O)C(O)C1O",
            "OCC(O)C(O)CO",
            "O=Cc1ccc(O)c(O)c1",
            "O=C(O)CO",
            "O=CC(=O)O",
            "CCCCCCCCCCCCCCCCCCCCCCCC(=O)O",
            "O=C(O)C(=O)O",
            "OCC1OC(Oc2ccccc2)C(O)C(O)C1O",
            "CC(CCC(O)=NCCOS(=O)(=O)O)C1CCC2C3C(O)CC4CC(O)CCC4(C)C3CCC12C",
            "Oc1cc(O)c2c(c1)OC(c1ccc(O)c(O)c1)C(O)C2",
            "O=C(OC1Cc2c(O)cc(O)cc2OC1c1ccc(O)c(O)c1)c1cc(O)c(O)c(O)c1",
            "Oc1cc(O)c2c(c1)OC(c1cc(O)c(O)c(O)c1)C(O)C2",
            "O=C(O)c1cc(O)cc(O)c1",
            "O=C(O)c1ccc(O)cc1",
            "OCC1OC(Oc2cc(O)cc(C=Cc3ccc(O)c(O)c3)c2)C(O)C(O)C1O",
            "O=C(C=Cc1ccc(O)c(O)c1)OC(C(=O)O)C(O)C(=O)O",
            "OCC1OC(Oc2cc(O)cc(C=Cc3ccc(O)cc3)c2)C(O)C(O)C1O",
            "Oc1ccc(C=Cc2cc(O)cc(O)c2)cc1",
            "O=c1oc2c(O)c(O)cc3c(=O)oc4c(O)c(O)cc1c4c23",
            "Oc1ccc(C2Oc3cc(O)cc(O)c3CC2O)cc1",
            "O=c1ccc2cc(O)c(O)cc2o1",
            "COc1cc(C=CC(=O)OC(C(=O)O)C(O)C(=O)O)ccc1O",
            "COc1cc2ccc(=O)oc2c(OC2OC(CO)C(O)C(O)C2O)c1O",
            "COc1ccc(C2CC(=O)c3c(O)cc(O)cc3O2)cc1O",
            "COc1cc(-c2oc3cc(O)cc(O)c3c(=O)c2OC2OC(CO)C(O)C(O)C2O)ccc1O",
            "COc1cc(-c2oc3cc(O)cc(O)c3c(=O)c2OC2OC(COC3OC(C)C(O)C(O)C3O)C(O)C(O)C2O)ccc1O",
            "COc1cc(C=Cc2cc(O)cc(OC3OC(CO)C(O)C(O)C3O)c2)ccc1O",
            "O=c1c(OC2OC(CO)C(O)C(O)C2O)c(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",
            "CC1OC(OCC2OC(Oc3c(-c4ccc(O)cc4)oc4cc(O)cc(O)c4c3=O)C(O)C(O)C2O)C(O)C(O)C1O",
            "O=c1cc(-c2ccc(O)c(O)c2)oc2cc(OC3OC(CO)C(O)C(O)C3O)cc(O)c12",
            "COC(=O)c1cc(O)c(O)c(O)c1",
            "O=c1c(O)c(-c2cc(O)c(O)c(O)c2)oc2cc(O)cc(O)c12",
            "O=C1CC(c2ccc(O)cc2)Oc2cc(O)cc(O)c21",
            "O=C1CC(c2ccc(O)c(O)c2)c2c(cc(O)c3c2OC(c2ccc(O)c(O)c2)C(O)C3)O1",
            "Oc1ccc(C=Cc2cc(O)cc3c2C(c2cc(O)cc(O)c2)C(c2ccc(O)cc2)O3)cc1",
            "O=C1CC(c2ccc(O)cc2)Oc2cc(OC3OC(CO)C(O)C(O)C3O)cc(O)c21",
            "O=C(O)C=Cc1ccccc1O",
            "COc1cc(-c2oc3cc(O)cc4oc(=O)cc(c2OC2OC(CO)C(O)C(O)C2O)c34)cc(OC)c1O",
            "Oc1ccc(C2c3c(O)cc(O)cc3C3C(c4ccc(O)cc4)c4c(O)cc(O)cc4C23)cc1",
            "O=C(CCc1ccc(O)cc1)c1c(O)cc(O)cc1OC1OC(CO)C(O)C(O)C1O",
            "Oc1cc(O)cc(C=Cc2ccc(O)c(O)c2)c1",
            "Oc1cc(O)c2c(c1)OC(c1ccc(O)c(O)c1)C(O)C2c1c(O)cc(O)c2c1OC(c1ccc(O)c(O)c1)C(O)C2"]

    # Get kernel matrix from fingerprints without substructure learning
    fps_mat = CircularFPFeaturizer(fp_mode="count").transform(smis)
    print("Is instance of 'csr_matrix': %d" % sp.isspmatrix_csr(fps_mat))
    print(fps_mat.shape)
    times = timeit.repeat(lambda: _min_max_sparse_csr(fps_mat, fps_mat, n_jobs=4), number=1, repeat=3)
    print("min time:", np.min(times))

    # Now with substructure learning
    fps_mat = CircularFPFeaturizer(fp_mode="count", only_freq_subs=True, output_dense_matrix=True).fit_transform(smis)
    print("Is instance of 'csr_matrix': %d" % sp.isspmatrix_csr(fps_mat))
    print(fps_mat.shape)
    times = timeit.repeat(lambda: _min_max_dense(fps_mat, fps_mat), number=1, repeat=3)
    print("min time:", np.min(times))
