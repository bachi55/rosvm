####
#
# The MIT License (MIT)
#
# Copyright 2017-2020 Eric Bach <eric.bach@aalto.fi>
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
import scipy.sparse as sp_sparse
import itertools

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_random_state
from collections.abc import Sequence

from pair_utils_2 import get_pairs_multiple_datasets
from kernel_utils import tanimoto_kernel, minmax_kernel


class Labels(Sequence):
    """
    Class to a list of (RT, dataset) label tuples compatible with the RankSVM.
    """
    def __init__(self, rts, dss):
        """
        :param rts: list of scalars, retention times
        :param dss: list of identifiers, dataset identifier, can be strings, integers
        """
        self._rts = rts
        self._dss = dss
        if len(self._rts) != len(self._dss):
            raise ValueError("Number of retention times must be equal the number of the dataset identifiers.")

        self.shape = (len(self._rts),)  # needed for scikit-learn input checks

    def __getitem__(self, item):
        """
        Access the labels. If an integer is provided, the label tuple at the specified index is returned.
        If a slice or an integer list is provided, a new Labels object containing the requested sub-set is
        created and returned.
        """
        if isinstance(item, int) or isinstance(item, np.int64):
            return self._rts[item], self._dss[item]
        elif isinstance(item, slice):
            return Labels(self._rts[item], self._dss[item])
        elif isinstance(item, list):
            if len(item) == 0:
                return Labels([], [])

            if isinstance(item[0], bool):
                # Boolean indexing
                return Labels([self._rts[i] for i, b in enumerate(item) if b],
                              [self._dss[i] for i, b in enumerate(item) if b])
            elif isinstance(item[0], int):
                # Integer indexing
                return Labels([self._rts[i] for i in item], [self._dss[i] for i in item])
            else:
                raise NotImplementedError("For lists only int and bool are allowed as index type.")
        elif isinstance(item, np.ndarray):
            return self.__getitem__(item.tolist())
        else:
            raise TypeError("Label indices must integers, slices, list or numpy.ndarray, not %s." % type(item))

    def __len__(self):
        """
        Number of label tuples.
        """
        return len(self._rts)

    def __iter__(self):
        return zip(self._rts, self._dss)

    def get_rts(self):
        return self._rts

    def get_dss(self):
        return self._dss

    def get_data(self):
        return self.get_rts(), self.get_dss()

    def __eq__(self, other):
        """
        Two Labels objects are equal, if the retention time and dataset identifier lists are equal.
        """
        if len(self) != len(other):
            return False

        if self._rts != other.get_rts():
            return False

        if self._dss != other.get_dss():
            return False

        return True

    def __str__(self):
        return self.__dict__.__str__()

    def __repr__(self):
        return self.__str__()


class FeasibilityError(RuntimeError):
    """
    Error throw if the dual variables are not in the feasible domain.
    """
    def __init__(self, msg):
        super(FeasibilityError, self).__init__(msg)


class KernelRankSVC (BaseEstimator, ClassifierMixin):
    """
    Implementation of the kernelized Ranking Support Vector Classifier.

    The optimization is performed in the dual-space using the conditional gradient (a.k.a. Frank-Wolfe
    algorithm[1,2]). See also the paper for details on the optimization problem.

    [1] An algorithm for quadratic programming, Frank M. and Wolfe P, Navel Research Logistic Quarterly banner,
        1956
    [2] Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization, Jaggi M., Proceedings of the 30th
        International Conference on Machine Learning, 2013

    :param C: scalar, regularization parameter of the SVM (default = 1.0)

    :param kernel: string or callable, determining the kernel used for the tutorial. Logic of this function
        (default = "precomputed")
        - "precomputed": The kernel matrix is precomputed and provided to the fit and prediction function.
        - ["rbf", "polynomial", "linear"]: The kernel is computed by the scikit-learn build it functions
        - callable: The kernel is computed using the provided callable function, e.g., to get the tanimoto
            kernel.

    :param tol: scalar, tolerance of the change of the alpha vector. (default = 0.001)
        E.g. if "convergence_criteria" == "alpha_change_max" than
         |max (alpha_old - alpha_new)| < tol ==> convergence.

    :param max_iter: scalar, maximum number of iterations (default = 1000)

    :param min_iter: scalar, minimum number of iterations (default = 5)

    :param t_0: scalar, initial step-size (default = 0.1)

    :param step_size_algorithm: string, which step-size calculation method should be used.
        (default = "diminishing")
        - "diminishing": iterative decreasing stepsize
        - "diminishing_2": iterative decreasing stepsize, another formula
        - "fixed": fixed step size t_0

        NOTE: Check corresponding functions "_get_step_size_*" for implementation.

    :param gamma: scalar, scaling factor of the gaussian and polynomial kernel. If None, than it will
        be set to 1 / #features.

    :param coef0: scalar, parameter for the polynomial kernel

    :param degree: scalar, degree of the polynomial kernel

    :param kernel_params: dictionary, parameters that are passed to the kernel function. Can be used to
        input own kernels.

    :param convergence_criteria: string, how the convergence of the gradient descent should be determined.
        (default = "alpha_change_max")
        - "alpha_change_max": maximum change of the dual variable
        - "gs_change": change of the dual objective
        - "alpha_change_norm": change of the norm of the dual variables
        - "max_iter": stop after maximum number of iterations has been reached

    :param verbose: boolean, should the optimization be verbose (default = False)

    :param debug: scalar, debug level, e.g., calculation duality gap. This increases the complexity.
        (default = 0)

    :param random_state: integer, used as seed for the random generator. The randomness would
        effect the labels of the training pairs. Check the 'fit' and 'get_pairwise_labels' functions
        for details.

    Kernels:
    --------
    "linear": K(X, Y) = <X, Y>
    "polynomial": K(X, Y) = (gamma <X, Y> + coef0)^degree
    "rbf": K(x, y) = exp(- gamma ||x - y||^2)

    SOURCE: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise
    """
    def __init__(self, C=1.0, kernel="precomputed", max_iter=1000, gamma=None, coef0=1, degree=3, kernel_params=None,
                 random_state=None, pair_generation="eccb", alpha_threshold=1e-4):

        # Parameter for the optimization
        self.max_iter = max_iter

        # Kernel parameters
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params

        # General Ranking SVM parameter
        self.C = C
        self.alpha_threshold = alpha_threshold

        # Debug parameters
        self.random_state = random_state

        # Training tutorial used for fitting
        self.pair_generation = pair_generation

        # Model parameters
        #   self.pairs_train_ = None
        #   self.X_train_ = None
        #   self.A_ = None
        #   self.last_AKAt_y_ = None
        #   self.KX_train_ = None
        #   self.py_train_ = None
        #   self.pdss_train_ = None
        #   self.alpha_ = None
        #   self.is_sv_ = None

    def fit(self, X, y):
        """
        Estimating the parameters of the dual ranking svm with scaled margin.
        The conditional gradient descent algorithm is used to find the optimal
        alpha vector.

        :param X: array-like, shape = (n_samples, n_features) or (n_samples, n_samples)
            Object features or object similarities (kernel). If self.kernel == "precomputed"
            then X is interpreted as symmetric kernel matrix, otherwise as feature matrix.
            In this case the kernel is calculated on the fly.

        :param y: list of tuples, length = n_samples, the targets values, e.g. retention times,
            for all molecules measured with a set of datasets.

            Example:
            [..., (rts_i, ds_i), (rts_j, ds_j), (rts_k, ds_k), ...]

            rts_i ... retention time of measurement i
            ds_i ... identifier of the dataset of measurement i
        """
        rs = check_random_state(self.random_state)

        # Handle training tutorial
        if self.kernel == "precomputed":
            if X.shape[0] != X.shape[1]:
                raise ValueError("Precomputed kernel matrix must be squared: You provided KX.shape = (%d, %d)."
                                 % (X.shape[0], X.shape[1]))
            self.KX_train_ = X
        else:
            self.X_train_ = X
            self.KX_train_ = self._get_kernel(self.X_train_)

        # Generate training pairs
        if self.pair_generation == "eccb":
            pair_params = {"d_upper": 16, "d_lower": 1}
            select_random_pairs = False
        elif self.pair_generation == "all":
            pair_params = {"d_upper": np.inf, "d_lower": 1}
            select_random_pairs = False
        elif self.pair_generation == "random":
            pair_params = {"d_upper": np.inf, "d_lower": 1}
            select_random_pairs = True
        else:
            raise ValueError("Invalid pair generation approach: %s. Choices are: 'eccb', 'all' and 'random'."
                             % self.pair_generation)

        self.pairs_train_, self.py_train_, self.pdss_train_ = get_pairs_multiple_datasets(
            y, d_lower=pair_params["d_lower"], d_upper=pair_params["d_upper"])

        if select_random_pairs:
            _idc = rs.choice(
                range(len(self.pairs_train_)),
                size=self._get_p_perc(len(self.pairs_train_), 5),
                replace=False)
            self.pairs_train_ = [self.pairs_train_[idx] for idx in _idc]
            self.py_train_ = [self.py_train_[idx] for idx in _idc]
            self.pdss_train_ = [self.pdss_train_[idx] for idx in _idc]

        # Build A matrix
        self.A_ = self._build_A_matrix(self.pairs_train_, self.py_train_, len(X))

        # Initialize alpha: all dual variables are zero
        alpha = np.full(len(self.pairs_train_), fill_value=0)  # shape = (n_pairs_train, )

        k = 0
        while k < self.max_iter:
            s = self._solve_sub_problem(alpha)

            tau = 2 / (k + 2)  # step-width

            alpha = alpha + tau * (s - alpha)

            self._assert_is_feasible(alpha)

            k += 1

        # Threshold dual variables to the boarder ranges, if there are very close to it.
        alpha = self._bound_alpha(alpha, self.alpha_threshold, 0, self.C)
        self._assert_is_feasible(alpha)

        # Only store tutorial related to the support vectors
        self.is_sv_ = (alpha > 0)
        self.A_ = self.A_[self.is_sv_]
        self.alpha_ = alpha[self.is_sv_]
        self.pairs_train_ = [self.pairs_train_[idx] for idx, is_sv in enumerate(self.is_sv_) if is_sv]
        self.py_train_ = [self.py_train_[idx] for idx, is_sv in enumerate(self.is_sv_) if is_sv]
        self.pdss_train_ = [self.pdss_train_[idx] for idx, is_sv in enumerate(self.is_sv_) if is_sv]

        # print("n_support: %d (out of %d)" % (np.sum(self.is_sv_).item(), len(self.is_sv_)))

        return self

    def predict(self, X):
        """
        Calculates the RankSVM preference score < w , phi_i > for a set of examples.

        Note: < w , phi_i - phi_j > = < w , phi_i > - < w , phi_j >

        :param X: array-like, tutorial description
            feature-vectors: shape = (n_samples_test, d)
            -- or --
            kernel-matrix: shape = (n_samples_test, n_samples_train),

        :return: array-like, shape = (n_samples_test, ), mapped values for all examples.
        """
        if self.kernel == "precomputed":
            if not X.shape[0] == self.KX_train_.shape[0]:
                raise ValueError("Test-train kernel must have as many columns as training examples.")
        else:
            X = self._get_kernel(X, self.X_train_)  # shape = (n_test, n_train)

        wtx = (X @ (self.A_.T @ self.alpha_)).flatten()  # shape = (n_test, )
        assert wtx.shape == (len(X),)

        return wtx

    def score(self, X, y, sample_weight=None, return_detailed_results=False):
        """
        :param X: array-like, tutorial description
            feature-vectors: shape = (n_samples_test, d)
            -- or --
            kernel-matrix: shape = (n_samples_test, n_samples_train),

        :param y: list of tuples, length = n_samples, the targets values, e.g. retention times,
            for all molecules measured with a set of datasets.

            Example:
            [..., (rts_i, ds_i), (rts_j, ds_j), (rts_k, ds_k), ...]

            rts_i ... retention time of measurement i
            ds_i ... identifier of the dataset of measurement i

        :param sample_weight: parameter currently not used.

        :param return_detailed_results: boolean, indicating whether instead of an average score across
            all datasets separate scores are returned.

        :return:
            scalar, pairwise prediction accuracy separately calculated for all provided
                datasets and subsequently averaged.

            or

            dictionary, pairwise prediction accuracy separately calculated for each dataset with additional
                information about the number of test samples per set.
                keys: string identifying the dataset
                values: list, [score, n_test_samples, n_test_pairs]
        """
        if sample_weight:
            return NotImplementedError("Samples weights in the scoring are currently not supported.")

        rts, dss = zip(*y)
        rts = np.array(rts)
        dss = np.array(dss)

        scores = {}
        for ds in set(dss):  # get unique datasets
            # Calculate the score for each dataset individually
            scr, ntp = self.score_pointwise_using_predictions(rts[dss == ds], self.predict(X[dss == ds]))
            scores[ds] = [scr, np.sum(dss == ds).item(), ntp]

        if return_detailed_results:
            out = scores
        else:
            out = 0.0
            for val in scores.values():
                out += val[0]
            out /= len(scores)

        return out

    def _get_kernel(self, X, Y=None, n_jobs=1):
        """
        Calculate kernel matrix for given sets of features.

        :param X: array-like, shape = (n_samples_a, n_features)

        :param Y: array-like, shape = (n_samples_b, n_features), default=None

        :param n_jobs: integer, number of jobs passed to 'pairwise_kernels', default=1

        :return: array-like, Kernel matrix between feature sets A and A or A and B, shape:
            (n_samples_a, n_samples_a) if     Y is None
            (n_samples_a, n_samples_b) if not Y is None
        """
        # Set up kernel parameters
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}

        # Calculate the kernel
        if self.kernel == "tanimoto":
            K_XY = tanimoto_kernel(X, Y)
        elif self.kernel == "minmax":
            K_XY = minmax_kernel(X, Y)
        else:
            K_XY = pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, n_jobs=n_jobs, **params)

        return K_XY

    def predict_pairwise(self, X1, X2=None, return_label=True):
        """
        Predict for each example u in {1,...,n} (set 1) whether it elutes before each example v in {1,...,m} (set 2).

        :param X1: array-like, object description
            feature-vectors: shape = (n_samples_set1, d)
            -- or --
            kernel-matrix: shape = (n_samples_train, n_samples_set1),

        :param X2: array-like, object description, default=None
            feature-vectors: shape = (n_samples_set2, d)
            -- or --
            kernel-matrix: shape = (n_samples_train, n_samples_set2)

        :param return_label: boolean, indicating whether the label {-1, 1} should be returned, default=True

        :return: array-like, shape = (n_samples_set1, n_samples_set2). Entry at index [u, v] is either:
             1: u (set 1 example) elutes before v (set 2 example) <==> wtx(v) > wtx(u) <==> sign(wtx(v) - wtx(u)) > 0
            -1: otherwise (v elutes before u)
             0: predicted target values for u and v are equal: wtx(v) == wtx(u)
        """
        wtX1 = self.predict(X1)

        if X2 is None:
            wtX2 = wtX1
        else:
            wtX2 = self.predict(X2)

        # Calculate pairwise predictions
        Y = - wtX1[:, np.newaxis] + wtX2[np.newaxis, :]
        if return_label:
            Y = np.sign(Y)  # -sign(x) = sign(-x), need to flip sign as we have (i, j) => j - i > 0

        return Y

    def _evaluate_primal_objective(self, alpha):
        """
        Get the function value of f_0:

            f_0(w(alpha), xi(alpha)) = 0.5 * || w(alpha) || + C sum_ij xi_ij(alpha)

        :param alpha: array-like, shape = (p, 1), current alpha estimate.

        :return: scalar, f_0(w(alpha), xi(alpha))
        """
        wtw = alpha @ self.last_AKAt_y_

        # See Juho's lecture 4 slides: "Convex optimization methods", slide 38
        xi = np.maximum(0, 1 - self.last_AKAt_y_)

        return wtw / 2 + self.C * np.sum(xi)

    def _evaluate_dual_objective(self, alpha):
        """
        Get the function of g:
            g(alpha, r) = sum_ij alpha_ij - 0.5 * alpha.T @ H alpha

        :param alpha: array-like, shape = (p, 1), current alpha estimate.

        :return: scalar, g(alpha, r)
        """
        return np.sum(alpha) - (alpha @ self.last_AKAt_y_) / 2.0

    def _assert_is_feasible(self, alpha):
        """
        Check whether the provided alpha vector is in the feasible set:

            1) alpha.shape == (|P|, 1)
            2) 0 <= alpha_ij <= (C / N) for all (i, j) in P

        :param alpha: array-like, shape = (p, 1), dual variables

        :return: Boolean:
            True,  if alpha is in the feasible set
            False, otherwise
        """
        n_pairs = len(self.pairs_train_)
        n_dual = alpha.shape[0]

        if np.any(alpha < 0):
            raise FeasibilityError("Some dual variables are not positive. Min alpha_i = %f" % np.min(alpha))

        if np.any(alpha > self.C):
            raise FeasibilityError("Some dual variables are larger C. Max alpha_i = %f" % np.max(alpha))

        if n_dual != n_pairs:
            raise FeasibilityError("Wrong number of dual variables.")

    def _solve_sub_problem(self, alpha):
        """
        Finding the feasible update direction.

        :return: array-like, shape = (p, 1), s
        """
        self.last_AKAt_y_ = self._x_AKAt_y(y=alpha)
        d = 1 - self.last_AKAt_y_

        s = np.zeros_like(alpha)
        s[d > 0] = self.C

        return s

    def _x_AKAt_y(self, x=None, y=None):
        """
        Function calculating:
            1) x^T \widetilde{A} K_phi \widetilde{A}^T y, if x and y are not None
            2) \widetilde{A} K_phi \widetilde{A}^T y,     if x is None and y is not None
            3) x^T\widetilde{A} K_phi \widetilde{A}^T,    if x is not None and y is None
            4) \widetilde{A} K_phi \widetilde{A}^T,       if x and y are None

        :param x: array-like, shape = (p, 1), vector on the left side

        :param y: array-like, shape = (p, 1), vector on the right side

        :return: array-like,
            1) scalar, if x and y are not None
            2) shape = (p, 1), if x is None and y is not None
            3) shape = (1, p), if x is not None and y is None
            4) shape = (p, p), if x and y are None
        """
        if (x is not None) and (y is not None):

            return (x.T @ (self.A_ @ (self.KX_train_ @ (self.A_.T @ y)))).item()

        elif (x is None) and (y is not None):

            return self.A_ @ (self.KX_train_ @ (self.A_.T @ y))

        elif (x is not None) and (y is None):

            return (self.A_ @ ((self.A_.T @ x).T @ self.KX_train_).T).T

        else:

            return self.A_ @ self.KX_train_ @ self.A_.T

    # ---------------------------------
    # Static methods
    # ---------------------------------
    @staticmethod
    def _bound_alpha(alpha, threshold, min_alpha_val, max_alpha_val):
        """
        Set dual variables very close to the minimum and maximum value, e.g. 0 and C,
        to these extreme points. During the optimization, the dual variables might not
        go exactly to the extreme values, but we can apply a threshold to them.


        """
        alpha = np.array(alpha)
        alpha[np.isclose(alpha, min_alpha_val, atol=threshold)] = min_alpha_val
        alpha[np.isclose(alpha, max_alpha_val, atol=threshold)] = max_alpha_val
        return alpha

    @staticmethod
    def _build_A_matrix(pairs, y, n_samples):
        """
        Construct a matrix A (p x l) so that:

            A_{(i, j), :} = y_ij * (0...0, 1, 0...0, -1, 0...0)       ROW IN MATRIX A
                                           i          j

        This matrix is used to simplify the optimization using 'difference' features.

        :param pairs: List of tuples, length = n_train_pairs (= p), containing the
            training pair indices. Here, i and j correspond to the index in the training
            feature or kernel matrix of the individual examples

        :param n_samples: scalar, number of training examples (= l)

        :param y: list, length = p. Label for each training example pair. The label is
            defined as:

                y_ij = sign(t_i - t_j)   for pair (i, j):

        :return: sparse array-like, shape = (p, l)
        """
        n_pairs = len(pairs)

        row_ind = np.append(np.arange(n_pairs), np.arange(n_pairs))
        col_ind = np.append([pair[0] for pair in pairs], [pair[1] for pair in pairs])
        data = np.append(np.ones(n_pairs), -1 * np.ones(n_pairs))

        # Apply labels to all rows
        data = data * np.append(y, y)

        return sp_sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n_pairs, n_samples))

    @staticmethod
    def _get_p_perc(n, p):
        """
        Return the number of samples corresponding to the specified percentage.

        :param n: scalar, number of samples
        :param p: scalar, percentage
        :return: scalar, fraction of the samples corresponding to the specified percentage
        """
        if p < 0 or p > 100:
            raise ValueError("Percentage must be from range [0, 100]")

        return np.round((n * p) / 100).astype("int")

    @staticmethod
    def _get_step_size_diminishing(k, C, t_0):
        """
        Calculate the step size using the diminishing strategy.

            step size = t_0 / (1 + t_0 * C * (k - 1))

            Border cases:
                k = 1, first iteration: step size = t_0
                t_0 = 1: step size = 1 / (1 + C * (k - 1))

        :param k: scalar, current iteration
        :param C: scalar, regularization parameter ranksvm
        :param t_0: scalar, initial step-size
        :return: scalar, step size

        Ref: [1] "Towards Optimal One Pass Large Scale Learning with Averaged Stochastic Gradient Descent",
                 Wei Xu, 2011, ArXiv
             [2] "Large-Scale Machine Learning with Stochastic Gradient Descent", Leo Bottou, 2010
        """
        return t_0 / (1 + t_0 * C * (k - 1))

    @staticmethod
    def _get_step_size_diminishing_2(t):
        """
        Step size after Sandor:

            t = t - (t**2 / 2)

        :param t: scalar, current step size
        :return: scalar, step size
        """
        return t - (t**2 / 2.0)

    @staticmethod
    def score_pointwise_using_predictions(y, y_pred, normalize=True):
        """
        :param y: array-like, shape = (n_samples_test,), true target values for the test samples.
        :param y_pred: array-like, shape = (n_samples_test,), predicted target values for the test samples.
        :param normalize: logical, indicating whether the number of correctly classified pairs
            should be normalized by the total number of pairs (n_pairs_test).

        :return: tuple(scalar, scalar), pairwise prediction accuracy and number of test pairs used
        """
        n_decisions = 0.0
        tp = 0.0

        for i, j in itertools.permutations(range(len(y_pred)), 2):
            if y[i] < y[j]:  # i is preferred over j
                n_decisions += 1
                if y_pred[j] == y_pred[i]:  # tie predicted
                    tp += 0.5
                elif y_pred[i] < y_pred[j]:  # i is _predicted_ to be preferred over j
                    tp += 1.0

        if n_decisions == 0.0:
            raise RuntimeError("Score is undefined of all true target values are equal.")

        if normalize:
            tp /= n_decisions

        return tp, n_decisions

    @staticmethod
    def score_pairwise_using_prediction(Y, pairs, normalize=True):
        """
        :param Y: array-like, shape = (n_samples_set1, n_samples_set2), pairwise object
            preference prediction

        :param pairs: list of index-tuples, shape = (n_pairs_test,), encoding the pairwise
            object relations:
                i is preferred over j, u is preferred over v, ... --> [(i,j), (u,v), ...]},

                where i, j, u, v, ... are the indices corresponding to the object descriptions
                given in X.

            In this implementation we assume that, e.g.,
                i is preferred over j <==> target[i] < target[j], ...

        :param normalize: logical, indicating whether the number of correctly classified pairs
            should be normalized by the total number of pairs (n_pairs_test).

        :return: scalar, pairwise prediction accuracy
        """
        if len(pairs) == 0:
            raise RuntimeError("Score is undefined of the number of test pairs is zero.")

        tp = 0.0
        for i, j in pairs:
            if Y[i, j] == 0 and i != j:
                # If target values are equal, we assume random performance.
                tp += 0.5
            else:
                # All (i, j) pairs are assumed to be satisfy: i is preferred over j <==> target[i] < target[j]
                tp += (Y[i, j] == 1)

        if normalize:
            tp /= len(pairs)

        return tp


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import GroupKFold, GridSearchCV

    # Load example tutorial
    data = pd.read_csv("ranksvm/tutorial/example_data.csv", sep="\t")
    X = np.array(list(map(lambda x: x.split(","), data.substructure_count.values)), dtype="float")
    y = Labels(data.rt.values, data.dataset.values)
    mol = data.smiles.values

    # Split into training and test
    train, test = next(GroupKFold(n_splits=3).split(X, y, groups=mol))
    print("(n_train, n_test) = (%d, %d)" % (len(train), len(test)))
    assert not (set(mol[train]) & set(mol[test]))

    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    mol_train = mol[train]

    n_jobs = 1
    assert n_jobs == 1, "Pickle does not work here. Run the tutorial."
    ranksvm = GridSearchCV(
        estimator=KernelRankSVC(kernel="minmax", pair_generation="random", random_state=2921, alpha_threshold=1e-2),
        param_grid={"C": [0.5, 1, 2, 4, 8]},
        cv=GroupKFold(n_splits=3),
        n_jobs=1).fit(X_train, y_train, groups=mol_train)
    print(ranksvm.cv_results_["mean_test_score"])

    # Inspect RankSVM prediction
    print("Score: %3f" % ranksvm.score(X_test, y_test))

    fig, axrr = plt.subplots(1, 2, figsize=(12, 6))
    dss_test = np.array(y_test.get_dss())
    rts_test = np.array(y_test.get_rts())
    axrr[0].scatter(rts_test[dss_test == "FEM_long"], ranksvm.predict(X_test[dss_test == "FEM_long"]))
    axrr[1].scatter(rts_test[dss_test == "UFZ_Phenomenex"], ranksvm.predict(X_test[dss_test == "UFZ_Phenomenex"]))
    plt.show()
