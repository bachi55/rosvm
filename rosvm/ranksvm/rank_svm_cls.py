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
from sklearn.model_selection import train_test_split
from collections.abc import Sequence
from typing import TypeVar

from rosvm.ranksvm.pair_utils import get_pairs_multiple_datasets
from rosvm.ranksvm.kernel_utils import tanimoto_kernel, minmax_kernel


RANKSVM_T = TypeVar('RANKSVM_T', bound='KernelRankSVC')


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
        self._unique_ds = set(self._dss)

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

    def get_unique_dss(self):
        return self._unique_ds

    def get_idc_for_ds(self, ds, on_missing_raise=True):
        if on_missing_raise and (ds not in set(self.get_unique_dss())):
            raise KeyError("No example in the label-set belongs to the dataset '%s'." % ds)

        return [i for i, _ds in enumerate(self.get_dss()) if _ds == ds]

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

    :param max_iter: scalar, maximum number of iterations (default = 1000)

    :param gamma: scalar, scaling factor of the gaussian and polynomial kernel. If None, than it will
        be set to 1 / #features.

    :param coef0: scalar, parameter for the polynomial kernel

    :param degree: scalar, degree of the polynomial kernel

    :param kernel_params: dictionary, parameters that are passed to the kernel function. Can be used to
        input own kernels.

    :param random_state: integer, used as seed for the random generator. The randomness would
        effect the labels of the training pairs. Check the 'fit' and 'get_pairwise_labels' functions
        for details.

    :param debug: boolean, indicating whether debug information should be stored in the RankSVM
        class. Those include:
            - Objective function values (primal, dual and duality gap)
            - Prediction accuracy on a validation set (sub-set of the provided training set) throughout the epochs
            - Step size

    Kernels:
    --------
    "linear": K(X, Y) = <X, Y>
    "polynomial": K(X, Y) = (gamma <X, Y> + coef0)^degree
    "rbf": K(x, y) = exp(- gamma ||x - y||^2)

    SOURCE: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise
    """
    def __init__(self, C=1.0, kernel="precomputed", max_iter=1000, gamma=None, coef0=1, degree=3, kernel_params=None,
                 random_state=None, pair_generation="eccb", alpha_threshold=1e-3, pairwise_features="difference",
                 debug=False, step_size="diminishing_1", tau_0=0.5, duality_gap_threshold=1e-3):

        # Parameter for the optimization
        self.max_iter = max_iter
        self.step_size = step_size
        if self.step_size not in ["diminishing_1", "diminishing_2", "diminishing_3", "linesearch"]:
            raise ValueError("Invalid step-size method: '%s'. Choices are 'diminishing_1', 'diminishing_2', "
                             "'diminishing_3' and 'linesearch'." % self.step_size)
        self.tau_0 = tau_0
        self.duality_gap_threshold = duality_gap_threshold

        # Kernel parameters
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params

        # General Ranking SVM parameter
        self.C = C
        self.alpha_threshold = alpha_threshold
        self.pairwise_features = pairwise_features
        if self.pairwise_features not in ["difference", "exterior_product"]:
            raise ValueError("Invalid pairwise feature: '%s'. Choices are 'difference' or 'exterior_product'"
                             % self.pairwise_features)
        self.pair_generation = pair_generation
        if self.pair_generation not in ["eccb", "all", "random"]:
            raise ValueError("Invalid pair generation approach: %s. Choices are: 'eccb', 'all' and 'random'."
                             % self.pair_generation)

        # Debug parameters
        self.random_state = random_state
        self.debug = debug

        # Model parameters
        #   self.pairs_train_ = None
        #   self.X_train_ = None
        #   self.A_ = None
        #   self.last_AKAt_y_ = None
        #   self.KX_train_ = None
        #   self.py_train_ = None
        #   self.pdss_train_ = None
        #   self.alpha_ = None

    def fit(self, X: np.ndarray, y: Labels) -> RANKSVM_T:
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

        if self.debug:
            if self.kernel == "precomputed":
                raise ValueError("Precomputed kernels cannot be provided in the debug mode.")

            # Separate 15% of the data into the validation set
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.15, random_state=rs)

            self.debug_data_ = {
                "train_score": [],
                "val_score": [],
                "primal_obj": [],
                "dual_obj": [],
                "duality_gap": [],
                "step_size": [],
                "step": [],
                "convergence_criteria": "max_iter"
            }
        else:
            X_val, y_val = None, None

        # Handle training data and calculate kernels if needed
        if self.kernel == "precomputed":
            if X.shape[0] != X.shape[1]:
                raise ValueError("Precomputed kernel matrix must be squared: You provided KX.shape = (%d, %d)."
                                 % (X.shape[0], X.shape[1]))
            self.KX_train_ = X
        else:
            self.X_train_ = X
            self.KX_train_ = self._get_kernel(self.X_train_)

        # Generate training pairs
        select_random_pairs = False
        pair_params = {"d_upper": np.inf, "d_lower": 1}
        if self.pair_generation == "eccb":
            pair_params["d_upper"] = 16
        elif self.pair_generation == "random":
            select_random_pairs = True

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

        if self.pairwise_features == "difference":
            self.A_ = self._build_A_matrix(self.pairs_train_, self.py_train_, self.KX_train_.shape[0])
        elif self.pairwise_features == "exterior_product":
            self.P_0_, self.P_1_ = self._build_P_matrices(self.pairs_train_, self.KX_train_.shape[0])

        # Initialize alpha: all dual variables are equal C
        self.alpha_ = np.full(len(self.pairs_train_), fill_value=self.C)  # shape = (n_pairs_train, )

        k = 0
        while k < self.max_iter:
            if k % 10 == 0:
                # We evaluate the duality gap every 10'th iteration to check for convergence
                prim, dual, gap = self._evaluate_primal_and_dual_objective(self.alpha_)
                if gap < self.duality_gap_threshold:
                    if self.debug:
                        self.debug_data_["convergence_criteria"] = "Duality gap lower than threshold: gap %.5f < %.5f" \
                                                                   % (gap, self.duality_gap_threshold)
                    break

                if self.debug and (k % 10 == 0):
                    # Validation and training scores
                    self.debug_data_["train_score"].append(self.score(X, y))
                    self.debug_data_["val_score"].append(self.score(X_val, y_val))

                    # Objective values
                    self.debug_data_["primal_obj"].append(prim)
                    self.debug_data_["dual_obj"].append(dual)
                    self.debug_data_["duality_gap"].append(gap)

                    # General information about the convergence
                    self.debug_data_["step"].append(k)

            s = self._solve_sub_problem(self.alpha_)  # feasible update direction

            # Get the step-size
            if self.step_size == "diminishing_1":
                tau = self._get_step_size_diminishing_1(k, self.C, self.tau_0)
            elif self.step_size == "diminishing_2":
                tau = self._get_step_size_diminishing_2(tau) if k > 0 else self.tau_0
            elif self.step_size == "diminishing_3":
                tau = self._get_step_size_diminishing_3(k)
            elif self.step_size == "linesearch":
                tau = self._get_step_size_linesearch(self.alpha_, s)

            if self.debug:
                self.debug_data_["step_size"].append(tau)

            if tau <= 0:
                if self.debug:
                    self.debug_data_["convergence_criteria"] = "%s step-size <= 0 (tau = %.5f)." % (self.step_size, tau)
                break

            self.alpha_ = self.alpha_ + tau * (s - self.alpha_)  # update alpha^{(k)} --> alpha^{(k + 1)}

            self._assert_is_feasible(self.alpha_)

            k += 1

        # Threshold dual variables to the boarder ranges, if there are very close to it.
        self.alpha_ = self._bound_alpha(self.alpha_, self.alpha_threshold, 0, self.C)
        self._assert_is_feasible(self.alpha_)

        # Only store information related to the support vectors
        is_sv = (self.alpha_ > 0)
        if self.pairwise_features == "difference":
            self.A_ = self.A_[is_sv]
        elif self.pairwise_features == "exterior_product":
            self.P_0_ = self.P_0_[is_sv]
            self.P_1_ = self.P_1_[is_sv]
        self.alpha_ = self.alpha_[is_sv]
        self.pairs_train_ = [self.pairs_train_[idx] for idx, _is_sv in enumerate(is_sv) if _is_sv]
        self.py_train_ = [self.py_train_[idx] for idx, _is_sv in enumerate(is_sv) if _is_sv]
        self.pdss_train_ = [self.pdss_train_[idx] for idx, _is_sv in enumerate(is_sv) if _is_sv]

        if self.debug:
            self.debug_data_ = {key: np.array(value) for key, value in self.debug_data_.items()}

        return self

    def predict_pointwise(self, X):
        """
        Calculates the RankSVM preference score < w , phi_i > for a set of examples.

        Note: < w , phi_i - phi_j > = < w , phi_i > - < w , phi_j >

        :param X: array-like, tutorial description
            feature-vectors: shape = (n_samples_test, d)
            -- or --
            kernel-matrix: shape = (n_samples_test, n_samples_train),

        :return: array-like, shape = (n_samples_test, ), mapped values for all examples.
        """
        if self.pairwise_features == "exterior_product":
            raise ValueError("Pointwise prediction is only possible for the 'difference' pairwise-features.")

        X = self._get_test_kernel(X)

        wtx = (X @ (self.A_.T @ self.alpha_)).flatten()  # shape = (n_test, )
        assert wtx.shape == (len(X),)

        return wtx

    def predict(self, X, return_margin=True):
        if self.pairwise_features == "difference":
            wtx = self.predict_pointwise(X)
            Y_pred = wtx[:, np.newaxis] - wtx[np.newaxis, :]  # shape = (n_samples, n_samples)
        elif self.pairwise_features == "exterior_product":
            X = self._get_test_kernel(X).T  # shape = (n_train, n_test)

            # Get all pairs between all test examples. For those we wanna predict_pointwise the order.
            n_samples_test = X.shape[1]
            # pairs = list(itertools.product(range(n_samples_test), range(n_samples_test)))
            pairs = list(itertools.combinations(range(n_samples_test), 2))
            P_0_test, P_1_test = self._build_P_matrices(pairs, n_samples_test)

            t_0 = self.alpha_ * np.array(self.py_train_)
            T_1 = self.P_0_ @ X
            T_2 = self.P_1_ @ X

            T_3 = (T_1 @ P_0_test.T) * (T_2 @ P_1_test.T) - (T_1 @ P_1_test.T) * (T_2 @ P_0_test.T)

            wtxy = 2 * t_0 @ T_3  # shape = (n_pairs_test, )

            Y_pred = np.zeros((n_samples_test, n_samples_test))
            p_0_test, p_1_test = zip(*pairs)
            Y_pred[p_0_test, p_1_test] = wtxy
            Y_pred = Y_pred - Y_pred.T

            assert np.all(np.diag(Y_pred) == 0)

        if not return_margin:
            Y_pred = np.sign(Y_pred)

        return Y_pred

    def score(self, X, y, sample_weight=None, return_score_per_dataset=False):
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

        :param return_score_per_dataset: boolean, indicating whether instead of an average score across
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
            Y = np.sign(rts[dss == ds][:, np.newaxis] - rts[dss == ds][np.newaxis, :])
            Y_pred = self.predict(X[dss == ds])
            scr, ntp = self.score_pairwise_using_prediction(Y, Y_pred)
            scores[ds] = [scr, np.sum(dss == ds).item(), ntp]

        if return_score_per_dataset:
            out = scores
        else:
            # Average the score across all datasets
            out = 0.0
            for val in scores.values():
                out += val[0]
            out /= len(scores)

        return out

    def _get_test_kernel(self, X):
        if self.kernel == "precomputed":
            if not X.shape[1] == self.KX_train_.shape[0]:
                raise ValueError("Test-train kernel must have as many columns as training examples.")
        else:
            X = self._get_kernel(X, self.X_train_)  # shape = (n_test, n_train)

        return X

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

    def _evaluate_primal_and_dual_objective(self, alpha):
        """
        Get the primal and dual objective value

            Primal: f_0(w(alpha), xi(alpha)) = 0.5 * || w(alpha) || + C sum_ij xi_ij(alpha)
            Dual:   g(alpha) =  sum_ij alpha_ij - 0.5 * alpha.T @ A @ K @ A' @ alpha

        The function furthermore returns the duality gap calculated as in [1].

        :param alpha: array-like, shape = (p, 1), current alpha estimate.

        :return: tuple (primal objective value, dual objective value, duality gap)

        [1] "A tutorial on support vector regression" by Smola et al. (2004)
        """
        if self.pairwise_features == "difference":
            predicted_margins_ = self._x_AKAt_y(y=alpha)

        elif self.pairwise_features == "exterior_product":
            predicted_margins_ = self._grad_exterior_feat(alpha)

        wtw = alpha @ predicted_margins_
        assert wtw >= 0, "Norm of the primal parameters must be >= 0"

        # See Juho's lecture 4 slides: "Convex optimization methods", slide 38
        sum_xi = np.sum(np.maximum(0, 1 - predicted_margins_))

        prim_obj = wtw / 2 + self.C * sum_xi

        dual_obj = np.sum(alpha) - wtw / 2

        return prim_obj, dual_obj, (prim_obj - dual_obj) / (np.abs(prim_obj) + 1)

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
        if self.pairwise_features == "difference":
            predicted_margins_ = self._x_AKAt_y(y=alpha)

        elif self.pairwise_features == "exterior_product":
            predicted_margins_ = self._grad_exterior_feat(alpha)

        d = 1 - predicted_margins_  # expected margin - predicted margin = 1 - predicted margin
        s = np.zeros_like(d)
        s[d > 0] = self.C

        return s

    def _get_T_5_(self):
        if not hasattr(self, "T_5_"):
            T_0 = self.P_0_ @ self.KX_train_
            T_1 = self.P_1_ @ self.KX_train_
            T_2 = (T_0 @ self.P_0_.T) * (T_1 @ self.P_1_.T)
            T_3 = T_0 @ self.P_1_.T
            T_4 = T_1 @ self.P_0_.T
            self.T_5_ = T_2 - (T_4 * T_3)

        return self.T_5_  # shape = (n_pairs, n_pairs)

    def _grad_exterior_feat(self, alpha):
        """
        Gradient of the dual objective when the exterior features are used.

        Gradient was calculated using the online service: http://www.matrixcalculus.org/ [1]

        [1] "Computing Higher Order Derivatives of Matrix and Tensor Expressions", S. Laue et al. (2018)
        """
        y = np.array(self.py_train_)

        T_5 = self._get_T_5_()

        # shape = (n_pairs,)
        t_7 = y * alpha  # originally t_6
        t_6 = T_5 @ t_7  # originally t_5

        return (t_6 + (T_5 @ t_7)) * y  # originally (t_6 * y) + ((T_5 @ t_7) * y)

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
        alpha[np.isclose(alpha, min_alpha_val, atol=threshold, rtol=1e-9)] = min_alpha_val
        alpha[np.isclose(alpha, max_alpha_val, atol=threshold, rtol=1e-9)] = max_alpha_val
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

        A = sp_sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n_pairs, n_samples))

        # TODO: We can build the a matrix using the P matrix constructor
        # P_0, P_1 = self._build_P_matrices(pairs, n_samples)
        # A = P_0 - P_1
        # A = y * A   # multiply row-wise
        # assert np.issparse(A)

        return A

    @staticmethod
    def _build_P_matrices(pairs, n_samples):
        n_pairs = len(pairs)

        row_ind = np.arange(n_pairs)
        col_ind_0 = np.array([pair[0] for pair in pairs])
        col_ind_1 = np.array([pair[1] for pair in pairs])

        P_0 = sp_sparse.csr_matrix((np.ones(n_pairs), (row_ind, col_ind_0)), shape=(n_pairs, n_samples))
        P_1 = sp_sparse.csr_matrix((np.ones(n_pairs), (row_ind, col_ind_1)), shape=(n_pairs, n_samples))

        return P_0, P_1

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

    # ---------------------------------------
    # STEP SIZE CALCULATION
    # ---------------------------------------
    @staticmethod
    def _get_step_size_diminishing_1(k, C, t_0):
        """
        Calculate the step size using the diminishing strategy.

            step size = t_0 / (1 + t_0 * C * k)

            Border cases:
                k = 0, first iteration: step size = t_0
                t_0 = 1: step size = 1 / (1 + C * k)

        :param k: scalar, current iteration. First iteration is assumed to be k = 0.
        :param C: scalar, regularization parameter RankSVM
        :param t_0: scalar, initial step-size
        :return: scalar, step size

        Ref: [1] "Towards Optimal One Pass Large Scale Learning with Averaged Stochastic Gradient Descent",
                 Wei Xu, 2011, ArXiv
             [2] "Large-Scale Machine Learning with Stochastic Gradient Descent", Leo Bottou, 2010
        """
        return t_0 / (1 + t_0 * C * k)

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
    def _get_step_size_diminishing_3(k):
        """
        Simple step size only depending on the current iteration. Proposed in [1].

            step size = 2 / (k + 2)

        Ref: [1] "Stochastic Block-Coordinate Frank-Wolfe Optimization for Structural SVMs", Lacoste-Julien et al.,
                 2012
        """
        return 2 / (k + 2)

    def _get_step_size_linesearch(self, alpha, s):
        """
        Calculate step-size using line-search.

        :param alpha: array-like, shape = (p,), dual variables in step k, before update.

        :param s: array-like, shape = (p,), solution of the sub-problem

        :return: scalar, step-size using line-search
        """
        if self.pairwise_features == "exterior_product":
            raise NotImplementedError("Linea-search currently implemented only for 'difference' features.")

        delta_alpha = s - alpha

        nom = np.sum(delta_alpha) - self._x_AKAt_y(alpha, delta_alpha)
        den = self._x_AKAt_y(delta_alpha, delta_alpha)

        tau = nom / den

        assert tau <= 1.0

        return tau


    @staticmethod
    def score_pointwise_using_predictions(y, y_pred, normalize=True):
        """
        :param y: array-like, shape = (n_samples, ), true target values for the test samples.

        :param y_pred: array-like, shape = (n_samples, ), predicted target values for the test samples.

        :param normalize: logical, indicating whether the number of correctly classified pairs
            should be normalized by the total number of true target values for which holds y_i < y_j.

        :return: tuple(scalar, scalar), pairwise prediction accuracy and number of test pairs used
        """
        n_decisions = 0
        tp = 0.0

        for i, j in itertools.permutations(range(len(y_pred)), 2):
            if y[i] < y[j]:  # i is preferred over j
                n_decisions += 1
                if y_pred[j] == y_pred[i]:  # tie predicted
                    tp += 0.5
                elif y_pred[i] < y_pred[j]:  # i is _predicted_ to be preferred over j
                    tp += 1.0

        if n_decisions == 0:
            raise RuntimeError("Score is undefined of all true target values are equal.")

        if normalize:
            tp /= n_decisions

        return tp, n_decisions

    @staticmethod
    def score_pairwise_using_prediction(Y, Y_pred, normalize=True):
        """
        :param Y: array-like, shape = (n_samples, n_samples), pairwise object preference

        :param Y_pred: array-like, shape = (n_samples, n_samples), pairwise object preference prediction

        :param normalize: logical, indicating whether the number of correctly classified pairs
            should be normalized by the total number of pairs (n_pairs_test).

        :return: tuple(scalar, scalar), pairwise prediction accuracy and number of test pairs used
        """
        assert Y.shape == Y_pred.shape, "True and predicted label matrices must have the same size."
        assert Y.shape[0] == Y.shape[1], "Label matrix must be squared."
        assert np.all(np.diag(Y) == 0), "Assume that the diagonal labels are: Is i preferred over i?"
        assert np.all(np.diag(Y_pred) == 0), "Assume that the diagonal labels are: Is i preferred over i?"
        assert np.all(Y == - Y.T), "Assume anti-symmetric relation"
        assert np.all(Y_pred == - Y_pred.T), "Predicted relations must be anti-symmetric"

        n_samples = Y.shape[0]

        n_decisions = 0
        tp = 0.0

        for i, j in itertools.permutations(range(n_samples), 2):
            if Y[i, j] < 0:  # i is preferred over j
                n_decisions += 1
                if Y_pred[i, j] == 0:  # tie predicted
                    tp += 0.5
                elif Y_pred[i, j] < 0:  # i is _predicted_ to be preferred over j
                    tp += 1.0

        if n_decisions == 0:
            raise RuntimeError("Score is undefined of all true target values are equal.")

        if normalize:
            tp /= n_decisions

        return tp, n_decisions


if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import GroupKFold

    # Load example data
    data = pd.read_csv("tutorial/example_data.csv", sep="\t")
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

    for feature in ["difference", "exterior_product"]:
        ranksvm = KernelRankSVC(C=8, kernel="minmax", pair_generation="random", random_state=292, max_iter=1000,
                                alpha_threshold=1e-2, pairwise_features=feature).fit(X_train, y_train)

        print(feature, ranksvm.score(X_test, y_test))
