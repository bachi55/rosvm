####
#
# The MIT License (MIT)
#
# Copyright 2017-2019 Eric Bach <eric.bach@aalto.fi>
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

from ranksvm.kernel_utils import tanimoto_kernel, minmax_kernel

import numpy as np
import scipy.sparse as sp_sparse
import itertools
import warnings
warnings.simplefilter('ignore', sp_sparse.SparseEfficiencyWarning)
warnings.simplefilter('always', UserWarning)

# Sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection._validation import check_random_state

from time import process_time, sleep
from collections import deque


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
    def __init__(self, C=1.0, kernel="precomputed", tol=0.001, max_iter=1000, min_iter=5, t_0=0.1,
                 step_size_algorithm="diminishing", gamma=None, coef0=1, degree=3, kernel_params=None,
                 convergence_criteria="max_alpha_change", verbose=False, debug=0, random_state=None):

        if convergence_criteria not in ["rel_dual_obj_change", "max_alpha_change",
                                        "rel_prim_dual_gap_change", "max_iter"]:
            raise ValueError("Invalid convergence criteria: %s" % convergence_criteria)

        if step_size_algorithm not in ["diminishing", "diminishing_2", "fixed"]:
            raise ValueError("Invalid step-size algorithm: %s" % step_size_algorithm)

        # Parameter for the optimization
        self.tol = tol
        self.max_iter = max_iter
        self.min_iter = min_iter
        self.step_size_algorithm = step_size_algorithm
        self.convergence_criteria = convergence_criteria
        self.t_0 = t_0

        # Kernel parameters
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel_params = kernel_params

        # General Ranking SVM parameter
        self.C = C

        # Debug parameters
        self.verbose = verbose
        self.debug = debug
        self.random_state = random_state  # used to split the examples into positive and negative class.
        self.error_state = None
        self.k_convergence = None
        self.t_convergence = None
        self.obj_has_converged = False

        # Training tutorial used for fitting
        self._pairs_fit = None
        self._pairwise_labels_fit = None
        self._KX_fit = None
        self._X_fit = None
        self._pairwise_confidences_fit = None
        self._A = None
        self._last_AKAt_y = None

        # Model parameters
        self._alpha = None
        self._idx_sv = None

    def fit(self, X, pairs, pairwise_labels=None, pairwise_confidences=None, alpha_init=None):
        """
        Estimating the parameters of the dual ranking svm with scaled margin.
        The conditional gradient descent algorithm is used to find the optimal
        alpha vector.

        :param X: array-like, shape = (n_samples, n_features) or (n_samples, n_samples)
            Object features or object similarities (kernel). If self.kernel == "precomputed"
            then X is interpreted as symmetric kernel matrix, otherwise as feature matrix.
            In this case the kernel is calculated on the fly.

        :param pairs: list of index-tuples, shape = (n_pairs, 1), encoding the pairwise
            object relations:
                i is preferred over j, u is preferred over v, ... --> [(i,j), (u,v), ...]},

                where i, j, u, v, ... are the indices corresponding to the object descriptions
                given in X.

        :param pairwise_labels: array-like, shape = (n_pairs,). Labels for pairwise relations:
            Positive and negative class:
                y =  1 (positive) : pair = (i,j) with i elutes before j
                y = -1 (negative) : pair = (i,j) with j elutes before i

            Default:

                In this implementation we assume that, e.g.,
                    i is preferred over j <==> target[i] < target[j], ...

                That means that we set all y_ij = 1.

        :param pairwise_confidences: array-like, shape = (n_pairs, 1). Margins r_ij's for each
            pair: w^T(phi_j - phi_i) >= r_ij - xi_ij

            The default value is None, which means that all margins r_ij = 1.

        :param alpha_init: array-like, shape = (n_pairs,) or scalar
            if array: initial values for the dual variables alpha_ij's
            if scalar: all dual variables are initialized with this value

            The default value is None, which means all the dual variables alpha_ij = 0

        :return: pointer, to estimator it self
        """
        self._pairs_fit = pairs
        n_pairs = len(self._pairs_fit)

        # Handle object representation
        if self.kernel == "precomputed":
            if X.shape[0] != X.shape[1]:
                raise ValueError("Precomputed kernel matrix must be squared: KX.shape = (%d, %d)."
                                 % (X.shape[0], X.shape[1]))

            self._KX_fit = X
        else:
            self._X_fit = X
            self._KX_fit = self._get_kernel(self._X_fit)

        # Handle / initialize pairwise labels
        if pairwise_labels is not None:
            if len(pairwise_labels) != n_pairs:
                raise ValueError("The pairwise_labels vector and the list of pairwise relations "
                                 "must have the same length: %d vs %d" % (len(pairwise_labels), n_pairs))

            self._pairwise_labels_fit = pairwise_labels

        # Handle / initialize pairwise confidence
        if pairwise_confidences is None:
            self._pairwise_confidences_fit = np.ones((n_pairs, 1))
        else:
            if len(pairwise_confidences) != n_pairs:
                raise ValueError("The pairwise_confidences must have the same length as the "
                                 "list of pairwise relations: %d vs %d." % (len(pairwise_confidences), n_pairs))

            self._pairwise_confidences_fit = pairwise_confidences

            # Make the confidence vector being column vector
            if len(self._pairwise_confidences_fit.shape) == 1:
                self._pairwise_confidences_fit = self._pairwise_confidences_fit.reshape((-1, 1))

            assert (self._pairwise_confidences_fit.shape == (n_pairs, 1)), "Check shape of pairwise confidences."

        # Handle / initialize initial dual variables
        if alpha_init is None:
            alpha_k = np.zeros((n_pairs, 1))
        else:
            if np.isscalar(alpha_init):
                alpha_k = np.ones((n_pairs, 1)) * alpha_init
            else:
                alpha_k = alpha_init

            if not self._is_feasible(alpha_k):
                ValueError("The provided initial alpha is not in the feasible set. "
                           "Check that 0 <= alpha_ij <= C for all (i,j) in P.")

        # == Conditional gradient algorithm ==
        if self.verbose:
            starttime = process_time()

        self._A = self._build_A_matrix(pairs=self._pairs_fit, n_samples=self._KX_fit.shape[0],
                                       y=self._pairwise_labels_fit)
        k = 1  # first iteration
        t = self.t_0  # initial step-width used for 'diminishing_2'
        obj_change = np.inf  # Track the change of the objective.

        if self.convergence_criteria == "rel_dual_obj_change":
            dual_obj_hist = deque(maxlen=2)  # store the last two dual objective values
            self._last_AKAt_y = self._x_AKAt_y(y=alpha_k)
            dual_obj_hist.append(self._evaluate_dual_objective(alpha_k, self._pairwise_confidences_fit))

        while k <= self.max_iter:
            # Determine a feasible update direction
            alpha_delta = self._get_optimal_alpha_bar(alpha_k, self._pairwise_confidences_fit) - alpha_k

            # Determine the step width
            if self.step_size_algorithm == "diminishing":
                t = self._get_step_size_diminishing(k, self.C, self.t_0)
            elif self.step_size_algorithm == "diminishing_2":
                t = self._get_step_size_diminishing_2(t)
            elif self.step_size_algorithm == "fixed":
                t = self.t_0

            # Store currently alpha
            alpha_old = alpha_k

            # Update alpha
            # alpha_k+1 = alpha_k + tau * alpha_delta
            #           = alpha_k + tau * (alpha* - alpha_k)
            #           = (1 - tau) * alpha_k + tau * alpha*
            alpha_k = alpha_k + t * alpha_delta

            # Check feasibility
            if not self._is_feasible(alpha_k):
                self.error_state = {"alpha_old": alpha_old, "alpha_new": alpha_k, "alpha_delta": alpha_delta, "t": t}
                raise RuntimeError("Alpha is after update not in the feasible set anymore.")

            # Calculate convergence criteria value
            if self.convergence_criteria == "rel_dual_obj_change":
                # Calculate the relative change of the dual objective
                dual_obj_hist.append(self._evaluate_dual_objective(alpha_k, self._pairwise_confidences_fit))

                obj_change = np.abs((dual_obj_hist[-2] - dual_obj_hist[-1]) / dual_obj_hist[-1])
            elif self.convergence_criteria == "max_alpha_change":
                # Calculate the maximum change of the updated alpha
                obj_change = np.max(np.abs(alpha_k - alpha_old))
            elif self.convergence_criteria == "rel_prim_dual_gap_change":
                dual_obj = self._evaluate_dual_objective(alpha_k, self._pairwise_confidences_fit)
                prim_obj = self._evaluate_primal_objective(alpha_k, self._pairwise_confidences_fit)

                obj_change = (prim_obj - dual_obj) / (np.abs(prim_obj) + 1)  # See Smola2004, Tutorial on SVR
            elif self.convergence_criteria == "max_iter":
                if k + 1 > self.max_iter:
                    obj_change = 0.0

            if self.verbose and k % 50 == 0:
                print("\rIteration %d: Objective = %f, Objective change = %f, Step t = %f" %
                      (k, self._evaluate_dual_objective(alpha_k, self._pairwise_confidences_fit), obj_change, t),
                      end="", flush=True)
                sleep(0.25)

            if (k >= self.min_iter) and obj_change < self.tol:
                self.obj_has_converged = True
                break
            else:
                self.obj_has_converged = False
                k += 1

        self.k_convergence = k if self.obj_has_converged else (k - 1)
        if self.verbose:
            print("\r", end="", flush=True)
            print("\rIteration %d: Objective = %f, Objective change = %f, Step t = %f" %
                  (self.k_convergence, self._evaluate_dual_objective(alpha_k, self._pairwise_confidences_fit),
                   obj_change, t))
            print("Convergence: Time = %.3fs" % (process_time() - starttime))

        if self.k_convergence == self.max_iter and self.convergence_criteria != "max_iter":
            warnings.warn("Optimization algorithm stopped due to maximum number of iterations.")

        # Store final step size
        self.t_convergence = t

        # Find the indices of the support vectors
        self._alpha = alpha_k
        self._idx_sv = np.where(self._alpha > 0)[0]
        if self.verbose:
            print("Number of support vectors: %d (out of %d)." % (self._idx_sv.shape[0], self._alpha.shape[0]))

        return self

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

    def predict(self, X1, X2=None, n_jobs=1, return_label=True):
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

        :param n_jobs: integer, number of jobs passed to '_get_kernel', default=1

        :param return_label: boolean, indicating whether the label {-1, 1} should be returned, default=True

        :return: array-like, shape = (n_samples_set1, n_samples_set2). Entry at index [u, v] is either:
             1: u (set 1 example) elutes before v (set 2 example) <==> wtx(v) > wtx(u) <==> sign(wtx(v) - wtx(u)) > 0
            -1: otherwise (v elutes before u)
             0: predicted target values for u and v are equal: wtx(v) == wtx(u)
        """
        wtX1 = self.map_values(X1, n_jobs=n_jobs)

        if X2 is None:
            wtX2 = wtX1
        else:
            wtX2 = self.map_values(X2, n_jobs=n_jobs)

        # Calculate pairwise predictions
        Y = - wtX1 + wtX2.T
        if return_label:
            Y = np.sign(Y)  # -sign(x) = sign(-x), need to flip sign as we have (i, j) => j - i > 0

        return Y

    def map_values(self, X, n_jobs=1):
        """
        Calculates w^T\phi for a set of examples.

        :param X: array-like, tutorial description
            feature-vectors: shape = (n_samples_test, d)
            -- or --
            kernel-matrix: shape = (n_samples_train, n_samples_test),

        :param n_jobs: integer, number of jobs passed to '_get_kernel' (default = 1)

        :return: array-like, shape = (n_samples_test, ), mapped values for all examples.
        """
        if self.kernel == "precomputed":
            if not X.shape[0] == self._KX_fit.shape[0]:
                raise ValueError("Train-test kernel must have as many rows as training examples.")
        else:
            X = self._get_kernel(self._X_fit, X, n_jobs=n_jobs)  # shape = (n_samples_train, n_samples_test)

        return ((self._alpha.T @ self._A) @ X).T  # TODO: Consider the support vectors

    def score(self, X, pairs, sample_weight=None, normalize=True):
        """
        :param X: array-like, tutorial description
            feature-vectors: shape = (n_samples_test, d)
            -- or --
            kernel-matrix: shape = (n_samples_train, n_samples_test),

            with n being the number of test molecules.

        :param pairs: list of index-tuples, shape = (n_pairs_test,), encoding the pairwise
            object relations:
                i is preferred over j, u is preferred over v, ... --> [(i,j), (u,v), ...]},

                where i, j, u, v, ... are the indices corresponding to the object descriptions
                given in X.

            In this implementation we assume that, e.g.,
                i is preferred over j <==> target[i] < target[j], ...

        :param sample_weight: parameter currently not used.

        :param normalize: logical, indicating whether the number of correctly classified pairs
            should be normalized by the total number of pairs (n_pairs_test).

        :return: scalar, pairwise prediction accuracy
        """
        if len(pairs) == 0:
            raise RuntimeError("Score is undefined of the number of test pairs is zero.")

        return self.score_pairwise_using_prediction(self.predict(X), pairs, normalize=normalize)

    def score_pointwise(self, X, y, normalize=True):
        """
        :param X: array-like, tutorial description
            feature-vectors: shape = (n_samples_test, d)
            -- or --
            kernel-matrix: shape = (n_samples_train, n_samples_test),

            with n being the number of test molecules.

        :param y: array-like, shape = (n_samples_test,), true target values for the test samples.

        :param normalize: logical, indicating whether the number of correctly classified pairs
            should be normalized by the total number of pairs (n_pairs_test).

        :return: scalar, pairwise prediction accuracy
        """
        return self.score_pointwise_using_predictions(self.map_values(X), y, normalize)

    def _evaluate_primal_objective(self, alpha, pairwise_confidence):
        """
        Get the function value of f_0:
            f_0(w(alpha), xi(alpha, r)) = 0.5 w(alpha)^Tw(alpha) + C 1^Txi(alpha, r)

        :param alpha: array-like, shape = (p, 1), current alpha estimate.
        :param pairwise_confidence: array-like, shape = (p, 1), confidence for each pairwise relation.

        :return: scalar, f_0(w(alpha), xi(alpha, r))
        """
        wtw = alpha.T @ self._last_AKAt_y

        # See Juho's lecture 4 slides: "Convex optimization methods", slide 38
        xi = np.maximum(0, pairwise_confidence - self._last_AKAt_y)

        return wtw / 2 + self.C * np.sum(xi)

    def _evaluate_dual_objective(self, alpha, pairwise_confidence):
        """
        Get the function of g:
            g(alpha, r) = r^T alpha - 0.5 alpha^T H alpha

        :param alpha: array-like, shape = (p, 1), current alpha estimate.
        :param pairwise_confidence: array-like, shape = (p, 1), confidence for each pairwise relation.

        :return: scalar, g(alpha, r)
        """
        return pairwise_confidence.T @ alpha - (alpha.T @ self._last_AKAt_y) / 2.0

    def _is_feasible(self, alpha):
        """
        Check whether the provided alpha vector is in the feasible set:

            1) alpha.shape == (|P|, 1)
            2) 0 <= alpha_ij <= C for all (i,j) in P

        :param alpha: array-like, shape = (p, 1), dual variables

        :return: Boolean:
            True,  if alpha is in the feasible set
            False, otherwise
        """
        n_pairs = len(self._pairs_fit)

        is_feasible = (0 <= alpha).all() and (alpha <= self.C).all() and (alpha.shape == (n_pairs, 1))

        return is_feasible

    def _get_optimal_alpha_bar(self, alpha, r):
        """
        Finding the alpha*

        :param r: array-like, shape = (p, 1), confidence values for the pairwise
            relations

        :return: array-like, shape = (p, 1), alpha*
        """
        self._last_AKAt_y = self._x_AKAt_y(y=alpha)
        d = r - self._last_AKAt_y

        alpha_bar = np.zeros((self._A.shape[0], 1))
        alpha_bar[d > 0] = self.C

        return alpha_bar

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

            return (x.T @ (self._A @ (self._KX_fit @ (self._A.T @ y)))).item()

        elif (x is None) and (y is not None):

            return self._A @ (self._KX_fit @ (self._A.T @ y))

        elif (x is not None) and (y is None):

            return (self._A @ ((self._A.T @ x).T @ self._KX_fit).T).T

        else:

            return self._A @ self._KX_fit @ self._A.T

    # ---------------------------------
    # Static methods
    # ---------------------------------
    @staticmethod
    def _build_A_matrix(pairs, n_samples, y=None):
        """
        Construct a matrix A (p x l) so that:
            A_{(i,j),:} = y_ij (0...0, -1, 0...0, 1, 0...0).

        This matrix is used to simplify the optimization using 'difference' features.

        :param pairs: List of tuples (i,j) of length p containing the pairwise relations:
            i elutes before j, ... --> [(i,j), (u,v), ...]

        :param n_samples: scalar, number of training examples

        :param y: array-like, shape = (p, 1). Label for each pairwise relation (optional):
            Positive and negative class:
                y =  1 (positive) : pair = (i,j) with i elutes before j
                y = -1 (negative) : pair = (j,i) with i elutes before j

            If y is None: it is assumed that all pairs belong to the positive class.

        :return: sparse array-like, shape = (p, l)
        """
        n_pairs = len(pairs)

        row_ind = np.append(np.arange(0, n_pairs), np.arange(0, n_pairs))
        col_ind = np.append([pair[0] for pair in pairs], [pair[1] for pair in pairs])
        data = np.append(-1 * np.ones((1, n_pairs)), np.ones((1, n_pairs)))

        if y is not None:
            data = data * np.append(y, y)

        return sp_sparse.csr_matrix((data, (row_ind, col_ind)), shape=(n_pairs, n_samples))

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
    def score_pointwise_using_predictions(y_pred, y, normalize=True):
        """
        :param y_pred: array-like, shape = (n_samples_test,), predicted target values for the test samples.
        :param y: array-like, shape = (n_samples_test,), true target values for the test samples.
        :param normalize: logical, indicating whether the number of correctly classified pairs
            should be normalized by the total number of pairs (n_pairs_test).

        :return: scalar, pairwise prediction accuracy
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

        return tp

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


# ---------------------------------
# Deprecated code
# ---------------------------------
def get_pairwise_labels(pairs, balance_classes=True, random_state=None):
    """
    Task: Get the labels for each pairwise relation. By default the pairs are randomly distributed across
          a positive and negative class:
          y =  1 (positive) : pair = (i,j) with i elutes before j
          y = -1 (negative) : pair = (j,i) with i elutes before j

    :param pairs: list of index pairs considered for the pairwise features
        [(i,j), ...] for which it holds: i elutes before j

    :param balance_classes: binary indicating, whether 50% of the pairs should be
        swapped and assigned a negative target value. This is use-full if a binary
        SVM is training, default=True

    :param random_state: int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    :return: tuple (pairs_out, y_out):
        pairs_out: List of pairwise relations of length k
        y_out: array-like, shape=(n_pairs, 1), label vector
    """
    n_pairs = len(pairs)

    y = np.ones((n_pairs, 1), dtype="int")  # pairwise labels

    if balance_classes:
        pairs_out = pairs.copy()

        # Get a random generator
        rng = check_random_state(random_state)

        # Split the example into positive and negative class: 50/50
        idc_n = rng.choice(np.arange(0, n_pairs), size=n_pairs // 2, replace=False)

        # Swap label and pair for negative class
        for idx_n in idc_n:
            pairs_out[idx_n] = (pairs_out[idx_n][1], pairs_out[idx_n][0])
            y[idx_n] = -y[idx_n]
    else:
        pairs_out = pairs

    return pairs_out, y
