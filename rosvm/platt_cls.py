# --[Basic Function]---------------------------------------------------------------------
# input decision_values, real_labels{1,-1}, #positive_instances, #negative_instances
# output [A,B] that minimize sigmoid likilihood
# refer to Platt's Probablistic Output for Support Vector Machines

# Source: https://work.caltech.edu/~htlin/program/libsvm/

import numpy as np
import logging as lg
LOG = lg.getLogger("PlattProbabilities")
LOG.addHandler(lg.NullHandler())
lg.basicConfig(format="[PlattProbabilities] %(message)s", level=lg.INFO)

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


# TODO: Make class compatible with sklearn's (>= 0.22) 'check_is_fitted' function.


class PlattProbabilities(BaseEstimator, ClassifierMixin):
    def __init__(self, prior1=None, prior0=None, maxiter=100, minstep=1e-10, sigma=1e-12, eps=1e-5):
        """

        :param prior1:
        :param prior0:
        :param maxiter: scalar, Maximum number of iterations (default: 100)
        :param minstep: scalar, Minimum step taken in line search (default: 1e-10)
        :param sigma: scalar, For numerically strict PD of Hessian (default: 1e-12)
        :param eps:
        :return:
        """
        # Class priors
        self.prior1 = prior1
        self.prior0 = prior0

        # Optimization parameter
        self.maxiter = maxiter
        self.minstep = minstep
        self.sigma = sigma
        self.eps = eps

        # Sigmoid parameter
        self.A = None
        self.B = None

    def fit(self, X, y):
        assert (np.all(np.isin(y, [-1, 1]))), "Labels must be in {-1, 1}."

        # Count prior0 and prior1 if needed
        if self.prior1 is None or self.prior0 is None:
            self.prior1 = np.sum(y == 1)
            self.prior0 = np.sum(y == -1)

        # Construct Target Support
        t = self._getTargetSupport(y)

        # Initial Point and Initial Fun Value
        self.A = 0.0
        self.B = np.log((self.prior0 + 1.0) / (self.prior1 + 1.0))
        fval = PlattProbabilities._getFVal(X, t, self.A, self.B)

        it = 0
        while it < self.maxiter:
            # Update Gradient and Hessian (use H' = H + sigma I)
            h11 = self.sigma  # Numerically ensures strict PD
            h22 = self.sigma
            h21 = g1 = g2 = 0.0
            for i in range(len(y)):
                fApB = X[i] * self.A + self.B
                if fApB >= 0:
                    p = np.exp(-fApB) / (1.0 + np.exp(-fApB))
                    q = 1.0 / (1.0 + np.exp(-fApB))
                else:
                    p = 1.0 / (1.0 + np.exp(fApB))
                    q = np.exp(fApB) / (1.0 + np.exp(fApB))
                d2 = p * q
                h11 += X[i] * y[i] * d2
                h22 += d2
                h21 += X[i] * d2
                d1 = t[i] - p
                g1 += X[i] * d1
                g2 += d1

            # Stopping Criteria
            if abs(g1) < self.eps and abs(g2) < self.eps:
                break

            # Finding Newton direction: -inv(H') * g
            det = h11 * h22 - h21 * h21
            dA = -(h22 * g1 - h21 * g2) / det
            dB = -(-h21 * g1 + h11 * g2) / det
            gd = g1 * dA + g2 * dB

            # Line Search
            stepsize = 1
            while stepsize >= self.minstep:
                newA = self.A + stepsize * dA
                newB = self.B + stepsize * dB

                # New function value
                newf = PlattProbabilities._getFVal(X, t, newA, newB)

                # Check sufficient decrease
                if newf < fval + 0.0001 * stepsize * gd:
                    self.A, self.B, fval = newA, newB, newf
                    break
                else:
                    stepsize = stepsize / 2.0

            if stepsize < self.minstep:
                LOG.warning("Line-search failed: iter=%d, A=%f, B=%f, g1=%f, g2=%f, dA=%f, dB=%f, gd=%f" % (
                    it, self.A, self.B, g1, g2, dA, dB, gd))
                break

            it += 1

        if it >= self.maxiter - 1:
            LOG.warning("Reached maximum iterations: g1=%f, g2=%f" % (g1, g2))

        return self

    def predict(self, X):
        """
        Predict Platt posterior probability for a set of decision values

        :param X: array-like, shape=(n_samples,), decision function values

        :return: array-like, shape=(n_samples,), Platt posterior probabilities
        """
        # check_is_fitted(self, ["A", "B"])

        fApB = X * self.A + self.B

        probs = np.full_like(fApB, fill_value=np.nan)
        probs[fApB >= 0] = np.exp(-fApB[fApB >= 0]) / (1.0 + np.exp(-fApB[fApB >= 0]))
        probs[fApB < 0] = 1.0 / (1 + np.exp(fApB[fApB < 0]))

        return probs

    def score(self, X, y, sample_weight=None):
        """
        Return the log-likelihood score

        :param X: array-like, shape=(n_samples,), decision function values
        :param y: array-like, shape=(n_samples,), labels
        :param sample_weight: not used

        :return: scalar, log-likelihood score
        """
        # check_is_fitted(self, ["A", "B"])

        t = self._getTargetSupport(y)

        return - self._getFVal(X, t, self.A, self.B)

    def _getTargetSupport(self, y):
        # check_is_fitted(self, ["prior1", "prior0"])

        hiTarget = (self.prior1 + 1.0) / (self.prior1 + 2.0)
        loTarget = 1 / (self.prior0 + 2.0)

        t = np.full_like(y, fill_value=np.nan)
        t[y == 1] = hiTarget
        t[y == -1] = loTarget

        return t

    @staticmethod
    def _getFVal(X, t, A, B):
        """
        Return the negative log-likelihood score (Eq. (2) in [Lin2007])

        :param X: array-like, shape=(n_samples,), decision function values
        :param t: array-like, shape=(n_samples,), target support
        :param A: scalar, sigmoid scaling parameter
        :param B: scalar, sigmoid shift parameter

        :return: scalar, negative log-likelihood score
        """
        fApB = X * A + B

        fval = np.full_like(t, fill_value=np.nan)
        if np.any(fApB >= 0):
            fval[fApB >= 0] = t[fApB >= 0] * fApB[fApB >= 0] + np.log(1 + np.exp(-fApB[fApB >= 0]))
        if np.any(fApB < 0):
            fval[fApB < 0] = (t[fApB < 0] - 1) * fApB[fApB < 0] + np.log(1 + np.exp(fApB[fApB < 0]))

        sfval = np.sum(fval)
        assert (not np.isnan(sfval))

        return sfval
