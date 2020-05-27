"""
Sample code automatically generated on 2020-05-26 09:31:44

by www.matrixcalculus.org

from input

d/dalpha alpha' * (y * y' .* ((P * K * P') .* (Q * K * Q') - (P * K * Q') .* (Q * K * P'))) * alpha = (((P*K*P').*(Q*K*Q')-(P*K*Q').*(Q*K*P'))*(y.*alpha)).*y+(((P*K*P').*(Q*K*Q')-(Q*K*P').*(P*K*Q'))*(alpha.*y)).*y

where

K is a symmetric matrix
P is a matrix
Q is a matrix
alpha is a vector
y is a vector

The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

import numpy as np

def fAndG(K, P, Q, alpha, y):
    assert isinstance(K, np.ndarray)
    dim = K.shape
    assert len(dim) == 2
    K_rows = dim[0]
    K_cols = dim[1]
    assert isinstance(P, np.ndarray)
    dim = P.shape
    assert len(dim) == 2
    P_rows = dim[0]
    P_cols = dim[1]
    assert isinstance(Q, np.ndarray)
    dim = Q.shape
    assert len(dim) == 2
    Q_rows = dim[0]
    Q_cols = dim[1]
    assert isinstance(alpha, np.ndarray)
    dim = alpha.shape
    assert len(dim) == 1
    alpha_rows = dim[0]
    assert isinstance(y, np.ndarray)
    dim = y.shape
    assert len(dim) == 1
    y_rows = dim[0]
    assert Q_rows == y_rows == P_rows == alpha_rows
    assert P_cols == Q_cols == K_rows == K_cols

    T_0 = (P).dot(K)
    T_1 = (Q).dot(K)
    T_2 = ((T_0).dot(P.T) * (T_1).dot(Q.T))
    T_3 = (T_0).dot(Q.T)
    T_4 = (T_1).dot(P.T)
    t_5 = ((T_2 - (T_3 * T_4))).dot((y * alpha))
    t_6 = (alpha * y)
    functionValue = (t_6).dot(t_5)
    gradient = ((t_5 * y) + (((T_2 - (T_4 * T_3))).dot(t_6) * y))

    return functionValue, gradient

def checkGradient(K, P, Q, alpha, y):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(4)
    f1, _ = fAndG(K, P, Q, alpha + t * delta, y)
    f2, _ = fAndG(K, P, Q, alpha - t * delta, y)
    f, g = fAndG(K, P, Q, alpha, y)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=1)))

def generateRandomData():
    K = np.random.randn(3, 3)
    K = 0.5 * (K + K.T)  # make it symmetric
    P = np.random.randn(4, 3)
    Q = np.random.randn(4, 3)
    alpha = np.random.randn(4)
    y = np.random.randn(4)

    return K, P, Q, alpha, y

if __name__ == '__main__':
    K, P, Q, alpha, y = generateRandomData()
    functionValue, gradient = fAndG(K, P, Q, alpha, y)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(K, P, Q, alpha, y)
