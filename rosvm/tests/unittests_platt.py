import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ranksvm.rank_svm_cls import KernelRankSVC
from ranksvm.pair_utils import get_pairs_single_dataset
from ranksvm.kernel_utils import minmax_kernel
from ranksvm.platt_cls import PlattProbabilities
from ranksvm.model_selection_cls import find_hparam_single_dataset

from sklearn.model_selection import train_test_split


def load_data(idir="../../example_data/retention_times/", dataset="FEM_long"):
    """
    Load retention timed dataset and molecular fingerprints for compounds.
    """
    # Set up filenames
    feat_fn = idir + "/fps_substructurecount_" + dataset + ".csv"
    rts_fn = idir + "/rts_" + dataset + ".csv"

    # Load features and retention times
    X = pd.read_csv(feat_fn, header=None, delimiter=",", index_col=None).sort_values(by=0)
    y = pd.read_csv(rts_fn, header=None, delimiter=",", index_col=None).sort_values(by=0)
    assert (np.all(X[0].values == y[0].values)), "Molecules are not in the same order"

    return X.loc[:, 1:].values, y.loc[:, 1].values


def sigmoid(x, k=1., x_0=0., T=1.):
    """
    Sigmoid function defined as[1]:

        f(x) = T / (1 + exp(-k * (x - x_0)))

    :param x: scalar or array-like with shape=(n, m), values to transform using the logistic function.
    :param k: scalar, The logistic growth rate or steepness of the curve.
    :param x_0: scalar, The x-value of the sigmoid's midpoint
    :param T: scalar, The curve's maximum value
    :return: scalar or array-like with shape=(n, m), transformed input values

    Reference:
        [1] https://en.wikipedia.org/wiki/Logistic_function
    """
    return T / (1 + np.exp(-k * (x - x_0)))


class TestPlattEstimation(unittest.TestCase):
    def test_on_small_example(self):
        X, y = load_data(dataset="LIFE_old")
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1201)

        # Calculate kernels
        K_train = minmax_kernel(X_train)
        K_test = minmax_kernel(X_train, X_test)

        # Get training pairs
        P_train, _ = get_pairs_single_dataset(y_train, 0, 16)
        n_pairs_train = len(P_train)
        print("#Pairs (train):", n_pairs_train)

        # prepare plotting
        fig, axrr = plt.subplots(3, 3, figsize=(10, 7), sharex="col")

        # Train the ranksvm
        for i, C in enumerate([0.0625, 1.0, 8.0]):
            ranksvm = KernelRankSVC(random_state=787, t_0=0.5,
                                    step_size_algorithm="diminishing_2",
                                    C=C, convergence_criteria="max_iter", max_iter=1000) \
                .fit(K_train, P_train, alpha_init=C / n_pairs_train)

            # get pairwise predictions
            Y_test_pred = ranksvm.predict(K_test, return_label=False)
            # get pairwise ground-truth labels
            Y_test = - np.sign(y_test[:, np.newaxis] - y_test[np.newaxis, :])

            # Select examples with true-label being non-zero
            Y_test_pred = Y_test_pred[Y_test != 0].flatten()
            Y_test = Y_test[Y_test != 0].flatten()

            platt_est = PlattProbabilities().fit(Y_test_pred, Y_test)

            print(platt_est)
            print(platt_est.A, platt_est.B, platt_est.score(Y_test_pred, Y_test))

            # plot sigmoid
            xx_range = np.arange(np.min(Y_test_pred), np.max(Y_test_pred), 0.1)
            yy_range = sigmoid(xx_range, k=-platt_est.A, x_0=-(platt_est.B / platt_est.A))
            axrr[0, i].vlines(0.0, 0.0, 1.0, colors="black", linestyles="--", alpha=0.5)
            axrr[0, i].grid(axis="y")
            axrr[0, i].plot(xx_range, yy_range)

            # axrr[1, i].hlines(0.5, xx_range[0], xx_range[-1], colors="black", linestyles="--", alpha=0.5)
            axrr[1, i].vlines(0.0, 0.0, 1.0, colors="black", linestyles="--", alpha=0.5)
            axrr[1, i].grid(axis="y")
            axrr[1, i].scatter(Y_test_pred[Y_test == -1], platt_est.predict(Y_test_pred[Y_test == -1]), marker=".", color="red", alpha=0.1)
            sns.kdeplot(Y_test_pred[Y_test == -1], color="red", shade=True, cumulative=False, ax=axrr[1, i])

            # axrr[2, i].hlines(0.5, xx_range[0], xx_range[-1], colors="black", linestyles="--", alpha=0.5)
            axrr[2, i].vlines(0.0, 0.0, 1.0, colors="black", linestyles="--", alpha=0.5)
            axrr[2, i].grid(axis="y")
            axrr[2, i].scatter(Y_test_pred[Y_test == 1], platt_est.predict(Y_test_pred[Y_test == 1]), marker=".", color="blue", alpha=0.1)
            sns.kdeplot(Y_test_pred[Y_test == 1], color="blue", shade=True, cumulative=False, ax=axrr[2, i])

            axrr[0, i].hlines(0.5, axrr[2, i].get_xlim()[0], axrr[2, i].get_xlim()[1], colors="black", linestyles="--",
                              alpha=0.5)
            axrr[1, i].hlines(0.5, axrr[2, i].get_xlim()[0], axrr[2, i].get_xlim()[1], colors="black", linestyles="--",
                              alpha=0.5)
            axrr[2, i].hlines(0.5, axrr[2, i].get_xlim()[0], axrr[2, i].get_xlim()[1], colors="black", linestyles="--",
                              alpha=0.5)

            axrr[0, i].set_title("C=%.3f, A=%.3f, B=%.2f" % (C, platt_est.A, platt_est.B))
            axrr[1, i].set_title("Negative labels")
            axrr[2, i].set_title("Positive labels")
            axrr[2, i].set_xlabel("Preference score difference")
            if i == 0:
                axrr[0, i].set_ylabel("Prob. i elutes before j")
                axrr[1, i].set_ylabel("Prob. i elutes before j")
                axrr[2, i].set_ylabel("Prob. i elutes before j")

        plt.show()
        # plt.savefig("./platt_sigmoids_for_different_C.png")

    def test_with_parameter_selection(self):
        X, y = load_data(dataset="FEM_long")
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3112, test_size=0.2)

        # Calculate kernels
        K_train = minmax_kernel(X_train)
        K_test = minmax_kernel(X_train, X_test)

        ranksvm = KernelRankSVC(random_state=787, t_0=0.5,
                                step_size_algorithm="diminishing_2",
                                convergence_criteria="max_iter", max_iter=1000)

        est_best, l_params, param_scores_avg, param_scores, best_params, platt_est = \
            find_hparam_single_dataset(ranksvm, K_train, y_train, {"C": [0.0625, 1.0, 8.0]},
                                       estimate_platt_prob=True, cv=3, random_state=1010)


if __name__ == '__main__':
    unittest.main()
