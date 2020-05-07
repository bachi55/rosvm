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

'''

Collection of methods that can be used to estimate the model hyper-parameters
of the pairwise Support Vector Regression (SVR) and Ranking Support Vector Machine
(RankSVM).

'''

import numpy as np
import itertools as it
import logging as lg
LOG = lg.getLogger("ModelSelection")
LOG.addHandler(lg.NullHandler())
lg.basicConfig(format="[ModelSelection] %(message)s", level=lg.INFO)

from collections import OrderedDict

from scipy.stats import kendalltau

# Import some helper function from the sklearn package
from sklearn.model_selection._search import ParameterGrid, check_random_state
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RepeatedKFold
from sklearn.base import clone
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel
from sklearn.preprocessing import KernelCenterer

# Allow the paralellization of the parameter search
from joblib import Parallel, delayed

# Import the retention order class
from ranksvm.retentiongraph_cls import RetentionGraph

# Import stuff from the rank svm
from ranksvm.pair_utils import get_pairs_from_order_graph, get_pairs_single_dataset, get_pairs_multiple_datasets, \
    get_pairs_multiple_datasets_y_ds_separated
from ranksvm.kernel_utils import tanimoto_kernel, minmax_kernel
from ranksvm.mkl_utils import mkl_combine_kernels
from ranksvm.rank_svm_cls import KernelRankSVC
from ranksvm.platt_cls import PlattProbabilities

from rlscore.learner import PPRankRLS
from rlscore.measure import cindex
from rlscore.measure.measure_utilities import UndefinedPerformance


def find_hparam_ranksvm(estimator, X, y, h_param_grid, cv=3, pair_params=None, scalers=None, n_jobs=1,
                        fold_score_aggregation="weighted_average", all_pairs_as_test=True, random_state=None,
                        kernel_params=None, mkl_params=None):
    """
    Task: find the hyper-parameter from a set of parameters (param_grid),
          that performs best in an cross-validation setting for the given
          estimator.

    :param estimator: Estimator object, e.g. KernelRankSVC

    :param X: dictionary, (mol-id, system)-tuples as keys and molecular
              feature-vectors as values:

              Example:
                {("M1", "S1"): feat_11, ("M2", "S1"): feat_12, ...}

              In the MKL case: feat_ij could be list of np.ndarray, where each
              list element corresponds to a feature representation of the molecules.

    :param y: dictionary, (mol-id, system)-tuples as keys and retention
              times as values

              Example:
                {("M1", "S1"): rt_11, ("M2", "S1"): rt_12, ...}

    :param h_param_grid: dictionary, defining the grid-search space
        "C": Trade-of parameter for the SVM
        "gamma": width of the rbf/gaussian kernel
        ... etc. ...

        Example:
            {"C": [0.1, 1, 10], "gamma": [0.1, 0.25, 0.5, 1]}

    :param cv: cross-validation generator or scalar, see sklearn package, must be
               either a GroupKFold or GroupShuffleSplit object. If it is a scalar
               than a GroupKFold with 'scalar' splits is initialized. (default = 3)

    :param pair_params: dictionary, specifying parameters for the order graph:
        "ireverse": scalar, Should cross-system elution transitivity be included
            0: no, 1: yes
        "d_lower": scalar, minimum distance of two molecules in the elution order graph
                   to be considered as a pair.
        "d_upper": scalar, maximum distance of two molecules in the elution order graph
                   to be considered as a pair.
        "allow_overlap": scalar, Should overlap between the upper and lower sets
                         be allowed. Those overlaps originate from retention order
                         contradictions between the different systems.
        (default = None --> ECCB parameters are used)

    :param scalers: scaler object, per feature scaler, e.g. MinMaxScaler

    :param n_jobs: integer, number of jobs run in parallel. Parallelization is performed
        over the cv-folds. (default = 1)

    :param fold_score_aggregation: string, (default = "weighted_average")

    :param all_pairs_as_test: boolean, should all possible pairs (d_lower = 0, d_upper = np.inf)
        be used during the test. If 'False' than corresponding values are taking from the
        'pair_params' dictionary. (default = True)

    :param random_state: random state used to shuffle the molecules in case of GroupKFold.

    :param kernel_params: list of dictionaries, containing the desired kernels the corresponding
        kernel parameters for each input feature representation. This parameter is mainly intended
        to be used for MKL experiments.

        Example:
            [{"kernel": "tanimoto"}, {"kernel": "rbf", "gamma": "auto"}, ...]

    :param mkl_params: list of dictionaries, containing the kernel weights and kernel centerer objects.
        The centerer can be None, if no centering is needed.

        Example:
            [{"weight": scalar, "centerer": KernelCenterer}, ...]

    :return: dictionary, containing combination of best parameters
                Example:
                    {"C": 1, "gamma": 0.25}

             dictionary, all parameter combinations with corresponding scores
                 Example:
                    [{"C": 1, "gamma": 0.25, "score": 0.98},
                     {"C": 1, "gamma": 0.50, "score": 0.94},
                     ...]

             scalar, number of pairs used to train the final model

             estimator object, fitted using the best parameters
    """
    if np.isscalar(cv):
        cv = GroupKFold(n_splits=cv)

    if not (isinstance(cv, GroupKFold) or isinstance(cv, GroupShuffleSplit)):
        raise ValueError("Cross-validation generator must be either of "
                         "class 'GroupKFold' or 'GroupShuffleSplit'. "
                         "Provided class is '%s'." % cv.__class__.__name__)

    if not (isinstance(X, OrderedDict) or isinstance(y, OrderedDict)):
        raise ValueError("Features (or kernel values) and retention times must be "
                         "provided as OrderedDicts.")

    if X.keys() != y.keys():
        raise ValueError("Keys-set for features and retentions times must be equal.")

    if pair_params is None:
        # Default parameters used for the ECCB publication
        pair_params = {"d_lower": 0, "d_upper": 16, "allow_overlap": True, "ireverse": False}

    # Configure test set pairs
    if all_pairs_as_test:
        d_lower_test = 0
        d_upper_test = np.inf
    else:
        d_lower_test = pair_params["d_lower"]
        d_upper_test = pair_params["d_upper"]

    # Make a list of all combinations of parameters
    l_params = list(ParameterGrid(h_param_grid))
    param_scores = np.zeros((len(l_params),))

    # Get all (mol-id, system)-tuples used for the parameter search
    keys = list(X.keys())

    if isinstance(cv, GroupKFold):
        keys = shuffle(keys, random_state=check_random_state(random_state))

    # Determine the number of feature representations per molecule. This is
    # for example used to determine the number of kernels in the MKL case.
    n_feat_rep_per_mol = _get_number_of_feat_rep(X)
    if (n_feat_rep_per_mol > 1) and (estimator.kernel != "precomputed"):
        raise ValueError("If more than one feature representation is given, the kernels "
                         "will be precomputed and this should be specified in the estimator.")

    if scalers is None:
        scalers = [None] * n_feat_rep_per_mol
    else:
        raise NotImplementedError("Currently we do not support the feature scalers. This "
                                  "might, however, become interesting for the molecular "
                                  "descriptors again.")

    if len(l_params) > 1:
        mol_ids = list(zip(*keys))[0]
        cv_splits = cv.split(mol_ids, groups=mol_ids)

        # Precompute the training / test pairs to save computation time as
        # we do not need to repeat this for several parameter settings.
        pairs_train_sets, pairs_test_sets = [], []
        X_train_sets, X_test_sets = [], []
        n_pairs_test_sets = []

        print("Get pairs for hparam estimation: ", end="", flush=True)
        for k_cv, (train_set, test_set) in enumerate(cv_splits):
            print("%d " % k_cv, end="", flush=True)

            # 0) Get keys (mol-id, system)-tuples, corresponding to the training
            #    and test sets.
            keys_train = [keys[idx] for idx in train_set]
            keys_test = [keys[idx] for idx in test_set]

            # Check for overlap of molecular ids, e.g. InChIs. Between training and test
            # molecular ids should not be shared, e.g. if they appear in different systems
            # at the same time.
            mol_ids_train = [mol_ids[idx] for idx in train_set]
            mol_ids_test = [mol_ids[idx] for idx in test_set]

            if set(mol_ids_train) & set(mol_ids_test):
                if isinstance(cv, GroupKFold) or isinstance(cv, GroupShuffleSplit):
                    raise RuntimeError("As grouped cross-validation is used the training "
                                       "and test molecules, i.e. mol_ids, are not allowed "
                                       "to overlap. This can happen if molecular structures "
                                       "are appearing in different systems. During the "
                                       "learning of hyper-parameter the training set should "
                                       "not contain any structure also in the test set.",
                                       set(mol_ids_train) & set(mol_ids_test))
                else:
                    print("Training and test keys overlaps.", set(mol_ids_train) & set(mol_ids_test))

            # 1) Extract the target values from y (train and test) using the keys
            y_train, y_test = OrderedDict(), OrderedDict()
            for key in keys_train:
                y_train[key] = y[key]
            for key in keys_test:
                y_test[key] = y[key]

            # 2) Calculate the pairs (train and test)
            cretention_train, cretention_test = RetentionGraph(), RetentionGraph()

            #   a) load 'lrows' in the RetentionGraph
            cretention_train.load_data_from_target(y_train)
            cretention_test.load_data_from_target(y_test)

            #   b) build the digraph
            cretention_train.make_digraph(ireverse=pair_params["ireverse"])
            cretention_test.make_digraph(ireverse=pair_params["ireverse"])

            #   c) find the upper and lower set
            cretention_train.dmolecules_inv = cretention_train.invert_dictionary(cretention_train.dmolecules)
            cretention_train.dcollections_inv = cretention_train.invert_dictionary(cretention_train.dcollections)
            cretention_test.dmolecules_inv = cretention_test.invert_dictionary(cretention_test.dmolecules)
            cretention_test.dcollections_inv = cretention_test.invert_dictionary(cretention_test.dcollections)

            #   d) get the pairs from the upper and lower sets
            pairs_train = get_pairs_from_order_graph(cretention_train, keys_train,
                                                     allow_overlap=pair_params["allow_overlap"], n_jobs=n_jobs,
                                                     d_lower=pair_params["d_lower"], d_upper=pair_params["d_upper"])
            pairs_train_sets.append(pairs_train)

            pairs_test = get_pairs_from_order_graph(cretention_test, keys_test,
                                                    allow_overlap=pair_params["allow_overlap"], n_jobs=n_jobs,
                                                    d_lower=d_lower_test, d_upper=d_upper_test)
            pairs_test_sets.append(pairs_test)
            n_pairs_test_sets.append(len(pairs_test))

            # 3) Extract the features from X (train and test) using the keys
            X_train_sets.append(
                [np.array([X[key][idx_feat_rep] for key in keys_train]) for idx_feat_rep in range(n_feat_rep_per_mol)])
            X_test_sets.append(
                [np.array([X[key][idx_feat_rep] for key in keys_test]) for idx_feat_rep in range(n_feat_rep_per_mol)])

        print("")

        # Iterate over all cv-fold and parameter combinations:
        # (consider: product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy)
        #
        #   [({'C': 1}, 0), ({'C': 1}, 1), ..., ({'C': 10}, 0), ({'C': 10}, 1), ...],
        #
        # with resulting score list:
        #
        #   [({'C': 1}, 0) : 0.1, ({'C': 1}, 1): 0.2, ..., ({'C': 10}, 0): 0.3, ({'C': 10}, 1) : 0.1, ...]
        fold_scores = Parallel(n_jobs=n_jobs, verbose=False)(
            delayed(_fit_and_score_ranksvm)(param.copy(), clone(estimator),
                                            X_train_sets[k_cv], X_test_sets[k_cv],
                                            pairs_train_sets[k_cv], pairs_test_sets[k_cv],
                                            scalers, kernel_params, mkl_params)
            for param, k_cv in it.product(l_params, range(cv.get_n_splits())))

        # we can reshape this result into a matrix, shape = (n_param, n_fold)
        #
        #   [0.1, 0.2,
        #    0,3, 0.1]
        fold_scores = np.array(fold_scores).reshape((len(l_params), cv.get_n_splits()))
        print(fold_scores)

        # and calculate the average for each parameter across the folds.
        if fold_score_aggregation == "average":
            param_scores = np.mean(fold_scores / np.array(n_pairs_test_sets), axis=1)
        elif fold_score_aggregation == "weighted_average":
            param_scores = np.sum(fold_scores, axis=1) / np.sum(n_pairs_test_sets)
        else:
            raise ValueError("Invalid fold-scoring aggregation: %s." % fold_score_aggregation)

    # Fit model using the best parameters
    # Find the best params
    best_params = l_params[np.argmax(param_scores)].copy()

    # Fit the model using the best parameters
    best_estimator = clone(estimator)
    best_estimator.set_params(**_filter_params(best_params, best_estimator))

    # Build retention order graph
    cretention = RetentionGraph()
    cretention.load_data_from_target(y)
    cretention.make_digraph(ireverse=pair_params["ireverse"])
    cretention.dmolecules_inv = cretention.invert_dictionary(cretention.dmolecules)
    cretention.dcollections_inv = cretention.invert_dictionary(cretention.dcollections)

    pairs = get_pairs_from_order_graph(cretention, keys, allow_overlap=pair_params["allow_overlap"],
                                       n_jobs=n_jobs, d_lower=pair_params["d_lower"],
                                       d_upper=pair_params["d_upper"])
    n_pairs_train = len(pairs)

    # Extract features and transform if needed
    X_all = [np.array([X[key][idx_feat_rep] for key in keys]) for idx_feat_rep in range(n_feat_rep_per_mol)]

    # TODO: Handle scalers properly.
    # for idx_scaler, scaler in enumerate(scalers):
    #     if scaler is not None:
    #         X_all[idx_scaler] = scaler.transform(X_all[idx_scaler])

    # Calculate the kernels and combine if needed
    if estimator.kernel == "precomputed":
        X_all = _precompute_kernels(X_all, None, kernel_params)

        # Fit the KernelCenterers if needed
        for idx in range(len(X_all)):
            if mkl_params[idx]["centerer"] is not None:
                if not isinstance(mkl_params[idx]["centerer"], KernelCenterer):
                    raise ValueError("Centerer must of class KernelCenterer.")

                mkl_params[idx]["centerer"].fit(X_all[idx])

        # Calculate kernel combination
        X_all = mkl_combine_kernels(X_all, mkl_params)
    else:
        assert (n_feat_rep_per_mol == 1), "Ups!"
        X_all = X_all[0]

    # TODO: Handle alpha initialization, e.g. using alpha_ij = C / n_pairs_train
    best_estimator.fit(X_all, pairs, pairwise_labels=None, alpha_init=None)

    # Combine the mean fold scores with the list of parameter sets
    for k_param, _ in enumerate(l_params):
        l_params[k_param]["score"] = param_scores[k_param]

    return best_params, l_params, n_pairs_train, best_estimator, kernel_params, mkl_params


def _fit_and_score_ranksvm(param, estimator, X_train, X_test, pairs_train, pairs_test, scalers,
                           kernel_params, mkl_params):
    """
    Function to fit the estimator with training data and a set of parameters. The test set score
    is returned. Used for hyper-parameter selection.

    :return: scalar, number of correctly classified pairs.
    """
    # TODO: Handle scalers properly.
    # if scalers is not None:
    #     X_train = scalers.transform(X_train)
    #     X_test = scalers.transform(X_test)

    # Calculate the kernels and combine if needed
    if estimator.kernel == "precomputed":
        KX_test = _precompute_kernels(X_train, X_test, kernel_params)
        KX_train = _precompute_kernels(X_train, None, kernel_params)

        # Fit the KernelCenterers if needed
        for idx in range(len(X_train)):
            # TODO: I think we need to clone the centerer.
            if mkl_params[idx]["centerer"] is not None:
                if not isinstance(mkl_params[idx]["centerer"], KernelCenterer):
                    raise ValueError("Centerer must of class KernelCenterer.")

                mkl_params[idx]["centerer"].fit(KX_train)

        # Calculate kernel combination
        KX_train = mkl_combine_kernels(KX_train, mkl_params)
        KX_test = mkl_combine_kernels(KX_test, mkl_params)
    else:
        KX_train = X_train[0]
        KX_test = X_test[0]

    # Update the estimators parameters to the current ones
    estimator.set_params(**_filter_params(param, estimator))
    estimator.fit(KX_train, pairs_train, pairwise_labels=None, alpha_init=None)

    return estimator.score(KX_test, pairs_test, normalize=False)


def _filter_params(param, estimator):
    """
    Given a set of parameters and an estimator: Remove the parameters
    that are not belonging to the estimator.

    :return: dictionary, containing only the supported parameters
    """
    valid_params = estimator.get_params()
    out = dict()
    for key, value in param.items():
        if key in valid_params.keys():
            out[key] = value
    return out


def _get_number_of_feat_rep(X):
    """
    :param X:
    :return: scalar, number of feature vectors per molecule
    """
    n_feat_rep_per_mol = 0

    # Check structure of X
    assert (isinstance(X, OrderedDict)), "Features must be provided in an _OrderedDict_."
    for idx, x_l in enumerate(X.values()):
        assert (isinstance(x_l, list)), "Feature vectors must be stored as _list_ of feature vectors."
        if idx > 0:
            assert (n_feat_rep_per_mol == len(x_l)), "Number of feature vectors must be equal for all molecules."

        n_feat_rep_per_mol = len(x_l)

        for x in x_l:
            assert (isinstance(x, np.ndarray)), "Individual feature vectors must be stored as _np.array_."

    assert (n_feat_rep_per_mol > 0), "Number of feature vectors per molecule must be greater zero!"

    return n_feat_rep_per_mol


def _precompute_kernels(X1_l, kernel_params, X2_l=None):
    """
    Pre-compute kernels for a set of feature representations.

    :param X1_l: list of array-like, length = n_feature_rep, list of feature representation
        matrices with shape = (n_samples_A, n_features_i).

    :param kernel_params: list of dictionaries, length = n_feature_rep, each dictionary contains
        the kernel (parameters) of the individual feature representations

    :param X2_l: list of array-like, length = n_feature_rep, list of feature representation
        matrices with shape = (n_samples_B, n_features_i).

        OR

        None, than X2_l = X1_l                                                  (default = None)

    :return: list of array-like, length = n_feature_rep, list of kernel matrices with shape
        = (n_samples_A, n_samples_B)
    """
    # Determine number of feature representations
    n_feature_rep = len(X1_l)

    # Handle input
    if X2_l is None:
        X2_l = X1_l

    if n_feature_rep != len(X2_l):
        raise ValueError("Number of feature representations for both examples sets must be equal.")

    if n_feature_rep != len(kernel_params):
        raise ValueError("Number of kernel parameters does not match the number of feature representations.")

    # Calculate kernels
    K_l = []
    for X1, X2, kernel_param in zip(X1_l, X2_l, kernel_params):
        if kernel_param["kernel"] == "rbf":
            K_l.append(rbf_kernel(X1, X2, gamma=kernel_param.get("gamma", None)))
        elif kernel_param["kernel"] == "linear":
            K_l.append(linear_kernel(X1, X2))
        elif kernel_param["kernel"] == "tanimoto":
            K_l.append(tanimoto_kernel(X1, X2))
        elif kernel_param["kernel"] == "minmax":
            K_l.append(minmax_kernel(X1, X2))
        elif kernel_param["kernel"] in ["precomputed", "PrecomputedKernel"]:
            K_l.append(X1)  # pass through the kernel
        else:
            raise ValueError("Unsupported kernel: %s" % kernel_param["kernel"])

    return K_l


def find_hparam_multiple_datasets(est, X, y, mol, ds, param_grid, kernel_params=None, mkl_params=None, cv=10,
                                  pair_params=None, random_state=None, estimate_platt_prob=False):
    """
    Hyper parameter grid-search for order predictors, i.e. KernelRanKSVC and PPRankRLS. It supports
    multiple feature representations and multiple kernel learning, e.g. by weighted averages of the kernels.

    :param est:
    :param X:
    :param y:
    :param param_grid:
    :param kernel_params:
    :param mkl_params:
    :param cv:
    :param random_state:
    :param pair_params:
    :param pw_conf_fun:
    :return:
    """
    # Determine number of feature representations
    n_feat_rep = _determine_number_of_feature_representations(X)

    # Handle the different order predictors
    est_name, est_kernel = _handle_different_order_predictors(est, param_grid)

    # Check input parameters for kernels and MKL
    if n_feat_rep > 1:
        _check_kernel_and_mkl_params(mkl_params, kernel_params, est_kernel, n_feat_rep)

    if np.isscalar(cv):
        # Configure a GroupKFold cross-validation
        # We grouped kfold here, as we want to estimate the hyper-parameters using leave-molecule-out (LMO)
        cv = GroupKFold(n_splits=cv)
        # cv = GroupShuffleSplit(n_splits=cv, random_state=random_state, test_size=0.33) # TODO: would GroupShuffleSplit be better here?

    if pair_params is None:
        # Take parameters from ECCB publication
        pair_params = {"d_upper": 16, "d_lower": 0}

    # Set up parameter grid
    l_params = list(ParameterGrid(param_grid))
    param_scores = {"cindex": np.zeros((len(l_params), cv.get_n_splits())),
                    "cindex_ds_sep": np.zeros((len(l_params), cv.get_n_splits())),
                    "kendall_ds_sep": np.zeros((len(l_params), cv.get_n_splits()))}

    if n_feat_rep > 1:
        # Pre-compute and combine kernels
        KX = mkl_combine_kernels(_precompute_kernels(X, kernel_params), mkl_params)
    else:
        # Pass through kernel- respectively feature-matrix
        KX = X

    # Collect test-pair labels and predictions for
    platt_data = {"f": [np.array([]) for _ in l_params], "y": [np.array([]) for _ in l_params]}

    # Iterate over splits and parameters
    for fold_idx, (params, (train, test)) in enumerate(it.product(l_params, cv.split(y, groups=mol))):
        # Get train and test molecular structures to check the the grouping
        assert (len(set([mol[__i] for __i in train]) & set([mol[__i] for __i in test])) == 0), \
            "No training molecular structure should be in the test set."

        # Extract train and test data
        if n_feat_rep > 1:  # several features
            KX_train = KX[np.ix_(train, train)]
            KX_test = KX[np.ix_(train, test)]
        else:  # single feature
            KX_train = KX[np.ix_(train, train)] if est_kernel in ["precomputed", "PrecomputedKernel"] else KX[train]
            KX_test = KX[np.ix_(train, test)] if est_kernel in ["precomputed", "PrecomputedKernel"] else KX[test]

        # Get train and test targets
        y_train = y[train]
        y_test = y[test]

        # Get train and test dataset assignment of individual examples
        ds_train = np.array([ds[__i] for __i in train])
        ds_test = np.array([ds[__i] for __i in test])

        # Get training pairs
        p_train = get_pairs_multiple_datasets_y_ds_separated(
            y_train, ds_train, d_lower=pair_params["d_lower"], d_upper=pair_params["d_upper"])

        # Train order predictor
        if est_name == "KernelRankSVC":
            # Get "fresh" estimator and set parameters
            est_fold = clone(est)
            est_fold.set_params(**params)
            est_fold.fit(KX_train, p_train, alpha_init=(est_fold.C / len(p_train)))
        else:
            # PPRankRLS
            p_train_start = list(zip(*p_train))[1]
            p_train_end = list(zip(*p_train))[0]
            est_fold = PPRankRLS(KX_train, p_train_start, p_train_end, kernel=est_kernel,
                                 gamma=params.get("gamma", 1.0), regparam=params["regparam"])

        # Calculate score on test set
        fold_idx_tuple = np.unravel_index(fold_idx, (len(l_params), cv.get_n_splits()))

        if est_name == "KernelRankSVC":
            y_test_pred = est_fold.map_values(KX_test)
        else:
            # PPRankRLS
            y_test_pred = est_fold.predict(KX_test.T)[:, np.newaxis]

        # Get pairwise prediction matrix
        assert (y_test_pred.shape == (len(test), 1))
        Y_test_pred = np.sign(- y_test_pred + y_test_pred.T)
        assert (Y_test_pred.shape == (len(test), len(test)))
        assert (not np.all(Y_test_pred == 0)), "Model outputs constant prediction!"

        # Calculate Cindex using pairs from the test set. Those are only considering pairs within the
        # individual datasets.
        p_test = get_pairs_multiple_datasets_y_ds_separated(y_test, ds_test)
        param_scores["cindex"][fold_idx_tuple] = KernelRankSVC.score_pairwise_using_prediction(Y_test_pred, p_test)

        if estimate_platt_prob:
            _collect_platt_data(platt_data, y_test_pred, y_test, p_test, fold_idx_tuple[0])

        # Calculate Cindex and Kendalltau as average of the dataset specific performance
        _avg_cnt = 0
        for _ds_i in np.unique(ds_test):
            y_test_ds = y_test[ds_test == _ds_i]
            if len(y_test_ds) == 0:
                continue  # skip empty datasets

            # FIXME: Why do HILIC systems cause an exception here?
            try:
                param_scores["cindex_ds_sep"][fold_idx_tuple] += cindex(y_test_ds, y_test_pred[ds_test == _ds_i])
            except UndefinedPerformance:
                param_scores["cindex_ds_sep"][fold_idx_tuple] += np.nan

            param_scores["kendall_ds_sep"][fold_idx_tuple] += kendalltau(y_test_ds, y_test_pred[ds_test == _ds_i])[0]

            _avg_cnt += 1

        param_scores["cindex_ds_sep"][fold_idx_tuple] /= _avg_cnt
        param_scores["kendall_ds_sep"][fold_idx_tuple] /= _avg_cnt

    # Determine best parameter setting:
    param_scores_avg = param_scores["cindex"].mean(axis=1)  # calculate average cindex-score per split / fold
    idx_best_param = np.argmax(param_scores_avg)
    best_params = l_params[idx_best_param]

    # Get Platt probability estimates based on the test set predictions
    if estimate_platt_prob:
        platt_est = PlattProbabilities().fit(platt_data["f"][idx_best_param], platt_data["y"][idx_best_param])
        LOG.info("Platt: A=%f, B=%f, min_fx=%f, max_fx=%f" % (
            platt_est.A, platt_est.B, np.min(platt_data["f"][idx_best_param]), np.max(platt_data["f"][idx_best_param])))

    # Fit estimator with all data and best parameters
    p = get_pairs_multiple_datasets_y_ds_separated(y, ds, d_lower=pair_params["d_lower"],
                                                   d_upper=pair_params["d_upper"])

    if est_name == "KernelRankSVC":
        est_best = clone(est)
        est_best.set_params(**best_params)
        est_best.fit(KX, p, alpha_init=(est_best.C / len(p)))
    else:  # PPRankRLS
        p_start = list(zip(*p))[1]
        p_end = list(zip(*p))[0]
        est_best = PPRankRLS(KX, p_start, p_end, kernel=est_kernel,
                             gamma=best_params.get("gamma", 1.0), regparam=best_params["regparam"])

    if estimate_platt_prob:
        return est_best, l_params, param_scores_avg, param_scores, best_params, platt_est
    else:
        return est_best, l_params, param_scores_avg, param_scores, best_params


def find_hparam_single_dataset(est, X, y, param_grid, kernel_params=None, mkl_params=None, cv=3, random_state=None,
                               pair_params=None, pw_conf_fun=None, estimate_platt_prob=False):
    """
    Hyper parameter grid-search for order predictors, i.e. KernelRanKSVC and PPRankRLS. It supports
    multiple feature representations and multiple kernel learning, e.g. by weighted averages of the kernels.
    """
    # Determine number of feature representations
    n_feat_rep = _determine_number_of_feature_representations(X)

    # Handle the different order predictors
    est_name, est_kernel = _handle_different_order_predictors(est, param_grid)

    # Check input parameters for kernels and MKL
    if n_feat_rep > 1:
        _check_kernel_and_mkl_params(mkl_params, kernel_params, est_kernel, n_feat_rep)

    if np.isscalar(cv):
        # Configure a repeated KFold cross-validation
        cv = RepeatedKFold(n_splits=cv, n_repeats=3, random_state=random_state)

    if pair_params is None:
        # Take parameters from ECCB publication
        pair_params = {"d_upper": 16, "d_lower": 0}

    # Set up parameter grid
    l_params = list(ParameterGrid(param_grid))
    param_scores = {"cindex": np.zeros((len(l_params), cv.get_n_splits())),
                    "kendall": np.zeros((len(l_params), cv.get_n_splits()))}

    if n_feat_rep > 1:
        # Pre-compute and combine kernels
        KX = mkl_combine_kernels(_precompute_kernels(X, kernel_params), mkl_params)
    else:
        # Pass through kernel- respectively feature-matrix
        KX = X

    # Collect test-pair labels and predictions for
    platt_data = {"f": [np.array([]) for _ in l_params], "y": [np.array([]) for _ in l_params]}

    # Iterate over splits and parameters
    for fold_idx, (params, (train, test)) in enumerate(it.product(l_params, cv.split(y))):
        # Extract train and test data
        if n_feat_rep > 1:  # several features
            KX_train = KX[np.ix_(train, train)]
            KX_test = KX[np.ix_(train, test)]
        else:  # single feature
            KX_train = KX[np.ix_(train, train)] if est_kernel in ["precomputed", "PrecomputedKernel"] else KX[train]
            KX_test = KX[np.ix_(train, test)] if est_kernel in ["precomputed", "PrecomputedKernel"] else KX[test]

        y_train = y[train]
        y_test = y[test]

        # Get train and test pairs
        p_train, p_conf_train = get_pairs_single_dataset(
            y_train, d_lower=pair_params["d_lower"], d_upper=pair_params["d_upper"], pw_conf_fun=pw_conf_fun)

        # Train order predictor
        if est_name == "KernelRankSVC":
            # Get "fresh" estimator and set parameters
            est_fold = clone(est)
            est_fold.set_params(**params)
            est_fold.fit(KX_train, p_train, alpha_init=(est_fold.C / len(p_train)), pairwise_confidences=p_conf_train)
        else:
            # PPRankRLS
            p_train_start = list(zip(*p_train))[1]
            p_train_end = list(zip(*p_train))[0]
            est_fold = PPRankRLS(KX_train, p_train_start, p_train_end, kernel=est_kernel,
                                 gamma=params.get("gamma", 1.0), regparam=params["regparam"])

        # Calculate score on test set
        fold_idx_tuple = np.unravel_index(fold_idx, (len(l_params), cv.get_n_splits()))

        if est_name == "KernelRankSVC":
            y_test_pred = est_fold.map_values(KX_test)
        else:
            # PPRankRLS
            y_test_pred = est_fold.predict(KX_test.T)

        # Get pairwise predictions for the Platt estimates
        if estimate_platt_prob:
            p_test, _ = get_pairs_single_dataset(y_test)
            _collect_platt_data(platt_data, y_test_pred, y_test, p_test, fold_idx_tuple[0])

        param_scores["cindex"][fold_idx_tuple] = cindex(y_test, y_test_pred)
        param_scores["kendall"][fold_idx_tuple] = kendalltau(y_test, y_test_pred)[0]

    # Determine best parameter setting:
    param_scores_avg = param_scores["cindex"].mean(axis=1)  # calculate average cindex-score per split / fold
    idx_best_param = np.argmax(param_scores_avg)
    best_params = l_params[idx_best_param]

    # Get Platt probability estimates based on the test set predictions
    if estimate_platt_prob:
        platt_est = PlattProbabilities().fit(platt_data["f"][idx_best_param], platt_data["y"][idx_best_param])
        LOG.info("Platt: A=%f, B=%f, min_fx=%f, max_fx=%f" % (
            platt_est.A, platt_est.B, np.min(platt_data["f"][idx_best_param]), np.max(platt_data["f"][idx_best_param])))

    # Fit estimator with all data and best parameters
    p, p_conf = get_pairs_single_dataset(y, d_lower=pair_params["d_lower"], d_upper=pair_params["d_upper"],
                                         pw_conf_fun=pw_conf_fun)

    if est_name == "KernelRankSVC":
        est_best = clone(est)
        est_best.set_params(**best_params)
        est_best.fit(KX, p, alpha_init=(est_best.C / len(p)), pairwise_confidences=p_conf)
    else:  # PPRankRLS
        p_start = list(zip(*p))[1]
        p_end = list(zip(*p))[0]
        est_best = PPRankRLS(KX, p_start, p_end, kernel=est_kernel,
                             gamma=best_params.get("gamma", 1.0), regparam=best_params["regparam"])

    if estimate_platt_prob:
        return est_best, l_params, param_scores_avg, param_scores, best_params, platt_est
    else:
        return est_best, l_params, param_scores_avg, param_scores, best_params


# -------------------------------------------
# Utility functions
# -------------------------------------------
def _collect_platt_data(platt_data, y_test_pred, y_test, pairs_test, param_idx):
    _Y_test_pred = []
    _Y_test = []
    for i, j in pairs_test:
        _Y_test_pred.append((y_test_pred[j] - y_test_pred[i]).item())
        _Y_test.append(np.sign(y_test[j] - y_test[i]))

        _Y_test_pred.append((y_test_pred[i] - y_test_pred[j]).item())
        _Y_test.append(np.sign(y_test[i] - y_test[j]))

    # _Y_test_pred = - y_test_pred + y_test_pred.T  # shape=(n_test, n_test), [ij] = w . (phi_j - phi_i)
    # _Y_test = np.sign(- y_test[:, np.newaxis] + y_test[np.newaxis, :])  # shape=(n_test, n_test), [ij] = sign(t_j - t_i)

    # _Y_test_pred = _Y_test_pred.flatten()
    # _Y_test = _Y_test.flatten()

    # Remove elements for which i = j ==> t_i = t_j
    # _Y_test_pred = _Y_test_pred[_Y_test != 0]
    # _Y_test = _Y_test[_Y_test != 0]

    platt_data["f"][param_idx] = np.hstack((platt_data["f"][param_idx], np.array(_Y_test_pred)))
    platt_data["y"][param_idx] = np.hstack((platt_data["y"][param_idx], np.array(_Y_test)))


def _determine_number_of_feature_representations(X):
    """
    Function to determine the number of feature-representations, e.g. number of fingerprint sets.

    :param X: array-like or list of array-like, shape = (N, D) or length = number of feature
        representations, either a feature- / kernel-matrix or list of those.

    :return: scalar, number of feature representations.
    """
    # Determine number of feature representations
    if isinstance(X, np.ndarray):
        n_feat_rep = 1
    elif isinstance(X, list):
        n_feat_rep = len(X)
    else:
        raise ValueError("Features must either be given as feature matrix or as list of feature matrices.")

    return n_feat_rep


def _handle_different_order_predictors(est, param_grid):
    """
    Function to check and handle the different supported order predictors.

    :param est:
    :param param_grid: dict, parameters of the given estimator

    :return:
    """
    if isinstance(est, KernelRankSVC):
        est_name = "KernelRankSVC"
        est_kernel = est.kernel
    elif isinstance(est, dict) and est.get("name", None) == "PPRankRLS":
        est_name = "PPRankRLS"
        est_kernel = est["kernel"]
        if "regparam" not in param_grid:
            raise ValueError("For PPRankRLS the regularization parameter is specified using 'regparam' in the "
                             "'param_grid'.")
    else:
        raise ValueError("Invalid order predictor:", est)

    return est_name, est_kernel


def _check_kernel_and_mkl_params(mkl_params, kernel_params, est_kernel, n_feat_rep):
    """

    :param mkl_params:
    :param kernel_params:
    :param est_kernel:
    :param n_feat_rep:
    :return:
    """
    if mkl_params is None:
        raise ValueError("MKL parameters must be given, if several features are provided.")

    if kernel_params is None:
        raise ValueError("Kernel parameters must be given, if several feature are provided.")

    if est_kernel not in ["precomputed", "PrecomputedKernel"]:
        raise ValueError(
            "Estimator must be configured to use precomputed kernels, if several feature are provided.")

    if n_feat_rep != len(mkl_params):
        raise ValueError("Number of MKL parameters does not match the number of feature representations.")

    if n_feat_rep != len(kernel_params):
        raise ValueError("Number of kernel parameters does not match the number of feature representations.")
