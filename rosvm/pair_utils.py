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

import itertools
import numpy as np

from collections import OrderedDict
from joblib import Parallel, delayed
from scipy import stats as sp_stats


def get_pairs_from_order_graph(cretention, keys, allow_overlap, d_lower, d_upper, n_jobs=1):
    """
    Task: Get the pairs (i,j) with i elutes before j for the learning / prediction process
          given a set of input features with corresponding targets.

    :param cretention: RetentionGraph object, representation of the order graph.

    :param keys: list, of (mol-id, system)-tuples. This list defines the order,
                 in which the feature vectors are stored in the feature matrix X.
                 The pair indices (i,j) are the indices at which a particular
                 key is found in the 'keys' list.

                 Example:
                 For the difference features the row j of X is subtracted from
                 the row i of X: phi_ij = X[j] - X[i] for pair (i,j).

    :param allow_overlap: binary, should overlaps between the upper and lowers sets be
                          allowed. Those overlaps represent contradictions in the
                          elution orders between different systems.

    :param d_lower: scalar, minimum distance of two molecules in the elution order graph
                    to be considered as a pair.

    :param d_upper: scalar, maximum distance of two molecules in the elution order graph
                    to be considered as a pair.

    :param n_jobs: scalar, number of jobs to run pair extraction in parallel. (default = 1)

    :return: array-like, shape = (n_pairs,), list of tuples
         [(i,j),(j,k),...], with i elutes before j and j elutes before k, ...

         i,j,k are scalars that correspond to the particular index of the
         targets values (in 'targets'). So we need to keep the target values
         and the feature vectors in the same order!
    """

    # Get upper and lower sets of the nodes
    d_moleculecut = cretention.upper_lower_set_node(cretention.dG)

    # For each node in the graph and its corresponding upper and lower set do:
    pairs = Parallel(n_jobs=n_jobs)(delayed(_get_pairs_for_node)(
        node, ul_set_nodes, keys, cretention.dmolecules_inv,
        cretention.dcollections_inv, allow_overlap, d_lower, d_upper)
                                       for node, ul_set_nodes in d_moleculecut.items())

    # Flatten the result
    pairs = [pair for sublist in pairs for pair in sublist]

    # Due to the symmetry of the upper and lower sets we count every pairwise relation twice:
    #   For examples: Given A->B, B->C, we have C in L(A) ==> (A,C) and A in U(C) ==> (A,C)
    pairs = list(set(pairs))

    return pairs


def _get_pairs_for_node(node, ul_set_node, keys, dmolecules_inv, dcollections_inv, allow_overlap,
                        d_lower, d_upper):
    """
    Task: Get all the pairs for a particular node in the order graph.

    :param node: (mol-id, system)-tuple, node to process

    :param ul_set_node: dictionary, upper and lower set of the current node.

    :param keys: list, of (mol-id, system)-tuples. This list defines the order,
                 in which the feature vectors are stored in the feature matrix X.
                 The pair indices (i,j) are the indices at which a particular
                 key is found in the 'keys' list.

                 Example:
                 For the difference features the row j of X is subtracted from
                 the row i of X: phi_ij = X[j] - X[i] for pair (i,j).

    :param dmolecules_inv:
    :param dcollections_inv:

    :param allow_overlap: binary, should overlaps between the upper and lowers sets be
                          allowed. Those overlaps represent contradictions in the
                          elution orders between different systems.

    :param d_lower: scalar, minimum distance of two molecules in the elution order graph
                    to be considered as a pair.

    :param d_upper: scalar, maximum distance of two molecules in the elution order graph
                    to be considered as a pair.

    :return:
    """
    def _find_in(x, l, only_first_occurrence=True):
        """
        Task: Find all indices in the list l where the value l[i] equals x.

        :param x: object, value to search for

        :param l: list, list of objects

        :param only_first_occurrence: boolean, returned list only contains the
                                      index of the first occurrence of the searched
                                      item.

        :return: list of indices, at which x is found in l.
        """
        if only_first_occurrence:
            for i, xi in enumerate(l):
                if xi == x:
                    return [i]
        else:
            return [i for i, xi in enumerate(l) if xi == x]

    pairs = []

    row_i = _find_in((dmolecules_inv[node[0]], dcollections_inv[node[1]]), keys)
    assert (len(row_i) == 1)

    if allow_overlap:
        u_set_node = ul_set_node[0]
        l_set_node = ul_set_node[1]
    else:
        # Remove overlap between upper and lower set, e.g. due
        # to elution order contradictions.
        u_set_node = OrderedDict([(key, value) for key, value in ul_set_node[0].items()
                                  if key not in ul_set_node[1].keys()])
        l_set_node = OrderedDict([(key, value) for key, value in ul_set_node[1].items()
                                  if key not in ul_set_node[0].keys()])

    # for each node in the upper set of i do
    # Upper set: x_j > x_i, j is preferred over i
    for u_set_node, dist in u_set_node.items():
        # Exclude two types of pairs:
        # a) Molecules are the same between two systems
        # b) Their distance in the order graph does not reach
        #    a certain minimum distance
        # c) Their distance in the order graph exceeds a certain
        #    threshold.
        if dist == 0 or dist < d_lower or dist > d_upper:
            continue

        # find its row j in X
        # TODO: Bring this down constant time, by using a dictionary (hash-table).
        row_j = _find_in((dmolecules_inv[u_set_node[0]], dcollections_inv[u_set_node[1]]), keys)
        assert (len(row_j) == 1)

        # j elutes before i ==> t_j < t_i ==> w^T x_j < w^T x_i
        pairs.append((row_j[0], row_i[0]))

    # for each node in the lower set of i do
    # Lower set: x_i > x_j, i is preferred over j
    for l_set_node, dist in l_set_node.items():
        # Exclude two types of pairs:
        # a) Molecules are the same between two systems
        # b) Their distance in the order graph does not reach
        #    a certain minimum distance
        # c) Their distance in the order graph exceeds a certain
        #    threshold.
        if dist == 0 or dist < d_lower or dist > d_upper:
            continue

        # find its row j in X
        row_j = _find_in((dmolecules_inv[l_set_node[0]], dcollections_inv[l_set_node[1]]), keys)
        assert (len(row_j) == 1)

        # i elutes before j ==> t_i < t_j ==> w^T x_i < w^T x_j
        pairs.append((row_i[0], row_j[0]))

    return pairs


def get_pairs_multiple_datasets(targets, d_lower=0, d_upper=np.inf):
    """
    Task: Get the pairs (i,j) with i elutes before j for the learning / prediction process
          given a set of input features with corresponding targets.

          This function should be used if several datasets are considered, but
          inter-system transitivity is not considered. The construction of a
          retention order graph is not necessary.

          This function transforms the provided targets values into dense ranks
          using 'scipy.stats.rankdata'. In that way it is possible to exclude
          example pairs, those rank differs more than a specified threshold. We
          can in that way reduce the number of learning pairs.

    :param targets: array-like, shape = (n_samples, 2), the targets values, e.g.
                   retention times, for all molecules measured with a set of
                   datasets.

                   Example:
                   [rts_i, ds_i,
                    rts_j, ds_j,
                    rts_k, ds_k]

                   ds_i, ... are scalars representing the different datasets.

    :param d_lower: scalar, minimum distance of two molecules in the elution order graph
                    to be considered as a pair.

    :param d_upper: scalar, maximum rank difference for two examples, to be considered
                    as pair.

    :return: array-like, shape = (n_pairs,), list of tuples
             [(i,j),(j,k),...], with i elutes before j and j elutes before k, ...

             i,j,k are scalars that correspond to the particular index of the
             targets values (in 'targets'). So we need to keep the target values
             and the feature vectors in the same order!
    """
    # Number of feature vectors / samples
    n_samples = len(targets)

    pairs = []

    datasets = targets[:, 1]
    ranks = np.zeros(n_samples)

    # Ranks must be calculate per system
    for ds in np.unique(datasets):
        ranks[datasets == ds] = sp_stats.rankdata(targets[datasets == ds, 0], method="dense")

    for pair in itertools.combinations(range(n_samples), 2):
        i, j = pair

        # Skip pairs not in the same system
        if not (datasets[i] == datasets[j]):
            continue

        # Skip pairs with the same targets value
        if ranks[i] == ranks[j]:
            continue

        # Skip pairs those rank exceeds the threshold
        if np.abs(ranks[i] - ranks[j]) < d_lower or np.abs(ranks[i] - ranks[j]) > d_upper:
            continue

        if ranks[i] < ranks[j]:
            # i elutes before j ==> t_i < t_j ==> w^T(x_j - x_i) > 0 ==> w^Tx_j > w^Tx_i
            pairs.append((i, j))
        else:
            # j elutes before i ==> t_j < t_i ==> w^T(x_i - x_j) > 0 ==> w^Tx_i > w^Tx_j
            pairs.append((j, i))

    return pairs


def get_pairs_multiple_datasets_y_ds_separated(rts, datasets, d_lower=0, d_upper=np.inf):
    """
    Task: Get the pairs (i,j) with i elutes before j for the learning / prediction process
          given a set of input features with corresponding targets.

          This function should be used if several datasets are considered, but
          inter-system transitivity is not considered. The construction of a
          retention order graph is not necessary.

          This function transforms the provided targets values into dense ranks
          using 'scipy.stats.rankdata'. In that way it is possible to exclude
          example pairs, those rank differs more than a specified threshold. We
          can in that way reduce the number of learning pairs.

    :param targets: array-like, shape = (n_samples, 2), the targets values, e.g.
                   retention times, for all molecules measured with a set of
                   datasets.

                   Example:
                   [rts_i, ds_i,
                    rts_j, ds_j,
                    rts_k, ds_k]

                   ds_i, ... are scalars representing the different datasets.

    :param d_lower: scalar, minimum distance of two molecules in the elution order graph
                    to be considered as a pair.

    :param d_upper: scalar, maximum rank difference for two examples, to be considered
                    as pair.

    :return: array-like, shape = (n_pairs,), list of tuples
             [(i,j),(j,k),...], with i elutes before j and j elutes before k, ...

             i,j,k are scalars that correspond to the particular index of the
             targets values (in 'targets'). So we need to keep the target values
             and the feature vectors in the same order!
    """
    assert (isinstance(rts, np.ndarray))
    assert (isinstance(datasets, np.ndarray))
    assert (len(rts.shape) == 1)
    assert (rts.shape == datasets.shape), "To each target value, a dataset is needed: (rts_i, ds_i)."

    # Ranks must be calculate per system
    ranks = np.full_like(rts, fill_value=np.nan)
    for ds in np.unique(datasets):
        ranks[datasets == ds] = sp_stats.rankdata(rts[datasets == ds], method="dense")
    assert (not np.any(np.isnan(ranks))), "Ups: We do have ranks that are NaN."

    # Number of feature vectors / samples
    n_samples = len(rts)
    pairs = []
    for pair in itertools.combinations(range(n_samples), 2):
        i, j = pair

        # Skip pairs not in the same system
        if not (datasets[i] == datasets[j]):
            continue

        # Skip pairs with the same targets value
        if ranks[i] == ranks[j]:
            continue

        # Skip pairs those rank exceeds the threshold
        if np.abs(ranks[i] - ranks[j]) < d_lower or np.abs(ranks[i] - ranks[j]) > d_upper:
            continue

        if ranks[i] < ranks[j]:
            # i elutes before j ==> t_i < t_j ==> w^T(x_j - x_i) > 0 ==> w^Tx_j > w^Tx_i
            pairs.append((i, j))
        else:
            # j elutes before i ==> t_j < t_i ==> w^T(x_i - x_j) > 0 ==> w^Tx_i > w^Tx_j
            pairs.append((j, i))

    return pairs


def get_pairs_single_dataset(targets, d_lower=0, d_upper=np.inf, pw_conf_fun=None):
    """
    Task: Get the pairs (i,j) with i elutes before j for the learning / prediction process
          given a set of input features with corresponding targets.

          This function should be used if only one dataset is considered, i.e.
          we only need to get the learning pairs within one particular system.
          The construction of a retention order graph is not necessary.

          This function transforms the provided targets values into dense ranks
          using 'scipy.stats.rankdata'. In that way it is possible to exclude
          example pairs, those rank differs more than a specified threshold. We
          can in that way reduce the number of learning pairs.

    :param targets: array-like, shape = (n_samples,), the targets values, e.g.
                   retention times, for all molecules measured with a particular
                   system.

                   Example:
                   [rts_i, rts_j, rts_k, ...]

    :param d_lower: scalar, minimum distance of two molecules in the elution order graph
                    to be considered as a pair.

    :param d_upper: scalar, maximum rank difference for two examples, to be considered
              as pair.

    :param pw_conf_fun: function, calculating the pairwise confidence from the
        ranks.

    :return: array-like, shape = (n_pairs,), list of tuples
             [(i,j),(j,k),...], with i elutes before j and j elutes before k, ...

             OR (if return_rt_differences == True)

             tuple of lists
             * array-like, shape = (n_pairs,), list of tuples
               [(i,j),(j,k),...], with i elutes before j and j elutes before k, ...

             * array-like, shape = (n_pairs,), list of scalars pairwise confidence / margin
               [r_ij, r_kj, ...]

             i,j,k are scalars that correspond to the particular index of the
             targets values (in 'targets'). So we need to keep the target values
             and the feature vectors in the same order!
    """
    if pw_conf_fun is None:
        def __f(r_i, r_j, **kwargs): return 1.0
        pw_conf_fun = __f

    if not callable(pw_conf_fun):
        raise ValueError("Pairwise confidence calculation requires a function.")

    # Number of feature vectors / samples
    n_samples = len(targets)

    pairs = []
    pairwise_confidences = []

    ranks = sp_stats.rankdata(targets, method="dense")

    for pair in itertools.combinations(range(n_samples), 2):
        i, j = pair

        # Skip pairs with the same targets value
        if ranks[i] == ranks[j]:
            continue

        # Skip pairs those rank exceeds the threshold
        if np.abs(ranks[i] - ranks[j]) < d_lower or np.abs(ranks[i] - ranks[j]) > d_upper:
            continue

        if ranks[i] < ranks[j]:
            # i elutes before j ==> t_i < t_j ==> w^T(x_j - x_i) > 0 ==> w^Tx_j > w^Tx_i
            pairs.append((i, j))
            pairwise_confidences.append(pw_conf_fun(ranks[i], ranks[j], **{"max_rank": np.max(ranks)}))
        else:
            # j elutes before i ==> t_j < t_i ==> w^T(x_i - x_j) > 0 ==> w^Tx_i > w^Tx_j
            pairs.append((j, i))
            pairwise_confidences.append(pw_conf_fun(ranks[j], ranks[i], **{"max_rank": np.max(ranks)}))

    return pairs, np.array(pairwise_confidences)


def get_pairs_single_dataset_with_labels(targets, d_lower=0, d_upper=np.inf):
    """

    :param targets:
    :param d_lower:
    :param d_upper:
    :return:
    """
    # Number of feature vectors / samples
    n_samples = len(targets)

    pairs, labels = [], []

    ranks = sp_stats.rankdata(targets, method="dense")

    for k, pair in enumerate(itertools.combinations(range(n_samples), 2)):
        i, j = pair

        # Skip pairs with the same targets value
        if ranks[i] == ranks[j]:
            continue

        # Skip pairs those rank exceeds the threshold
        if np.abs(ranks[i] - ranks[j]) < d_lower or np.abs(ranks[i] - ranks[j]) > d_upper:
            continue

        if (k % 2) == 0:
            labels.append(1)
            if ranks[i] < ranks[j]:
                # i elutes before j ==> t_i < t_j ==> w^T(x_j - x_i) > 0 ==> w^Tx_j > w^Tx_i
                pairs.append((i, j))
            else:
                pairs.append((j, i))
        else:
            labels.append(-1)
            if ranks[i] < ranks[j]:
                pairs.append((j, i))
            else:
                pairs.append((i, j))

    return pairs, np.array(labels)
