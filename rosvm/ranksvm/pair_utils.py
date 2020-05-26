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

from itertools import combinations
from scipy.stats import rankdata


def get_pairs_multiple_datasets(targets, d_lower=1, d_upper=np.inf):
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

    :param targets: Labels, length = n_samples, the targets values, e.g.
                    retention times, for all molecules measured with a set of
                    datasets.

                    Example:
                    [..., (rts_i, ds_i), (rts_j, ds_j), (rts_k, ds_k), ...]

                    rts_i ... retention time of measurement i
                    ds_i ... identifier of the dataset of measurement i

    :param d_lower: scalar, minimum distance of two molecules in the elution order graph
                    to be considered as a pair (default = 1).

    :param d_upper: scalar, maximum rank difference for two examples, to be considered
                    as pair (default = np.inf).

    :return: tuple (pairs, signs, pdss)

             pairs:
             array-like, shape = (n_pairs,), list of included training tuples
             [..., (i, j), (i, k), ...]

             signs:
             array-like, shape = (n_pairs,), list of retention time / retention rank
             difference signs

             pdss:
             array-like, shape = (n_pairs,), list of dataset identifier corresponding
             to each pair.

             i, j, k are scalars that correspond to the particular index of the
             targets values (in 'targets'). So we need to keep the target values
             and the feature vectors in the same order!
    """
    # Number of feature vectors / samples
    n_samples = len(targets)

    # Separate retention times and corresponding dataset information
    rts = np.array(targets.get_rts())
    dss = np.array(targets.get_dss())

    # Calculate retention order rank for each rt measurement separately for each dataset.
    ranks = np.zeros(n_samples)
    for ds in set(dss):
        ranks[dss == ds] = rankdata(rts[dss == ds], method="dense")

    # Output tutorial
    pairs = []  # pair tuples
    signs = []  # sign of the retention time / retention order rank difference
    pdss = []  # dataset identifier belonging the each pair

    for i, j in combinations(range(n_samples), 2):
        # Skip pairs not in the same system
        if dss[i] != dss[j]:
            continue

        # Skip pairs with the same target value
        if ranks[i] == ranks[j]:
            assert rts[i] == rts[j]
            continue

        # Skip pairs those rank exceeds the threshold
        _rank_diff = ranks[i] - ranks[j]
        assert _rank_diff != 0
        if np.abs(_rank_diff) < d_lower or np.abs(_rank_diff) > d_upper:
            continue

        # Add pair
        pairs.append((i, j))
        signs.append(np.sign(_rank_diff).astype("int"))
        pdss.append(dss[i])

    return pairs, signs, pdss
