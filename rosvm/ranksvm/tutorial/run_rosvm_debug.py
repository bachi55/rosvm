####
#
# The MIT License (MIT)
#
# Copyright 2020 Eric Bach <eric.bach@aalto.fi>
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
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, GridSearchCV

from rosvm.ranksvm.rank_svm_cls import KernelRankSVC
from rosvm.ranksvm.tutorial.utils import read_dataset

if __name__ == "__main__":
    # Load example tutorial
    X, y, mol = read_dataset("./example_data.csv")

    # Split into training and test
    train, test = next(GroupKFold(n_splits=3).split(X, y, groups=mol))
    print("(n_train, n_test) = (%d, %d)" % (len(train), len(test)))
    assert not (set(mol[train]) & set(mol[test]))

    X_train, X_test = X[train], X[test]
    y_train, y_test = y[train], y[test]
    mol_train = mol[train]

    # Train the RankSVM and run gridsearch for best C
    ranksvm = KernelRankSVC(kernel="minmax", pair_generation="random", random_state=2921, alpha_threshold=1e-2,
                            max_iter=1000, debug=True, C=1, pairwise_features="exterior_product").fit(X_train, y_train)

    plt.figure()
    plt.semilogy(ranksvm.debug_data_["step"], ranksvm.debug_data_["primal_obj"], '.--')
    plt.semilogy(ranksvm.debug_data_["step"], ranksvm.debug_data_["dual_obj"], '.--')
    plt.show()

    plt.figure()
    plt.plot(ranksvm.debug_data_["step"], ranksvm.debug_data_["train_score"], '.--')
    plt.plot(ranksvm.debug_data_["step"], ranksvm.debug_data_["val_score"], '.--')
    plt.show()