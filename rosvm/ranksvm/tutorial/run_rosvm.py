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
    ranksvm = GridSearchCV(
        estimator=KernelRankSVC(kernel="minmax", pair_generation="random", random_state=2921, alpha_threshold=1e-2),
        param_grid={"C": [1/16, 1/4, 1, 4, 16]},
        cv=GroupKFold(n_splits=3),
        n_jobs=4).fit(X_train, y_train, groups=mol_train)
    print(np.round(ranksvm.cv_results_["mean_test_score"], 2))

    # Inspect RankSVM prediction
    print("Score: %3f" % ranksvm.score(X_test, y_test))
    print(ranksvm.best_estimator_.score(X_test, y_test, return_score_per_dataset=True))

    fig, axrr = plt.subplots(1, 2, figsize=(12, 6))
    dss_test = np.array(y_test.get_dss())
    rts_test = np.array(y_test.get_rts())
    axrr[0].scatter(rts_test[dss_test == "FEM_long"],
                    ranksvm.best_estimator_.predict_pointwise(X_test[dss_test == "FEM_long"]))
    axrr[1].scatter(rts_test[dss_test == "UFZ_Phenomenex"],
                    ranksvm.best_estimator_.predict_pointwise(X_test[dss_test == "UFZ_Phenomenex"]))
    plt.show()
