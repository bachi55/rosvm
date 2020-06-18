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

"""
This script runs the evaluations of paper by Bach et al. (2018) using the original datasets.

Changes to the experiments as previously presented in the paper:

    - Counting Substructure fingerprints rather than MACCS fingerprints are used
    - Slight differences in the datasets due to a modified exclusion of early eluting molecules
    - Different training and test set folds
    - As training pairs we use 5% of the total available pairs that can be generated from the training set

------

[Bach2018]
    Bach, E.; Szedmak, S.; Brouard, C.; Böcker, S. & Rousu, J.
    Liquid-chromatography retention order prediction for metabolite identification
    Bioinformatics, 2018, 34, i875-i883
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GridSearchCV

from rosvm.ranksvm.rank_svm_cls import KernelRankSVC
from rosvm.ranksvm.tutorial.utils import read_dataset

# Grid search and evaluation parameters
CV_FOLDS = 10
C_GRID = [0.25, 0.5, 1, 2, 4, 8, 16]
INNER_CV = GroupKFold(n_splits=3)
N_JOBS = 4

# RankSVM parameters
MAX_ITER = 500
PAIR_GENERATION = "random"


def get_scores(joint_model: bool, feature: str):
    gridseachcv = GridSearchCV(
        estimator=KernelRankSVC(
            kernel="minmax", pair_generation=PAIR_GENERATION, random_state=123, alpha_threshold=1e-2,
            max_iter=MAX_ITER, pairwise_features=feature),
        param_grid={"C": C_GRID},
        cv=INNER_CV,
        n_jobs=N_JOBS)

    scores = {ds: [] for ds in y.get_unique_dss()}
    for train, test in GroupKFold(n_splits=CV_FOLDS).split(X, y, groups=mol):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        mol_train = mol[train]

        if joint_model:
            ranksvm = gridseachcv.fit(X_train, y_train, groups=mol_train)
            for ds in y_train.get_unique_dss():
                idc_ds_test = y_test.get_idc_for_ds(ds)
                scores[ds].append(ranksvm.score(X_test[idc_ds_test], y_test[idc_ds_test]))
        else:
            for ds in y_train.get_unique_dss():
                idc_ds_train = y_train.get_idc_for_ds(ds)
                idc_ds_test = y_test.get_idc_for_ds(ds)
                ranksvm = gridseachcv.fit(X_train[idc_ds_train], y_train[idc_ds_train], groups=mol_train[idc_ds_train])
                scores[ds].append(ranksvm.score(X_test[idc_ds_test], y_test[idc_ds_test]))

    return scores


if __name__ == "__main__":
    # Load dataset
    X, y, mol = read_dataset("./ECCB2018_data.csv")

    # Run scoring for different pairwise features
    for pw_feat in ["difference", "exterior_product"]:
        print(pw_feat)
        scores_indv = get_scores(False, pw_feat)
        scores_joint = get_scores(True, pw_feat)

        res = pd.concat((pd.DataFrame(scores_indv), pd.DataFrame(scores_joint)), axis=0, sort=True)
        res["Model"] = ["single"] * CV_FOLDS + ["joint"] * CV_FOLDS
        res = res \
            .groupby("Model") \
            .aggregate(lambda x: "%.2f (%.2f)" % (np.mean(x), np.std(x))) \
            .reset_index()
        print(res)

        res.to_csv("./ECCB2018_results__pw_feat=%s__pgen=%s.csv" % (pw_feat, PAIR_GENERATION), index=False)