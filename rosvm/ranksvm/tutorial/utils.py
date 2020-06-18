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
import pandas as pd
import numpy as np

from typing import Tuple

from rosvm.ranksvm.rank_svm_cls import Labels


def read_dataset(fn: str) -> Tuple[np.ndarray, Labels, np.ndarray]:
    """
    :param fn: string, path to the dataset file.

    :return: tuple (
        X: array-like, shape = (n_samples, n_features), feature representation of the molecules in the dataset
        y: Labels, (rt, dataset) tuples corresponding to each molecule
        mol: array-like, shape = (n_samples, ), string representation of each molecule, e.g. SMILES
    )
    """
    data = pd.read_csv(fn, sep="\t")
    X = np.array(list(map(lambda x: x.split(","), data.substructure_count.values)), dtype="float")
    y = Labels(data.rt.values, data.dataset.values)
    mol = data.smiles.values

    return X, y, mol
