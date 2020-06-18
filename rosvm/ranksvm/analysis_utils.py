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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from typing import Optional, Iterable, Union, List

from rosvm.ranksvm.rank_svm_cls import KernelRankSVC


class RankSVMAnalyzer(object):
    """
    Class to inspect a fitted RankSVM model.
    """
    def __init__(self, ranksvm: Union[KernelRankSVC, List[KernelRankSVC]]):
        if not isinstance(ranksvm, list):
            self.ranksvm = [ranksvm]
        else:
            self.ranksvm = ranksvm

        self.n_models = len(self.ranksvm)

        # Read out debug information, if available
        self.objective_values_df = pd.DataFrame({"Model": [], "Step": [], "Primal": [], "Dual": [], "Gap": []})
        for i, model in enumerate(self.ranksvm):
            # Check, whether the model has debug information available
            if not model.debug:
                continue

            _obj_values_df = pd.DataFrame({"Step": model.debug_data_["step"],
                                           "Primal": model.debug_data_["primal_obj"],
                                           "Dual": model.debug_data_["dual_obj"],
                                           "Gap": model.debug_data_["duality_gap"]})
            _obj_values_df["Model"] = "RankSVM_%02d" % i

            self.objective_values_df = pd.concat((self.objective_values_df, _obj_values_df), axis=0, sort=True)

    def plot_objective_functions(self, ax: Optional[plt.Axes] = None, add_primal=True, add_dual=True,
                                 add_duality_gap=False, xscale="linear", yscale="log", use_col=False) -> plt.Axes:
        if ax is None:
            plt.figure()
            ax = plt.gca()

        value_vars = []
        if add_primal:
            value_vars.append("Primal")
        if add_dual:
            value_vars.append("Dual")
        if add_duality_gap:
            value_vars.append("Gap")

        ax.set(xscale=xscale, yscale=yscale)

        plot_data = self.objective_values_df.melt(id_vars=["Model", "Step"], value_vars=value_vars,
                                                  var_name="Objective Type", value_name="Objective Value")

        plot_params = {"hue": "Objective Type", "ax": ax}

        if self.n_models > 1:
            if use_col:
                g = sns.FacetGrid(data=plot_data, col="Model", sharey=False, hue="Objective Type")
                g.map(sns.lineplot, "Step", "Objective Value") \
                    .add_legend() \
                    .set(xscale=xscale, yscale=yscale)
            else:
                plot_params["style"] = "Model"
                sns.lineplot("Step", "Objective Value", data=plot_data, **plot_params)
        else:
            sns.lineplot("Step", "Objective Value", data=plot_data, **plot_params)
            ax.grid(axis="y")

        return ax
