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
import numpy as np

from typing import Optional, Union, List, Dict

from rosvm.ranksvm.rank_svm_cls import KernelRankSVC


class RankSVMAnalyzer(object):
    """
    Class to inspect a fitted RankSVM model.
    """
    def __init__(self, ranksvm: Union[KernelRankSVC, List[KernelRankSVC], Dict[str, KernelRankSVC]]):
        if isinstance(ranksvm, list):
            self.ranksvm = {"RankSVM_%02d" % i: v for i, v in enumerate(ranksvm)}
        elif isinstance(ranksvm, KernelRankSVC):
            self.ranksvm = {"RankSVM": ranksvm}
        else:
            assert isinstance(ranksvm, dict)
            self.ranksvm = ranksvm

        self.n_models = len(self.ranksvm)

        # Read out debug information, if available
        self.objective_values_df = pd.DataFrame({"Model": [], "Step": [], "Primal": [], "Dual": [], "Gap": []})
        self.performance_values_df = pd.DataFrame({"Model": [], "Step": [], "Training Accuracy": [],
                                                   "Validation Accuracy": []})
        self.optimization_values_df = pd.DataFrame({"Model": [], "Step": [], "Step-size": []})

        for name, model in self.ranksvm.items():
            # Check, whether the model has debug information available
            if not model.debug:
                continue

            _obj_values_df = pd.DataFrame({"Step": model.debug_data_["step"],
                                           "Primal": model.debug_data_["primal_obj"],
                                           "Dual": model.debug_data_["dual_obj"],
                                           "Duality Gap": model.debug_data_["duality_gap"]})
            _obj_values_df["Model"] = name
            self.objective_values_df = pd.concat((self.objective_values_df, _obj_values_df), axis=0, sort=True)

            _acc_values_df = pd.DataFrame({"Step": model.debug_data_["step"],
                                           "Training Accuracy": model.debug_data_["train_score"],
                                           "Validation Accuracy": model.debug_data_["val_score"]})
            _acc_values_df["Model"] = name
            self.performance_values_df = pd.concat((self.performance_values_df, _acc_values_df), axis=0, sort=True)

            _opt_values_df = pd.DataFrame({"Step": model.debug_data_["step"],
                                           "Step-size": model.debug_data_["step_size"]})
            _opt_values_df["Model"] = name
            self.optimization_values_df = pd.concat((self.optimization_values_df, _opt_values_df), axis=0, sort=True)

    def OLD_plot_objective_functions(self, ax: Optional[plt.Axes] = None, add_primal=True, add_dual=True,
                                     add_duality_gap=False, xscale="linear", yscale="log", use_col=False,
                                     sharex=True) -> plt.Axes:
        if ax is None:
            plt.figure()
            ax = plt.gca()

        value_vars = []
        if add_primal:
            value_vars.append("Primal")
        if add_dual:
            value_vars.append("Dual")
        if add_duality_gap:
            value_vars.append("Duality Gap")

        ax.set(xscale=xscale, yscale=yscale)

        plot_data = self.objective_values_df.melt(id_vars=["Model", "Step"], value_vars=value_vars,
                                                  var_name="Objective Type", value_name="Objective Value")

        plot_params = {"hue": "Objective Type", "ax": ax}

        if self.n_models > 1:
            if use_col:
                g = sns.FacetGrid(data=plot_data, col="Model", sharey=False, hue="Objective Type", sharex=sharex)
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

    def plot_objective_functions(self, sharex=True, sharey="row", xscale="linear", yscale="log", skip_measurements=0,
                                 aspect=1.25, height=2.5):
        plot_data = self.objective_values_df \
            .groupby(by="Model") \
            .apply(lambda x: x[skip_measurements:]) \
            .melt(id_vars=["Model", "Step"], value_vars=["Primal", "Dual", "Duality Gap"],
                  var_name="Objective Type", value_name="Value")
        g = sns.FacetGrid(data=plot_data, col="Model", row="Objective Type", sharex=sharex, sharey=sharey,
                          aspect=aspect, height=height)
        g.map(sns.lineplot, "Step", "Value") \
            .set_titles("{col_name}\n{row_name}") \
            .set(xscale=xscale, yscale=yscale)

        return g

    def plot_accuracies(self, sharex=True, sharey="row", xscale="linear", yscale="linear", skip_measurements=0,
                        aspect=1.25, height=2.5):
        plot_data = self.performance_values_df \
            .groupby(by="Model") \
            .apply(lambda x: x[skip_measurements:]) \
            .melt(id_vars=["Model", "Step"], value_vars=["Training Accuracy", "Validation Accuracy"],
                  var_name="Evaluation Set", value_name="Accuracy")
        g = sns.FacetGrid(data=plot_data, col="Model", row="Evaluation Set", sharex=sharex, sharey=sharey,
                          aspect=aspect, height=height)
        g.map(sns.lineplot, "Step", "Accuracy") \
            .set_titles("{col_name}\n{row_name}") \
            .set(xscale=xscale, yscale=yscale)

        print(self.performance_values_df[["Model", "Training Accuracy", "Validation Accuracy"]]
              .groupby(by="Model").aggregate([np.max, np.median]).round(2))

        return g

    def plot_step_size(self, xscale="linear", yscale="log", skip_measurements=0, ax=None):
        plot_data = self.optimization_values_df \
            .groupby(by="Model") \
            .apply(lambda x: x[skip_measurements:]) \
            .melt(id_vars=["Model", "Step"], value_vars=["Step-size"], value_name="Step-size")

        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.set(xscale=xscale, yscale=yscale)

        plot_params = {"hue": "Model", "ax": ax}
        sns.lineplot("Step", "Step-size", data=plot_data, **plot_params)
        ax.grid()

    def plot_alphas(self, zscale="log"):
        fig, axrr = plt.subplots(1, self.n_models, figsize=(25, 50), sharex="all")

        for i, (name, model) in enumerate(self.ranksvm.items()):
            min_alpha = np.min(model.debug_data_["alpha"])
            max_alpha = np.max(model.debug_data_["alpha"])

            hist = []
            for alpha_step in model.debug_data_["alpha"]:
                _tmp = np.histogram(alpha_step, bins=51, range=(min_alpha, max_alpha))
                hist.append(_tmp[0])
                yrange = _tmp[1]
            hist = np.array(hist)

            if zscale == "log":
                hist = np.log(hist + 1)

            ax = axrr[i]  # type: plt.Axes
            ax.matshow(hist.T, vmax=np.max(hist), vmin=np.min(hist))

            ax.set_title(name)
            ax.set_xticklabels([None] + [0, 100, 200, 300, 400, 500])
            ax.set_xlabel("Step")
            ax.set_yticklabels([None] + ["%.3f" % y for j, y in enumerate(yrange) if j in [0, 10, 20, 30, 40, 50]])
            ax.set_ylabel("Dual value")

        plt.show()
