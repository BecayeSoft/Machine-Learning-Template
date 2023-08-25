from sklearn.inspection import PartialDependenceDisplay

from src.utils.utils import load_config, replace_slashes
from os.path import dirname, join, abspath, normpath

import numpy as np
from time import time
import matplotlib.pyplot as plt
from matplotlib import cm


# Current directory
current_dir = dirname(abspath(__file__))

# Load the plot config and get the EDA plots path
plot_config_path = normpath(join(current_dir, "..", "..", "config", "plot.yaml"))
plot_config = load_config(plot_config_path)

interpretability_plots_path = plot_config.get("interpretability_plots")
interpretability_plots_path = normpath(join(current_dir, interpretability_plots_path))

# Plots params
common_params = {
    "subsample": 50,
    "n_jobs": -1,
    # "grid_resolution": 20,
    "random_state": 42,
}


def plot_partial_dependence(
    model,
    X,
    features,
    plot_shape,
    cat_features=None,
    figsize=(20, 12),
    title="",
    save=False,
    save_as=None,
):
    """
    Plot the partial dependence of the features on the target variable.

    The plot shape math with the number of features. FOr example,  if
    the plot shape is (2, 3), then the features should be a list of 6
    features.

    
    Parameters
    ----------
    model : sklearn estimator
        The model to use to compute the partial dependence.
    X : array-like of shape (n_samples, n_features)
        The data on which to compute the partial dependence.
    features : list of str
        The features for which to compute the partial dependence.
        Should be equal to the number of subplots (plot shape).
    plot_shape : tuple of int
        The shape of the plot.
    categories : list of str, optional
        The categorical features.
        Useful to display bar plots for categories instead of the default line plots.
    figsize : tuple of int, default=(20, 12)
        The figure size.
    title : str, default=''
        The title of the plot.
    save : bool, default=False
        Whether to save the plot.
    save_as : str, default=None
        The name of the file to save the plot.
    """
    features_info = {
        "features": features,
        "kind": "average",
        "categorical_features": cat_features,
    }

    # Custom tick labels for all one-hot encoded categorical features
    # tick_labels = {
    #     col_name: ["No", "Yes"] for col_name in categories_slice
    # }
  
    # Create subplots with custom ticks for categorical features
    _, ax = plt.subplots(
        ncols=plot_shape[0], 
        nrows=plot_shape[1],
        figsize=figsize,
        sharey=True,
        constrained_layout=True,
    )

    # Get the PartialDependenceDisplay object
    display = PartialDependenceDisplay.from_estimator(
        model,
        X,
        **features_info,
        ax=ax,
        **common_params,
    )

    # Add colorbar
    contour_set = display.contours_[0][0]
    levels = contour_set.levels
    cax = plt.axes([0.95, 0.15, 0.03, 0.7])
    cmap = cm.get_cmap('viridis')
    norm = plt.Normalize(vmin=levels.min(), vmax=levels.max())
    cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label('Partial Dependence')

    # Set custom tick labels for categorical features
    # for i, col_name in enumerate(categories_slice1):
    #     if col_name in tick_labels1:
    #         ax[i // 6, i % 6].set_xticks(np.arange(len(tick_labels1[col_name])))
    #         ax[i // 6, i % 6].set_xticklabels(tick_labels1[col_name])

    # Set the title for the entire figure
    
    if title == None:
        title = "Partial Dependence of Employee Attritions - "  + model.__class__.__name__

    _ = display.figure_.suptitle(title, fontsize=16)

    if save:
        if save_as is None:
            raise ValueError(
                'Please provide a filename to save the partial dependance plot.  E.g. "xgb_PDP"'
            )
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(interpretability_plots_path, filename))
        plt.savefig(path)
        print(f"Figure saved to {path}")

    plt.show()


def plot_ice(
    model,
    X,
    features,
    plot_shape,
    figsize=(20, 12),
    title=None,
    save=False,
    save_as=None
):
    """
    Plot the Individual Conditional Expectation (ICE) of the features on the target variable.

    The number of features be even. This is because the number of
    subplots is autmatically calulated based on the number of features.
    For instance, if we have 6 features, then the plot shape will be
    (2, 3).
    This is why we can't have an odd number of features like 7,
    because the plot shape would be (2, 4) and sklearn will raise
    an error.

    Parameters
    ----------
    model : sklearn estimator
        The model to use to compute the partial dependence.
    X : array-like of shape (n_samples, n_features)
        The data on which to compute the partial dependence.
    features : list of str
        The numerical features for which to compute the partial dependence.
        The number of features must be equal to the number of subplots
        (plot shape).
    figsize : tuple of int, default=(20, 12)
        The figure size.
    model_name : str, default=''
        The name of the model to use in the title of the plot.
    save : bool, default=False
        Whether to save the plot.
    save_as : str, default=None
        The name of the file to save the plot.
    """
    _, ax = plt.subplots(
        ncols=plot_shape[0], 
        nrows=plot_shape[1],
        figsize=figsize,
        sharey=True,
        constrained_layout=True,
    )

    features_info = {
        "features": features,
        "kind": "both",
        "centered": True,
    }
   
   
    display = PartialDependenceDisplay.from_estimator(
        model,
        X,
        **features_info,
        ax=ax,
        **common_params,
    )

    if title == None:
        title = "ICE of Employee Attritions - "  + model.__class__.__name__

    _ = display.figure_.suptitle(title, fontsize=16)

    if save:
        if save_as is None:
            raise ValueError(
                'Please provide a filename to save the ICE plot.  E.g. "xgb_ice"'
            )
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(interpretability_plots_path, filename))
        plt.savefig(path)
        print(f"Figure saved to {path}")

    plt.show()
