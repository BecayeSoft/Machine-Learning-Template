"""
A collection of useful methods to visualize data.
"""

import math
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.utils import load_config, replace_slashes

from os.path import dirname, join, abspath, normpath


"""
We use dynamic paths to avoid import errors.
Since notebooks/EDA.py is one level below the root folder (..)
and src/visualization/data_visualization is two levels below the root folder (../..),
loading config/plot_config.yaml from notebooks/EDA.py 
using ../../config/plot_config.yaml would cause errors.
On the other hand, loading config/plot_config.yaml from src/main.py 
using ../config/plot_config.yaml would also cause errors.
"""

# Current directory
current_dir = dirname(abspath(__file__))

# Load the plot config and get the EDA plots path
plot_config_path = normpath(join(current_dir, "..", "..", "config", "plot.yaml"))
plot_config = load_config(plot_config_path)

eda_plots_path = plot_config.get("eda_plots_path")
eda_plots_path = normpath(join(current_dir, eda_plots_path))


# -----------------------------------------
# Methods to prepare the data for plotting
# -----------------------------------------
def get_cat_features_by_unique_values(columns, df, min_unique=0, max_unique=np.inf):
    """
    Returns categories based on the number of unique values.

    Parameters
    ----------
    columns: array
        categorical columns to check.
    df: DataFrame
        DataFrame containing the data.
    min_unique: number, default=0
        minimum number of unique values a column should have to be kept.
    max_unique: number, default=30
        maximum number of unique values a column should have to be kept.

    Returns
    -------
    filtered_categories: array
        array of columns names with unique categories between `min_unique` and `max_unique`.
    """
    # len(df[col].unique()) is the number of unique values in the column.
    filtered_categories = [
        col
        for col in columns
        if len(df[col].unique()) >= min_unique and len(df[col].unique()) <= max_unique
    ]

    return filtered_categories


# -----------------------------------
# Methods to plot Numerical Features
# -----------------------------------


def histograms(
    columns,
    df,
    plot_shape=(4, 4),
    figsize=(16, 10),
    labelrotation=45,
    save=False,
    save_as=None,
):
    """
    Plots the histograms of the specified columns of the DataFrame.

    Notes:
    -----
        The columns should be numerical.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data.
    columns: List[str]
        numerical columns of `df`.
    binwidth: int, default=3
        width of the bins.
    plot_shape: tuple
        number of rows and columns to plot.
    figsize: tuple, optional
        size of the figure.
    labelrotation: int, default=45
        rotation of the x-axis labels.
    save: bool, default=False
        whether to save the figure or not.
    save_as: str, optional
        name of the file to save the figure to.
        If not specified, 'numeric-distribution.png' is used.
    """
    fig, axs = plt.subplots(plot_shape[0], plot_shape[1], figsize=figsize)
    axs = axs.flatten()  # flatten the axs array for easy iteration.

    # Iterate over the columns and plot each one
    for i, column in enumerate(columns):
        # sns.histplot(df[column], binwidth=3, kde=kde, ax=axs[i])   # too slow
        axs[i].hist(df[column])
        axs[i].set_title(column)
        axs[i].tick_params(axis="x", labelrotation=labelrotation)

    plt.subplots_adjust(hspace=0.75)
    plt.suptitle("Numerical Features Distribution", fontsize=30)

    if save:
        # if save_as is defined, use it as the filename, else use 'numeric-distribution.png'.
        save_as = save_as or "Numeric Distributions"
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(eda_plots_path, filename))
        plt.savefig(path)
        print(f"Figure saved to {path}")

    plt.show()


def histogram(
    column, df, kde=True, binwidth=3, figsize=(7, 4), save=False, save_as=None
):
    """
    Plots a histogram of the specified column of the DataFrame.

    Notes:
    -----
        The column should be numerical.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data.
    column: array
        numerical column to plot.
    kde: bool, default=False
        whether to plot the kernel density estimate.
    binwidth: int, default=3
        width of each bin.
    figsize: tuple, default=(7, 4)
        size of the figure.
    save: bool, default=False
        whether to save the figure or not.
    save_as: str, optional
        name of the file to save the figure to.
        If not specified, '{column name} distribution.png' is used.
    """
    plt.figure(figsize=figsize)
    sns.histplot(df[column], binwidth=binwidth, kde=kde)
    plt.title(f"{column} Distribution")

    if save:
        # if save_as is defined, use it as the filename, else use the column name.
        save_as = save_as or f"{column} distribution"
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(eda_plots_path, filename))
        plt.savefig(path)
        print(f"Figure saved to {path}")

    plt.show()


def boxes(
    columns,
    df,
    plot_shape=(4, 4),
    hspace=0.75,
    figsize=(16, 10),
    save=False,
    save_as=None,
):
    """
    Plots box plots of the specifed columns of the DataFrame.

    Notes:
    -----
        The columns should be numerical.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data.
    columns: array
        numerical columns of `df`.
    plot_shape: tuple
        number of rows and columns to plot.
    figsize: tuple
        size of the figure.
    save: bool, default=False
        whether to save the figure or not.
    save_as: str, optional
        name of the file to save the figure to.
        If not specified, 'numeric-boxplot.png' is used.
    """
    fig, axs = plt.subplots(plot_shape[0], plot_shape[1], figsize=figsize)
    axs = axs.flatten()  # flatten the axs array for easy iteration.

    # Iterate over the numerical features and plot each column
    for i, column in enumerate(columns):
        sns.boxplot(x=df[column], showmeans=True, ax=axs[i])
        axs[i].set_title(column)
        axs[i].set(xlabel=None)

    plt.subplots_adjust(hspace=hspace)
    plt.suptitle("Box Plot of the Numerical Features", fontsize=30)

    if save:
        # if save_as is defined, use it as the filename, else use 'numeric-boxplot.png'.
        save_as = save_as or "Numeric Boxplots"
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(eda_plots_path, filename))
        plt.savefig(path)
        print(f"Figure saved to {path}")

    plt.show()


def box(column, df, figsize=(7, 4), save=False, save_as=None):
    """
    Plots a box plot of the specifed column of the DataFrame.

    Notes:
    -----
        The columns should be numerical.

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data.
    column: array
        numerical column of `df`.
    plot_shape: tuple
        number of rows and columns to plot.
    figsize: tuple
        size of the figure.
    save: bool, default=False
        whether to save the figure or not.
    save_as: str, optional
        name of the file to save the figure to.
        If not specified, 'column name boxplot.png' is used.
    """

    plt.figure(figsize=figsize)
    sns.boxplot(x=df[column], showmeans=True)
    plt.set_title(column)
    plt.set(xlabel=None)

    if save:
        # if save_as is defined, use it as the filename, else use 'numeric-boxplot.png'.
        save_as = save_as or f"{column} boxplot"
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(eda_plots_path, filename))
        plt.savefig(path)
        print(f"Figure saved to {path}")

    plt.show()


def heatmap(
    columns,
    df,
    cmap="coolwarm",
    triangle=True,
    annotate=True,
    figsize=(12, 9),
    save=False,
    save_as=None,
):
    """
    Plots a heatmap of the specified columns of the DataFrame.

    Notes:
    -----
        The columns should be numerical.

    Parameters
    ----------
    columns: array
        Numerical columns to plot.
    df: DataFrame
        DataFrame containing the data.
    cmap: str, default='coolwarm'
        color map to use.
    triangle: bool, default=True
        whether to plot only the upper triangle of the heatmap.
    annotate: bool, default=False
        whether to annotate the heatmap. If True, values will be displayed on each cell.
    figsize: tuple, default=(9, 6)
        size of the figure.
    save: bool, default=True
        whether to save the figure or not.
    save_as: str, optional
        name of the file to save the figure to.
        If not specified, 'numeric-heatmap.png' is used.
    """
    plt.figure(figsize=figsize)

    # create a correlation matrix of the numerical columns
    corr_matrix = df[columns].corr()

    # if the triangle is True, plot only the upper triangle of the heatmap
    if triangle:
        # create a mask for the upper triangle of the heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, annot=annotate)
    else:
        sns.heatmap(corr_matrix, cmap="coolwarm", annot=annotate)

    plt.tight_layout()
    plt.title("Numerical Features Correlations", fontsize=30)
    if save:
        # if save_as is defined, use it as the filename, else use 'numeric heatmap.png'.
        save_as = save_as or "Numeric Heatmap"
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(eda_plots_path, filename))
        plt.savefig(path)
        print(f"Figure saved to {path}")

    plt.show()


def scatter(column1, column2, df, figsize=(7, 4), save=False, save_as=None):
    """
    Plots a scatter plot of the specified columns of the DataFrame.

    Notes:
    -----
        The columns should be numerical.

    Parameters
    ----------
    column1: str
        first column to plot.
    column2: str
        second column to plot.
    df: DataFrame
        DataFrame containing the data.
    figsize: tuple, default=(9, 6)
        size of the figure.
    save: bool, default=False
        whether to save the figure or not.
    save_as: str, optional
        name of the file to save the figure to.
        If not specified, 'column1 VS column2 scatter plot.png' is used.
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(x=df[column1], y=df[column2])
    plt.title(f"{column1} VS {column2} Scatter Plot")

    if save:
        # if save_as is defined, use it as the filename, else use 'column1 VS column2 scatter plot.png'.
        save_as = save_as or f"{column1} VS {column2} Scatter Plot"
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(eda_plots_path, filename))
        plt.savefig(path)
        print(f"Figure saved to {path}")

    plt.show()


def scatter_matrix(columns, df, figsize=None, save=False, save_as=None):
    """
    Plots a scatter matrix of the columns of a DataFrame.

    Notes:
    -----
        The columns should be numerical.

    Parameters
    ----------
    columns: array
        columns to plot.
    df: DataFrame
        DataFrame containing the data.
    figsize: tuple, optional
        size of the figure.
    save: bool, default=False
        whether to save the figure or not.
    save_as: str, optional
        name of the file to save the figure to.
        If not specified, 'numeric-scatter-matrix.png' is used.
    """
    if figsize is not None:
        plt.figure(figsize=figsize)

    sns.pairplot(df[columns])

    if save:
        # if save_as is defined, use it as the filename, else use 'numeric scatter matrix.png'.
        save_as = save_as or "Numeric Scatter Matrix.png"
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(eda_plots_path, filename))
        plt.savefig(path)
        print(f"Figure saved to {path}")

    plt.show()


# -------------------------------------
# Methods to plot Categorical Features
# -------------------------------------


def bars(
    columns,
    df,
    plot_shape=(6, 5),
    figsize=(20, 15),
    labelrotation=90,
    hspace=0.75,
    save=False,
    save_as=None,
):
    """
    Plots bar plots of the specified columns in the DataFrame.

    Notes:
    -----
        The columns should be categorical.

    Parameters
    ----------
    columns: List[str]
        categorical columns of `df`.
    df: DataFrame, default=None
        DataFrame containing the data.
    plot_shape: tuple
        number of rows and columns of the figure.
    figsize: tuple
        size of the figure.
    labelrotation: int, default=90
        rotation angle of the x labels. This is applied to features with more than 3 categories to avoid overlapping.
    hspace: float
        padding height between the plots.
    save: bool, default=False
        whether to save the figure or not.
    save_as: str, optional
        name of the file to save the figure to.
        If not specified, 'categories-bars.png' is used.
    """
    fig, axs = plt.subplots(plot_shape[0], plot_shape[1], figsize=figsize)
    axs = axs.flatten()  # flatten the axs array for easy iteration.

    # Iterate over the categorical features and plot each column
    for i, column in enumerate(columns):
        sns.countplot(x=df[column].astype("category"), ax=axs[i])
        axs[i].set_title(column)
        axs[i].set(xlabel=None)

        # Rotate x labels if the feature has more than 3 categories to avoid overlapping
        if len(df[column].unique()) > 3:
            axs[i].tick_params(axis="x", labelrotation=labelrotation)

    plt.subplots_adjust(hspace=hspace)
    plt.suptitle("Distribution of the Categorical Columns", fontsize=30)

    if save:
        # if save_as is defined, use it as the filename, else use 'categories-bars.png'.
        save_as = save_as or "Categories Bars"
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(eda_plots_path, filename))
        plt.savefig(path)
        print(f"Figure saved to {path}")

    plt.show()


def bar(
    column,
    df=None,
    title=None,
    ylim=None,
    xlabelrotation=0,
    figsize=(30, 7),
    save=False,
    save_as=None,
):
    """
    Plot a bar plot of the specified column.

    Parameters
    ----------
    column: string
        column of the DataFrame.
        If string, it represents the name of the column.
        If pandas.Series, it represents the column data itself.
    df: DataFrame, default=None
        DataFrame containing the data.
        If `column` is a string, `df` must be specified.
        If `column` is a pandas.Series, `df` is ignored.
    title: string, default=''
        If column is pandas series, `title` used as the title of the figure.
        If `column` is a string, `title` is ignored.
    ylim: tuple, default=None
        Tuple specifying the lower and upper limits of the y-axis.
    xlabelrotation: number, default=0
        Rotation angle of the x labels to avoid overlapping.
        If None, the y-axis scale will be determined automatically.
    figsize: tuple, default=(30, 7)
        Size of the figure.
    save: bool, default=False
        Whether to save the figure or not.
    save_as: str, optional
        Name of the file to save the figure to.

    Notes
    -----
    The xlabelrotation and figsize parameters are particularly useful for categories
    with a large number of unique values. They allow you to rotate the x-axis labels and
    adjust the size of the figure, respectively, to enhance readability.
    Moreover, ylim can should be used when creating different plots
    of the same column's categories to ensure that the y-axis scale is the same.
    """
    plt.figure(figsize=figsize)
    path = None  # path to save the figure to

    # `column` is a column name
    if isinstance(column, str):
        if df is None:
            raise ValueError("If 'column' is a column name, 'df' must be specified.")

        sns.countplot(x=df[column])
        plt.title(column)
        save_as = save_as or f"{column} Bar"
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(eda_plots_path, filename))

    # `column` is a pandas.Series
    elif isinstance(column, pd.Series):
        if title is None:
            raise ValueError(
                "If 'column' is a pandas.Series, 'title' must be specified."
            )

        sns.countplot(x=column)
        plt.title(title)
        save_as = save_as or f"{title} Bar"
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(eda_plots_path, filename))

    else:
        raise ValueError(
            "Invalid column type. 'column' should be a string (column name) or a pandas.Series (column data)."
        )

    plt.xlabel(None)
    plt.tick_params(axis="x", labelrotation=xlabelrotation)

    # Set y-axis limits if specified
    if ylim is not None:
        plt.ylim(ylim)

    if save:
        plt.savefig(path)
        print(f"Figure saved to {path}")

    plt.show()


# ---------------------------------------
# Mathods to compare Target and Features
# ---------------------------------------


def target_vs_categories(
    target,
    categorical_columns,
    df,
    plot_shape=(1, 1),
    figsize=(16, 10),
    xlabelrotation=75,
    save=False,
    save_as=None,
):
    """
    Visualize the distribution of the target variable across different categories of other variables using bar plots.

    Parameters:
    -----------
    df : DataFrame
        The DataFrame containing the data.
    target_variable : str
        The name of the target variable (e.g., 'Status').
    categorical_variables : list
        The list of categorical variables to compare the distribution.
    plot_shape: tuple
        number of rows and columns to plot.
    figsize: tuple
        size of the figure.
    xlabelrotation: int
        rotation angle of the x labels to avoid overlapping.
    save: bool
        whether to save the figure or not.
    save_as: str
        name of the file to save the figure to.

    Returns:
    --------
    None
        Displays the bar plots.
    """
    fig, axs = plt.subplots(plot_shape[0], plot_shape[1], figsize=figsize)
    axs = axs.flatten()  # flatten the axs array for easy iteration.

    # Iterate over the numerical features and plot each column
    for i, column in enumerate(categorical_columns):
        sns.countplot(x=df[column], hue=df[target], ax=axs[i])
        axs[i].set_title(f"{target} VS {column}")
        axs[i].set_xlabel(None)
        axs[i].legend(title=target)

        # Rotate x labels if the feature has more than 3 categories to avoid overlapping
        if len(df[column].unique()) > 3:
            axs[i].tick_params(axis="x", labelrotation=xlabelrotation)

    plt.subplots_adjust(hspace=1)
    plt.suptitle(f"{target} Distribution accross Categorical Features", fontsize=30)

    if save:
        # if save_as is defined, use it as the filename, else use 'Features VS {target} Distribution.png'.
        save_as = save_as or f"Features VS {target} Distribution"
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(eda_plots_path, filename))
        plt.savefig(path)
        print(f"Figure saved to {path}")

    plt.show()


def target_vs_numeric(
    target,
    columns,
    df,
    plot_shape=(5, 4),
    figsize=(20, 15),
    labelrotation=75,
    hspace=0.75,
    save=False,
    save_as=None,
):
    """
    Visualize the distribution of the target variable across numerical features using density plots.

    Parameters:
    -----------
    df : DataFrame
        The DataFrame containing the data.
    target_variable : str
        The name of the target variable (e.g., 'Status').
    numerical_features : list
        The list of numerical features to compare the distribution.
    plot_shape: tuple
        number of rows and columns to plot.
    figsize: tuple
        size of the figure.
    labelrotation: int
        rotation angle of the x labels to avoid overlapping.
    hspace: float
        padding height between the plots.
    save: bool
        whether to save the figure or not.
    save_as: str
        name of the file to save the figure to.
    """

    fig, axs = plt.subplots(plot_shape[0], plot_shape[1], figsize=figsize)
    axs = axs.flatten()

    active_filter = df["Status"] == "Active"
    terminated_filter = df["Status"] == "Terminated"

    for i, column in enumerate(columns):
        sns.kdeplot(df[active_filter][column], label="Active", ax=axs[i])
        sns.kdeplot(df[terminated_filter][column], label="Terminated", ax=axs[i])

        axs[i].set_title(f"{column} vs {target}")
        axs[i].set(xlabel=None)
        # axs[i].tick_params(axis='x', labelrotation=labelrotation)
        axs[i].legend()

        # Rotate x labels if the columns has big values to avoid overlapping
        if df[column].max() > 100:
            axs[i].tick_params(axis="x", labelrotation=labelrotation)

    plt.subplots_adjust(hspace=hspace)
    plt.suptitle(f"{target} Distribution accross Numerical Features", fontsize=30)

    if save:
        # if save_as is defined, use it as the filename, else use 'Numerical features VS {target}'.
        save_as = save_as or f"Numerical features VS {target}"
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(eda_plots_path, filename))
        plt.savefig(path)
        print(f"Figure saved to {path}")

    plt.show()
