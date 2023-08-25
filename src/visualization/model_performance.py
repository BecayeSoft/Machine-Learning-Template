"""
A collection of methods for visualizing the performance of a machine learning model.

There are four types of methods in this class. The first three types allow you to
evaluate the performance of the model, while the last type enables you to evaluate
the model's training.

These methods generally follow a consistent pattern:
- **model**: A fitted model.
- **X**: Array-like of shape (n_samples, n_features).
- **y**: Array-like of shape (n_samples,).

Based on the model and the data, these methods will generate plots depicting the model's performance.

Methods
-------

## 1. Overall Model Performance Plots

- `print_classification_report`: Prints the classification report of the model.

- `plot_model_performance`: Plots the confusion matrix, the ROC curve, the
  precision-recall curve, and the calibration curve of a model.

## 2. Confusion Matrix, ROC, Precision-Recall, and Calibration Plots

- `plot_roc_curve`: Plots the ROC curve of a model on a single set or on
  two sets (e.g., test and validation).

- `plot_confusion_matrix`: Plots the confusion matrix curve of a model
  on a single set or on two sets (e.g., test and validation).

- `plot_calibration_curve`: Plots the calibration curve of the model.

## 3. XGBoost Feature Importance

- `plot_xgboost_feature_importance`: Plots the feature importance of an XGBoost model.

## 4. Training Evaluation Plots

- `plot_learning_curve`: Plots the learning curve of the model.

- `plot_validation_curves`: Plots the validation curves of the model.
"""


from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    classification_report,
)
from sklearn.model_selection import ValidationCurveDisplay, LearningCurveDisplay
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import DecisionBoundaryDisplay

from xgboost import plot_importance

import matplotlib.pyplot as plt
import numpy as np

from src.utils.utils import load_config, replace_slashes
from os.path import dirname, join, abspath, normpath


# Current directory
current_dir = dirname(abspath(__file__))

# Load the plot config and get the EDA plots path
plot_config_path = normpath(join(current_dir, "..", "..", "config", "plot.yaml"))
plot_config = load_config(plot_config_path)

model_plots_path = plot_config.get("model_plots_path")
model_plots_path = normpath(join(current_dir, model_plots_path))

LABELS = ["Active", "Terminated"]
CM_CMAP = "Blues"


# --------------------------------------------------
# Classification Report
# --------------------------------------------------

def print_classification_report(
    model, X, y, title="Classification Report", save=False, save_as=None
):
    """
    Prints the classification report of the model.

    Parameters
    ----------
    model: A fitted model.
        The model to evaluate.
    X: array-like of shape (n_samples, n_features)
        The data to evaluate the classification report on.
    y: array-like of shape (n_samples, )
        The true labels.
    title: str, default='Classification Report'
        The title of the plot.
    save: bool, default=False
        If True, saves the plot to the EDA plots folder.
    save_as: str, default=None
        The filename to save the plot to.
    """	
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, target_names=LABELS)
    print(title)
    print(report)

    # Save he report as text file
    if save:
        if save_as is None:
            raise ValueError(
                'Please provide a filename to save the classification report. E.g. "model_classification_report"'
            )
        filename = replace_slashes(save_as + ".txt")
        path = normpath(join(model_plots_path, filename))
        with open(path, "w") as f:
            f.write(report)
        print(f"Classification report saved to {path}")


# ------------------------------------------------------------
# Overall Model Performance Plots
# ------------------------------------------------------------

def plot_model_performance(
    model,
    X,
    y,
    calibration_bins=15,
    title="Model Performance",
    model_name="Classifier",
    figsize=(9, 8),
    save=False,
    save_as=None,
):
    """
    Plot the confusion matrix, the ROC curve, the precision-recall curve
    and the calibration curve of the model.

    Parameters
    ----------
    model: A fitted model.
        The model to evaluate.
    X: array-like of shape (n_samples, n_features)
        The data to evaluate the calibration curve on.
    y: array-like of shape (n_samples, )
        The true labels.
    calibration_bins: int, default=15
        The number of bins to use when plotting the calibration curve.
    title: str, default='Model Performance'
        The title of the plot.
    model_name: str, default='Classifier'
        The name of the model to be displayed in the plot.
    save: bool, default=False
        If True, saves the plot to the EDA plots folder.
    save_as: str, default=None
        The filename to save the plot to.

    Examples
    --------
    >>> from xgboost import XGBClassifier
    >>> from src.visualization.model_performance import plot_model_performance
    >>> model = XGBClassifier()
    >>> model.fit(X_train, y_train)
    >>> plot_model_performance(model, X_test, y_test)
    """
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    # Confusion matrix
    ConfusionMatrixDisplay.from_estimator(model, X, y, ax=axs[0, 0],  cmap=CM_CMAP)
    axs[0, 0].set_title(f"Confusion Matrix")

    # PR-curve
    PrecisionRecallDisplay.from_estimator(
        model, X, y, ax=axs[0, 1], name=model_name, plot_chance_level=True
    )
    axs[0, 1].set_title(f"Precision-Recall Curve")

    # ROC curve
    RocCurveDisplay.from_estimator(
        model, X, y, ax=axs[1, 0], name=model_name, plot_chance_level=True
    )
    axs[1, 0].set_title(f"ROC Curve")

    # Calibration curve
    CalibrationDisplay.from_estimator(
        model, X, y, n_bins=calibration_bins, name=model_name, ax=axs[1, 1]
    )
    axs[1, 1].set_title(f"Calibration Curve")

    fig.suptitle(title, fontsize=16, y=1.03)
    plt.tight_layout()

    if save:
        if save_as is None:
            raise ValueError("Please provide a filename to save the plot.")
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(model_plots_path, filename))
        plt.savefig(path, bbox_inches="tight")
        print(f"Figure saved to {path}")

    plt.show()


# ----------------------------------------------------------------
# 2. Confusion Mattrix, ROC, Precision Recall and Calibration Plots
# -------------------------------------------------------------

def plot_confusion_matrix(
    model,
    X1,
    y1,
    X2=None,
    y2=None,
    title1="Set 1",
    title2="Set 2",
    title="Confusion Matrices Comparison",
    figsize=(5, 4),
    save=False,
    save_as=None,
):
    """
    Plot the confusion matrix.

    If a second set is provided (X2, y2), plot the confusion matrix of the second set as well.

    Parameters
    ----------
    model: A fitted model.
        The model to evaluate.
    X1: array-like of shape (n_samples, n_features)
        The first set to evaluate the ROC curve on.
    y1: array-like of shape (n_samples, )
        The true labels of the first confusion matrix curve.
    X2: array-like of shape (n_samples, n_features), optional
        The second set to evaluate the ROC curve on.
    y2: array-like of shape (n_samples, ), optional
        The true labels of the second confusion matrix curve.
    title1: str, default='Set 1'
        The title of the first confusion matrix curve.
    title2: str, default='Set 2'
        The title of the second confusion matrix curve.
    title: str, default='confusion matrix Curves Comparison'
        The title of the plot.
    figsize: tuple, default=(9, 4)
        The size of the figure.
    save: bool, default=False
        If True, saves the plot to the EDA plots folder.
    save_as: str, default=None
        The filename to save the plot to.

    Examples
    --------
    >>> from xgboost import XGBClassifier
    >>> from src.visualization.model_performance import plot_confusion_matrice
    >>> model = XGBClassifier()
    >>> model.fit(X_train, y_train)
    >>> # plot a single confusion matrix
    >>> plot_confusion_matrix(model, X_train, y_train)
    >>> # plot two confusion matrices
    >>> plot_confusion_matrix(model, X_train, X_test, y_train, y_test)
    """
    n_cols = 1
    if X2 is not None and y2 is not None:
        n_cols = 2
    
    fig, axs = plt.subplots(1, n_cols, figsize=figsize)

    # Confusion Matrix 1
    ConfusionMatrixDisplay.from_estimator(model, X1, y1, ax=axs[0], cmap=CM_CMAP)
    axs[0].set_title(title1)

    # If a second set is provided, plot the confusion matrix of the second set
    if X2 is not None and y2 is not None:
        ConfusionMatrixDisplay.from_estimator(model, X2, y2, ax=axs[1], cmap=CM_CMAP)
        axs[1].set_title(title2)

    fig.suptitle(title, fontsize=16, y=1.03)
    plt.tight_layout()

    if save:
        if save_as is None:
            raise ValueError(
                'Please provide a filename to save the confusion matrix.  E.g. "model_confusion_matrix"'
            )
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(model_plots_path, filename))
        plt.savefig(path)
        print(f"Figure saved to {path}")

    plt.show()


def plot_roc_curve(
    model,
    X1,
    y1,
    X2=None,
    y2=None,
    title1="Set 1",
    title2="Set 2",
    model_name=None,
    figsize=(5, 4),
    save=False,
    save_as=None,
):
    """
    Plot the ROC curve of the model.

    If a second set is provided (X2, y2), plot the ROC curve of the second set as well.

    Parameters
    ----------
    model: A fitted model.
        The model to evaluate.
    X1: array-like of shape (n_samples, n_features)
        The first set to evaluate the ROC curve on.
    y1: array-like of shape (n_samples, )
        The true labels of the first ROC curve.
    X2: array-like of shape (n_samples, n_features), optional
        The second set to evaluate the ROC curve on.
    y2: array-like of shape (n_samples, ), optional
        The true labels of the second ROC curve.
    title1: str, default='Set 1'
        The title of the first ROC curve.
    title2: str, default='Set 2'
        The title of the second ROC curve.
    title: str, default='ROC Curves Comparison'
        The title of the plot.
    model_name: str, default=None
        The name of the model to be displayed in the plot.
    save: bool, default=False
        If True, saves the plot to the EDA plots folder.
    save_as: str, default=None
        The filename to save the plot to.

    Examples
    --------
    >>> from xgboost import XGBClassifier
    >>> from src.visualization.model_performance import plot_roc_curve
    >>> model = XGBClassifier()
    >>> model.fit(X_train, y_train)
    >>> # Plot a single ROC curve
    >>> plot_roc_curve(model, X_train, y_train)
    >>> # Plot two ROC curves
    >>> plot_roc_curve(model, X_train, X_test, y_train, y_test)
    """
    fig, axs = plt.subplots(figsize=figsize)

    # If second set is provided
    # Do not plot the chance level for the first set
    plot_chance_first = True if X2 is None and y2 is None else False

    # Plot the ROC curve of the first set
    RocCurveDisplay.from_estimator(
        model, X1, y1, ax=axs, name=title1, alpha=0.8, plot_chance_level=plot_chance_first
    )

    # If the second set is provided, plot the ROC curve of the second set
    if X2 is not None and y2 is not None:
        RocCurveDisplay.from_estimator(
            model, X2, y2, ax=axs, name=title2, alpha=0.8, plot_chance_level=True
        )

    # If model_name is None, use the model class name
    if model_name is None:
        model_name = model.__class__.__name__ 

    fig.suptitle(f'{model_name} ROC Curve', fontsize=16, y=1.03)
    plt.tight_layout()

    if save:
        if save_as is None:
            raise ValueError("Please provide a filename to save the plot.")
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(model_plots_path, filename))
        plt.savefig(path, bbox_inches="tight")
        print(f"Figure saved to {path}")

    plt.show()


def plot_precision_recall_curve(
    model,
    X1,
    y1,
    X2=None,
    y2=None,
    title1="Set 1",
    title2="Set 2",
    model_name=None,
    figsize=(5, 4),
    save=False,
    save_as=None,
):
    """
    Plot the precision-recall curve of the model.

    If a second set is provided (X2, y2), plot the precision-recall curve of the second set as well.

    Parameters
    ----------
    model: A fitted model.
        The model to evaluate.
    X1: array-like of shape (n_samples, n_features)
        The first set to evaluate the ROC curve on.
    y1: array-like of shape (n_samples, )
        The true labels of the first confusion matrix curve.
    X2: array-like of shape (n_samples, n_features), optional
        The second set to evaluate the ROC curve on.
    y2: array-like of shape (n_samples, ), optional
        The true labels of the second confusion matrix curve.
    title1: str, default='Training'
        The title of the first PR curve.
    title2: str, default='Test'
        The title of the second PR curve.
    title: str, default='PR Curves Comparison'
        The title of the plot.
    model_name: str, default=None
        The name of the model to be displayed in the plot.
    figsize: tuple, default=(9, 4)
        The size of the figure.
    save: bool, default=False
        If True, saves the plot to the EDA plots folder.
    save_as: str, default=None
        The filename to save the plot to.

    Examples
    --------
    >>> from xgboost import XGBClassifier
    >>> from src.visualization.model_performance import plot_precision_recall_curve
    >>> model = XGBClassifier()
    >>> model.fit(X_train, y_train)
    >>> # Plot a single PR curve
    >>> plot_precision_recall_curves(model, X_train, y_train)
    >>> # Plot two PR curves
    >>> plot_precision_recall_curves(model, X_train, X_test, y_train, y_test)
    """
    fig, axs = plt.subplots(figsize=figsize)

    # If second set is provided
    # Do not plot the chance level for the first set
    plot_chance_first = True if X2 is None and y2 is None else False

    # Plot the ROC curve of the first set
    PrecisionRecallDisplay.from_estimator(
        model, X1, y1, ax=axs, name=title1, alpha=0.8, plot_chance_level=plot_chance_first
    )

    # If the second set is provided, plot the ROC curve of the second set
    if X2 is not None and y2 is not None:
        PrecisionRecallDisplay.from_estimator(
            model, X2, y2, ax=axs, name=title2, alpha=0.8, plot_chance_level=True
        )
    
    # If model_name is None, use the model class name
    if model_name is None:
        model_name = model.__class__.__name__ 

    fig.suptitle(f'{model_name} Precision-Recall Curve', fontsize=16, y=1.03)
    plt.tight_layout()

    if save:
        if save_as is None:
            raise ValueError("Please provide a filename to save the plot.")
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(model_plots_path, filename))
        plt.savefig(path, bbox_inches="tight")
        print(f"Figure saved to {path}")

    plt.show()



def plot_calibration_curve(
    model,
    X1,
    y1,
    X2=None,
    y2=None,
    calibration_bins=10,
    title1="Set 1",
    title2="Set 2",
    figsize=(5, 4),
    model_name=None,
    save=False,
    save_as=None,
):
    """
    Plot the probability calibration curve.
    If a second set is provided (X2, y2), plot the calibration curve of the second set as well.

    The probability calibration curve is a plot of the true probabilities
    against the predicted probabilities.
    It shows us how well the model predicts the true probabilities.

    A perfectly calibrated model should have the predicted probabilities
    approximately equal to the true probabilities. This means that points
    on the calibration curve should lie roughly along the diagonal.

    For more information, see
    https://scikit-learn.org/stable/modules/calibration.html.

    Parameters
    ----------
    model: A fitted model.
        The model to evaluate.
    X1: array-like of shape (n_samples, n_features)
        The data to evaluate the first calibration curve on.
    y1: array-like of shape (n_samples,)
        The true labels of the first calibration curve.
    X2: array-like of shape (n_samples, n_features), optional
        The data to evaluate the second calibration curve on.
    y2: array-like of shape (n_samples,), optional
        The true labels of the second calibration curve.
    calibration_bins: int, default=10
        The number of bins to use when plotting the calibration curve.
    model_name: str, default=None
        The name of the model to be displayed in the plot.
    save: bool, default=False
        If True, saves the plot to the EDA plots folder.
    save_as: str, default=None
        The filename to save the plot to.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from src.visualization.model_performance import plot_calibration_curve
    >>> model = XGClassifier()
    >>> model.fit(X_train, y_train)
    >>> plot_calibration_curve(model, X_test, y_test)
    """
    fig, axs = plt.subplots(figsize=figsize)

    # Plot the ROC curve of the first set
    CalibrationDisplay.from_estimator(
        model, X1, y1, ax=axs, name=title1, alpha=0.8, n_bins=calibration_bins
    )

    # If the second set is provided, plot the ROC curve of the second set
    if X2 is not None and y2 is not None:
        CalibrationDisplay.from_estimator(
            model, X2, y2, ax=axs, name=title2, alpha=0.8, n_bins=calibration_bins
        )
    
    # If model_name is None, use the model class name
    if model_name is None:
        model_name = model.__class__.__name__ 

    fig.suptitle(f'{model_name} Calibration Curve', fontsize=16, y=1.03)
    plt.tight_layout()

    if save:
        if save_as is None:
            raise ValueError("Please provide a filename to save the plot.")
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(model_plots_path, filename))
        plt.savefig(path, bbox_inches="tight")
        print(f"Figure saved to {path}")

    plt.show()

    # plt.hist(
    #     plot.y_prob,
    #     range=(0, 1),
    #     bins=10,
    #     label='Uncalibrated',
    # )
    # plt.title('Model')
    # plt.xlabel("Mean predicted probability")
    # plt.ylabel("Count")
    # plt.show()


# --------------------------------------------------
# 3. Feature Importance
# --------------------------------------------------

def plot_xgboost_feature_importance(
    model,
    importance_type="gain",
    title="Feature importance (gain)",
    max_num_features=10,
    height=0.5,
    show_values=False,
    xlabel="Gain",
):
    """
    Plots the feature importance of the model.
    Note that the model should an xgboost model.

    Parameters
    ----------
    model: A fitted XGBclassifer model.
    importance_type: str, default='gain'
        The type of feature importance to plot.
        E.g. 'gain', 'weight', 'cover', 'total_gain', 'total_cover'.
    title: str, default='Feature importance (gain)'
        The title of the plot.
    max_num_features: int, default=10
        The maximum number of features to plot.
    height: float, default=0.5
        The height of the bars.
    show_values: bool, default=False
        If True, shows the values of the feature importance.
    xlabel: str, default='Gain'
        The label of the x-axis.
    """
    plot_importance(
        model,
        importance_type=importance_type,
        title=title,
        max_num_features=max_num_features,
        height=height,
        show_values=show_values,
        xlabel=xlabel,
    )



# --------------------------------------------------
# 4. Training Evaluation Plots
# --------------------------------------------------


def plot_learning_curve(
    model, X, y, scoring="roc_auc", title="Learning Curve", save=False, save_as=None
):
    """
    Plots the learning curve of the model.

    Parameters
    ----------
    model: A fitted model.
        The model to evaluate.
    X: array-like of shape (n_samples, n_features)
        The training input samples.
    y: array-like of shape (n_samples, )
        The target values.
    scoring: str, default='f1'
        The scoring metric to use.
        E.g. 'roc_auc', 'f1', 'accuracy', 'precision', 'recall'.
    title: str, default='Learning Curve'
        The title of the plot.
    save: bool, default=False
        If True, saves the plot to the EDA plots folder.
    save_as: str, default=None
        The filename to save the plot to.
    """
    LearningCurveDisplay.from_estimator(model, X, y, scoring=scoring, n_jobs=-1)
    plt.title(title, fontsize=16, y=1.03)
    plt.tight_layout()

    if save:
        if save_as is None:
            raise ValueError("Please provide a filename to save the plot.")
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(model_plots_path, filename))
        plt.savefig(path, bbox_inches="tight")
        print(f"Figure saved to {path}")

    plt.show()


def plot_validation_curves(
    model,
    X,
    y,
    params,
    scoring="roc_auc",
    title="Validation Curve",
    figsize=(9, 8),
    save=False,
    save_as=None,
):
    """
    Plots the validation curves of the model.

    Parameters
    ----------
    model: A fitted model.
        The model to evaluate.
    X: array-like of shape (n_samples, n_features)
        The training input samples.
    y: array-like of shape (n_samples, )
        The target values.
    params: list of dicts
        A list of dictionaries containing the parameter name and the range of values to be tested.
    scoring: str, default='roc_auc'
        The scoring metric to use.
        E.g. 'roc_auc', 'f1', 'accuracy', 'precision', 'recall'.
    title: str, default='Validation Curve'
        The title of the plot.
    figsize: tuple of shape (width, height), default=(9, 8)
        The size of the figure.
    save: bool, default=False
        If True, saves the plot to the EDA plots folder.
    save_as: str, default=None
        The filename to save the plot to.

    Examples
    --------
    >>> form xgb import XGBClassifier
    >>> from src.visualization.model_performance import plot_validation_curves
    >>> model = XGBClassifier()
    >>> params = [
            {'name': 'learning_rate', 'range': np.arange(0.001, 0.5, 0.05)},
            {'name': 'gamma', 'range': np.arange(0.001, 0.5, 0.05)},
            {'name': 'reg_alpha', 'range': np.arange(0.001, 0.5, 0.05)},
            {'name': 'scale_pos_weight', 'range': np.arange(1, 10, 0.5)},
        ]
    >>> plot_validation_curves(model, X_train, y_train, params)
    """
    # Calculate the number of axes
    n_params = len(params)
    n_rows = np.ceil(np.sqrt(n_params)).astype(int)
    n_cols = np.floor(np.sqrt(n_params)).astype(int)

    _, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
    axs = axs.flatten()

    for i, param in enumerate(params):
        name = param["name"]
        range = param["range"]
        ValidationCurveDisplay.from_estimator(
            model,
            X,
            y,
            param_name=name,
            param_range=range,
            scoring=scoring,
            ax=axs[i],
            n_jobs=-1,
        )

    plt.suptitle(title, fontsize=16, y=1.03)
    plt.tight_layout()

    if save:
        if save_as is None:
            raise ValueError("Please provide a filename to save the plot.")
        filename = replace_slashes(save_as + ".png")
        path = normpath(join(model_plots_path, filename))
        plt.savefig(path, bbox_inches="tight")
        print(f"Figure saved to {path}")

    plt.show()

