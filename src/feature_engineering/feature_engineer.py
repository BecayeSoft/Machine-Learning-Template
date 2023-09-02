import pandas as pd
import warnings


class FeatureEngineer:
    """
    Class that handles the creation of new features based on the existing ones.

    Attributes:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the data to be processed.
    verbose : bool
        If True, prints the progress of the feature engineering process.
    """
