from sklearn.preprocessing import (
    StandardScaler,
    PowerTransformer,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    MaxAbsScaler,
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import set_config

set_config(transform_output="pandas")  # Set transformers outputs to pandas DataFrame

import pandas as pd
import numpy as np
from math import ceil
import warnings

# TODO Column transformer for both encoding  and scaling
# TODO Use pipeline instead


class Preprocessor:
    """
    The DataPreprocessor class contains methods for cleaning and preparing the data for training.
    It contains 3 types of methods:
    * Data Quality: These methods correct discrepancies in the data to ensure quality.
    * Data Preparation: These methods prepare the data for training.
    * Data Transformation: These methods transform the data before training.
    * Utility: These methods provide general-purpose helper functionality.

    Remarks:
    --------
    When performing exploratory data analysis (EDA), the term 'columns' was used.
    However, when preparing the data for training, the term 'features' is used since it conveys the purpose better.


    Attributes:
    -----------
    `df`: DataFrame
        The DataFrame to be preprocessed.

    `X`: 2D array
        A list of features to be used for training. The features will undergo preprocessing steps.

    `y`: 1D array
        The target value or output variable associated with the features.
        It is used for training machine learning models.

    Class Attributes:
    `verbose`: bool
        If True, prints progress messages.
    """

    verbose = True

    def __init__(self, df, verbose=True):
        """
        Initializes the DataPreprocessor.

        Parameters:
        -----------
        df: DataFrame
            The DataFrame to preprocess.
        """
        self.df = df
        self.X = None  # Features
        self.y = None  # Target

        Preprocessor.verbose = verbose  # Print progress messages

    # --------------------------------------------------------------------
    # 1. Data Quality
    # These functions correct discrepancies in the data to ensure quality.
    # --------------------------------------------------------------------

    def has_missing_values(self):
        """
        Checks if there are any missing values in the data.

        Returns:
        -------
        bool
            True if missing values are present, False otherwise.
        """
        has_missing = self.df.isnull().any().any()
        print("Missing values found." if has_missing else "No missing values found.")
        return has_missing

    def get_missing_columns(self):
        """
        Returns the columns with missing values.

        Returns:
        -------
        missing_columns: list
            List of columns with missing values.
        """
        nulls = self.df.isnull().sum()
        missing_columns = nulls[nulls > 0]
        return missing_columns.index.tolist()

    def check_missing_values(self):
        """
        Checks for missing values in the data and prints the columns with missing values.

        Returns:
        -------
        None
        """
        nulls = self.df.isnull().sum()
        missing_columns = nulls[nulls > 0]

        if len(missing_columns) > 0:
            print(f"{missing_columns.shape[0]} column(s) with missing value(s) found:")
            print(missing_columns)
        else:
            print("No missing values found.")


    def fill_missing_values(self, config):
        """
        Fills missing values in the specified feature with the specified value.

        Parameters:
        -----------
        config: list of dict
            The configuration to fill the missing values.
            It can be found in config/data.yaml > Preprocessing > missing_features.

        Examples
        --------
        >>> from src.data.preprocessor import Preprocessor
        >>> from src.utils.utils import load_config, print_config
        >>> # Read the config
        >>> prep_config = load_config(path='../config/data.yaml')['preprocessing']
        >>> missing_features = prep_config['missing_features']
        >>> print_config(missing_features)
        ... [{'feature_name': 'Gender', 'fill_value': 'Unknown'}]
        >>> # Assuming the data was loaded
        >>> preprocessor = Preprocessor(df)
        >>> preprocessor.fill_missing_values(config=missing_features)
        """
        for missing_feature in config:
            feature = missing_feature["feature_name"]
            fill_value = missing_feature["fill_value"]

            if Preprocessor.verbose:
                print(
                    f"{self.df[feature].isnull().sum()} missing value(s) found in {feature}. \
                    \nFilling with {fill_value}..."
                )

            self.df[feature].fillna(fill_value, inplace=True)

            if Preprocessor.verbose:
                print(f"{self.df[feature].isnull().sum()} missing value(s) left.")

    # -------------------------------------------------
    # 2. Data Preparation
    # These functions prepare the data before training.
    # -------------------------------------------------

    def drop_features(self, features_to_drop):
        """
        Drops the specified features from the DataFrame.

        Parameters:
        -----------
        features_to_drop: list
                The features to drop.
        """
        if Preprocessor.verbose:
            print(
                f"{self.df.shape[1]} column(s) found. Dropping {len(features_to_drop)} features..."
            )

        self.df.drop(columns=features_to_drop, inplace=True)

        if Preprocessor.verbose:
            print(f"{self.df.shape[1]} column(s) left.")


    def drop_values(self, config):
        """
        Remove the rows that contain the specified values in the specified feature in the DataFrame.

        For each dictionary in the list, the method will retrieve the feature name
        and drop the specified values from that feature.

        Parameters
        ----------
        config : list of dict
            The configuration to drop the values.
            It can be found in config/data.yaml > Preprocessing > values_to_drop.

        Examples
        --------
        >>> from src.data.preprocessor import Preprocessor
        >>> from src.utils.utils import load_config, print_config
        >>> # Read the config
        >>> prep_config = load_config(path='../config/data.yaml')['preprocessing']
        >>> values_to_drop = prep_config['values_to_drop']
        >>> print_config(values_to_drop)
        ... [{'feature_name': 'Generation', 'values': ['X, boomers']},
        ... {'feature_name': 'Planet', 'values': ['Wakanda']}]
        >>> # Assuming the data was loaded
        >>> preprocessor = Preprocessor(df)
        >>> preprocessor.drop_values(config=values_to_drop)
        """
        for value_to_drop in config:
            feature = value_to_drop["feature_name"]
            values = value_to_drop["values"]

            filter = self.df[self.df[feature].isin(values)]
            self.df.drop(labels=filter.index, inplace=True)

            if Preprocessor.verbose:
                print(
                    f"Removed {len(filter.index)} row(s) with values {values} from '{feature}'..."
                )

    def rename_features(self, config):
        """
        Rename the specified features in the DataFrame.

        Parameters
        ----------
        config : list of dict
            The configuration to rename the features.
            It can be found in config/data.yaml > Preprocessing > features_to_rename.

        Examples
        --------
        >>> from src.data.preprocessor import Preprocessor
        >>> from src.utils.utils import load_config, print_config
        >>> # Read the config
        >>> prep_config = load_config(path='../config/data.yaml')['preprocessing']
        >>> features_to_rename = prep_config['features_to_rename']
        >>> print_config(features_to_rename)
        ... [{'new_name': 'Feature X',
        ...    'old_name': 'X'},
        ...    {'new_name': 'Feature Y',
        ...    'old_name': 'Y'},]
        >>> # Assuming the data was loaded
        >>> preprocessor = Preprocessor(df)
        >>> preprocessor.rename_features(config=features_to_rename)
        """
        for feature_to_rename in config:
            old_name = feature_to_rename["old_name"]
            new_name = feature_to_rename["new_name"]

            self.df.rename(columns={old_name: new_name}, inplace=True)

            if Preprocessor.verbose:
                print(f"Renamed '{old_name}' to '{new_name}'...")

    def merge_categories(self, config):
        """
        Merge the categories in a new categories.

        For each feature in the list of dict `merge_config`, the method
        loops through each mapping, then replace the old category by the new one.

        Parameters
        ----------
        config: list of dict
            The merging configuration.
            It can be found in config/data.yaml > preprocessing > categories_to_merge

        Examples
        --------
        >>> from src.data.preprocessor	import Preprocessor
        >>> from src.utils.utils import load_config
        >>> # Read the config
        >>> prep_config = load_config(path='../config/data.yaml')['preprocessing']
        >>> categories_to_merge = prep_config['categories_to_merge']
        >>> print_config(categories_to_merge)
        ... [
                {
                    'feature_name': 'Contribution_Continuous_Feedback.Performance Score (Signed)',
                    'mappings': [
                        {'new': 'Not Completed', 'old': 'Manager Re-assessment'},
                        {'new': 'ND', 'old': 'NI'}
                    ]
                },
                {
                    'feature_name': 'Band',
                    'mappings': [
                        {'new': '12', 'old': '12+'}
                    ]
                }
            ]
        >>> # Assuming the data was loaded
        >>> preprocessor = Preprocessor(df)
        >>> preprocessor.merge_categories(config=categories_to_merge)
        """
        for to_merge in config:
            feature = to_merge["feature_name"]
            mappings = to_merge["mappings"]

            for mapping in mappings:
                old = mapping["old"]
                new = mapping["new"]
                old_count = self.df[feature].value_counts()[old]
                self.df[feature].replace({old: new}, inplace=True)

                if Preprocessor.verbose:
                    print(
                        f'Replaced {old_count} "{old}" value(s) by "{new}" in "{feature}".'
                    )

    def separate_features_target(self, target_name):
        """
        Separates the features and target columns from the DataFrame.

        Returns:
        -------
            A DataFrame of the features.
            A DataFrame of the target.
        """
        self.X = self.df.drop(columns=[target_name])
        self.y = self.df[target_name]

        if Preprocessor.verbose:
            print(f"Features: {self.X.shape} ---> Target: {self.y.shape}")


    @staticmethod
    def split_train_test_val(
        X, y, test_size=0.15, val_size=0.10, stratify=True, random_state=42
    ):
        """
        Split the `X` and `y` into training, test and validation sets.

        Parameters
        ----------
        X : array-like
            The features.
        y : array-like
            The target.
        test_size : float, default=0.15
            The proportion of the dataset to include in the test split.
        val_size : float, default=0.10
            The proportion of the dataset to include in the validation split.
        stratify : bool, default=True
            If True, stratify the data before splitting.
            Stratifing ensures that the original proportion of classes in the training,
            test and validation sets is preserved.
        random_state : int, default=42
            Controls the shuffling applied to the data before applying the split.

        Returns
        -------
        X_train, X_test, X_val : array-like
            The training, test and validation sets of the features.
        y_train, y_test, y_val : array-like
            The training, test and validation sets of the target.
        """
        if stratify:
            stratify = y
        else:
            stratify = None

        test_val_size = test_size + val_size
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_val_size, stratify=stratify, random_state=random_state
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_test,
            y_test,
            test_size=val_size / test_val_size,
            stratify=y_test,
            random_state=random_state,
        )

        return X_train, X_test, X_val, y_train, y_test, y_val

    # ------------------------------------------------
    # 3. Data Transformation
    # These functions transform the data.
    # ------------------------------------------------

    @staticmethod
    def log_transform(X, features):
        """
        Applies log transformation to the specified features.

        Parameters
        ----------
        X: DataFrame
            The data containing the features to transform.
        features : list
            The features to transform.

        Returns
        -------
        X: DataFrame
            The transformed data.
        """
        for feature in features:
            X[feature] = np.log1p(X[feature])

        return X


    @staticmethod
    def create_category_encoder(encoding_config):
        """
        Creates a ColumnTransformer for encoding categorical features.
        This transformer can then be used in a pipeline.

        This methods creates two transformers:
        - OneHotEncoder for nominal features
        - OrdinalEncoder for ordinal features

        If there are no nominal or no ordinal features, their respective
        transformers will do nothing.

        Parameters
        ----------
        encoding_config : dict
            The encoding configuration dictionary.
            This can be found in the data.yaml file under
            preprocessing.encoding.

        Returns
        -------
        category_encoder : ColumnTransformer
            The ColumnTransformer for encoding categorical features.

        Notes
        -----
        The remaining features are unchanged `remainder='passthrough'`
        and the feature names are kept `verbose_feature_names_out=False`.

        Examples
        --------
        First, load the config file:
        >>> from src.utils.utils import load_config
        >>> prep_config = load_config('../config/data.yaml')
        >>> encoding_config = prep_config['preprocessing']['encoding']
        >>> 1. Create a Preprocessor instance
        >>> from src.data.preprocessor import Preprocessor
        >>> preprocessor = Preprocessor(df)
        >>> 2. Encode the categories
        >>> encoder = preprocessor.create_category_encoder(encoding_config)
        >>> X_encoded = encoder.fit_transform(X)

        # TODO
        Or, you can use a pipeline:
        >>> from sklearn.pipeline import make_pipeline
        >>> pipeline = make_pipeline(category_encoder, model)
        >>> pipeline.fit_transform(X, y)
        """
        ordinal_features = encoding_config["ordinal"]["feature_names"]
        ordinal_mappings = encoding_config["ordinal"]["mappings"]

        onehot_features = encoding_config["one_hot"]["feature_names"]

        category_encoder = ColumnTransformer(
            transformers=[
                (
                    "ordinal-encoder",
                    OrdinalEncoder(
                        categories=ordinal_mappings, handle_unknown='ignore', dtype=int
                    ),
                    ordinal_features,
                ),
                (
                    "one-hot-encoder",
                    OneHotEncoder(
                        drop='first',
                        handle_unknown='ignore',
                        sparse_output=False,
                        dtype=int,
                    ),
                    onehot_features,
                ),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        # If the verbose class atrirbute is True, print a message
        if Preprocessor.verbose:
            print(
                f'Created "one-hot-encoder" for feature(s): {onehot_features} \n \
                and "ordinal-encoder" for feature(s): {onehot_features}".'
            )

        return category_encoder


    def encode_target(self):
        """Encode target."""
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)


    # TODO update doc
    @staticmethod
    def create_scaler(scaler_type, features):
        """
        Creates a ColumnTransformer for scaling features.

        Parameters
        ----------
        scaler_type : str
            The type of scaler to use. Supported types are 'standard', 'min-max', 'robust', 'power' and 'quantile'.
        feature_names : list
            The indices of the features to scale.

        Returns
        -------
        scaler : ColumnTransformer
            The ColumnTransformer for scaling features.

        Examples
        --------
        >>> # 1. Create a Preprocessor instance
        >>> from src.data.preprocessor import Preprocessor
        >>> preprocessor = Preprocessor(df)

        >>> # 2. Encode the categories
        >>> encoder = preprocessor.create_category_encoder(encoding_config)
        >>> X_encoded = encoder.fit_transform(X)

        >>> # 3. Split the data
        >>> X_train, X_test, X_val, y_train, y_test, y_val = preprocessor.split_train_test_val(
        >>>    X_encoded, y, test_size, val_size
        >>> )

        >>> # 4. Get the indices of the remainder features & pass it to the scaler
        >>> remainder_slice = encoder.output_indices_['remainder']
        >>> remainder_indices = list(range(remainder_slice.start, remainder_slice.stop))
        >>> remainder_indexes.insert(0, 0)   # Add the 'Band' index
        >>> scaler = preprocessor.create_scaler('standard', remainder_indices)

        >>> # 5. Fit the scaler to the training set and apply it to the test and validation sets
        >>> X_train_scaled = scaler.fit_transform(X_train)
        >>> X_test_scaled = scaler.transform(X_test)
        >>> X_val_scaled = scaler.transform(X_val)
        """
        scaler_ = None

        if scaler_type == "standard":
            scaler_ = StandardScaler()
        elif scaler_type == "min-max":
            scaler_ = MinMaxScaler()
        elif scaler_type == "max-abs":
            scaler_ = MaxAbsScaler()
        elif scaler_type == "robust":
            scaler_ = RobustScaler()
        elif scaler_type == "power":
            scaler_ = PowerTransformer()
        elif scaler_type == "quantile":
            scaler_ = QuantileTransformer()
        else:
            raise ValueError(
                "Unknown scaler type. Supported types are 'standard', 'min-max', 'max-abs', 'robust', 'power' and 'quantile'."
            )

        scaler = ColumnTransformer(
            transformers=[(f"{scaler_type}-scaler", scaler_, features)],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        # If the verbose class atrirbute is True, print a message
        if Preprocessor.verbose:
            print(
                f'Created "{scaler_type}-scaler" for feature(s) at position: {features}'
            )

        return scaler

    # --------------------
    # Utility functions
    # --------------------

    def get_features_by_type(self, feature_type, exclude=None):
        """
        Returns the features of the given DataFrame based on the specified type.

        Parameters:
        -----------
        feature_type: str
            The type of features to retrieve. Supported types are 'numerical', 'categorical', 'bool' and 'date'.
        exclude: str or list, default=None
            Feature(s) to exclude from the returned list.
            This is primarily meant to exclude the target.
            However, it can be used to exclude any feature.

        Returns:
        -------
            A list of features of the specified type from `df`.


        Notes
        -----
        This function will return booleans along with objects when feature_type='categorical'.
        This is to simplify the Exploratory Data Analysis. However, it is possible to select
        booleans only by setting feature_type='bool'.

        Examples
        --------
        >>> from src.data.preprocessor import Preprocessor
        >>> preprocess = Preprocessor(df)
        >>> preprocessor.get_cat_features_by_unique_values(min_unique=30, exclude='Feature X')
        """
        # Create a copy of the DataFrame to avoid modifying the original
        self.separate_features_target()
        X = self.X.copy()

        if exclude is not None:
            # If exclude is a string, convert it to a list
            if isinstance(exclude, str):
                exclude = [exclude]
            X.drop(columns=exclude, inplace=True)

        if feature_type == "numerical":
            return X.select_dtypes(include=["number"]).columns.tolist()
        elif feature_type == "categorical":
            return X.select_dtypes(
                include=["object", "bool", "category"]
            ).columns.tolist()
        elif feature_type == "bool":
            return X.select_dtypes(include=["bool"]).columns.tolist()
        elif feature_type == "date":
            return X.select_dtypes(
                include=["datetime64", "datetime", "timedelta64", "timedelta"]
            ).columns.tolist()
        else:
            raise ValueError(
                "Invalid feature_type. Supported types are 'numerical', 'categorical', 'bool' and 'date'."
            )


    def get_cat_features_by_unique_values(
        self, min_unique=0, max_unique=np.inf, exclude=None
    ):
        """
        Returns categorical features based on the number of unique values.

        Parameters
        ----------
        min_unique: number, default=0
            minimum number of unique values a column should have to be kept.
        max_unique: number, default=30
            maximum number of unique values a column should have to be kept.
        exclude: list, default=None
            List of features to exclude from the returned list.

        Returns
        -------
        filtered_categories: array
            array of columns names with unique categories between `min_unique` and `max_unique`.

        Examples
        --------
        >>> preprocessor = Preprocessor(df)
        >>> categories = preprocessor.get_cat_features_by_unique_values(min_unique=30, max_unique=100)
        """
        categories = self.get_features_by_type("categorical", exclude=exclude)
        filtered_categories = [
            cat
            for cat in categories
            if len(self.df[cat].unique()) >= min_unique
            and len(self.df[cat].unique()) <= max_unique
        ]

        return filtered_categories
