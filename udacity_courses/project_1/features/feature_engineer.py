from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from udacity_courses.project_1.features.ifeature_engineer import IFeatureEngineer
from udacity_courses.project_1.utils.utils import convert_to_float, number_of_days_passed


class FeatureEngineer(IFeatureEngineer):
    def __init__(self, cols_to_remove: List[str] = [],
                     cols_to_change_type: List[str] = [], date_columns: List[str] = [],
                     boolean_columns: List[str] = [], categorical_cols: List[str] = [],
                 cols_to_fill_with_mean: List[str] = [], cols_to_fill_with_imputer: List[str] = [],
                 cols_to_fill_with_0: List[str] = [],
                     scale: bool = True):
        self.cols_to_remove = cols_to_remove
        self.cols_to_change_type = cols_to_change_type
        self.date_columns = date_columns
        self.boolean_columns = boolean_columns
        self.categorical_cols = categorical_cols
        self.numeric_columns = []
        self.columns_with_high_missings = []
        self.cols_to_fill_with_mean = cols_to_fill_with_mean
        self.cols_to_fill_with_imputer = cols_to_fill_with_imputer
        self.cols_to_fill_with_0 = cols_to_fill_with_0
        self.encoded_columns = []
        self.fitted_columns = []
        self.common_columns = []
        self.scale = scale
        self.day = "2017-12-31"

        self.KNN_imputer = KNNImputer(n_neighbors=2)
        if self.scale:
            self.scaler = StandardScaler()


    def fit(self, df: pd.DataFrame) -> None:
        df = df.copy()

        df = self._preprocess_data(df)

        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in (self.boolean_columns + self.cols_to_remove
                                                                         + self.date_columns + self.categorical_cols
                                                                         + self.columns_with_high_missings)]
        self.numeric_columns = numeric_columns
        df = df.dropna(subset=numeric_columns)
        self.KNN_imputer.fit(df[numeric_columns])

        if self.scale:
            df[numeric_columns] = self.scaler.fit(df[numeric_columns])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.cols_to_remove:
            df = self._remove_columns(df, self.cols_to_remove)
            print(f"Columns {self.cols_to_remove}  removed")

        if self.cols_to_change_type:
            df = self._change_dtype(df)
            print(f"{len(self.cols_to_change_type)} columns with changed to float type")

        if self.date_columns:
            df = self._transform_date_columns(df)
            print(
                f"{len(self.date_columns)} date columns removed after get the number of days elapsed until the day {self.day}")

        if self.boolean_columns:
            df = self._transform_bool_columns(df)
            print(f"{len(self.boolean_columns)} boolean columns changed to bool type")

        if self.categorical_cols:
            df = self._get_dummies_(df)
            # df = self._handle_common_columns(df)
            columns_to_keep = [col for col in df.columns if col in self.fitted_columns]
            df = df[columns_to_keep]
            df[self.encoded_columns] = df.reindex(columns=self.encoded_columns, fill_value=0)[self.encoded_columns]
            print(f"{len(self.date_columns)} categorical columns removed after getting dummies")

        if self.columns_with_high_missings:
            df.drop(columns=self.columns_with_high_missings, inplace=True)

        if self.cols_to_fill_with_mean:
            df = self._fill_with_mean(df)

        if self.cols_to_fill_with_imputer:
            df = self._fill_with_imputer(df)

        if self.cols_to_fill_with_0:
            df[self.cols_to_fill_with_0] = df[self.cols_to_fill_with_0].fillna(0)

        df[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        df = self.transform(df)
        return df

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        columns_with_high_missings = self.get_columns_with_high_missing(df, 0.95)
        self.columns_with_high_missings = [col for col in columns_with_high_missings if col not in self.cols_to_remove]
        df = self._change_dtype(df)
        df = self._fill_with_mean(df)
        encoded_df = self._get_dummies_(df)
        self._get_encoded_cols(df, encoded_df)
        return df

    def _remove_columns(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = df.copy()
        df.drop(columns=columns, inplace=True)
        return df

    def _change_dtype(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.cols_to_change_type:
            df[col] = df[col].apply(convert_to_float)
        return df

    def _transform_date_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.date_columns:
            df[f"{col}_days"] = np.nan
            df[f"{col}_days"] = number_of_days_passed(df[col], self.day)
            # We fill with mean because we don't want to lose the information of the other columns
            df[f"{col}_days"] = df[f"{col}_days"].fillna(df[f"{col}_days"].mean())
        df.drop(columns=self.date_columns, inplace=True)
        return df

    def _transform_bool_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.boolean_columns:
            df[col] = df[col].map({"f": False, "t": True})
            df[col] = df[col].astype(bool)
        return df

    def _get_dummies_(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = pd.get_dummies(df, columns=self.categorical_cols, drop_first=True)
        return df

    def _fill_with_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in self.cols_to_fill_with_mean:
            df[col] = df[col].fillna(df[col].mean())
        return df

    def _fill_with_imputer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[self.cols_to_fill_with_imputer] = (
            pd.DataFrame(self.KNN_imputer.transform(df[self.numeric_columns]), columns=self.numeric_columns)
            .set_index(df.index)
        )[self.cols_to_fill_with_imputer]
        return df

    def _get_encoded_cols(self, original_df: pd.DataFrame, encoded_df: pd.DataFrame) -> None:
        original_cols = original_df.columns.tolist()
        encoded_cols = [col for col in encoded_df.columns if col not in original_cols]
        self.fitted_columns = encoded_df.columns.tolist()
        self.encoded_columns = encoded_cols

    def _handle_common_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        columns_to_keep = [col for col in df.columns if col in self.fitted_columns]
        if not self.common_columns:
            self.common_columns = set(columns_to_keep) & set(df.columns)

        return df


    @staticmethod
    def get_columns_with_high_missing(df, missing_threshold):
        """
        Get the columns of a DataFrame that have missing values exceeding the specified threshold.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - missing_threshold (float): The threshold for missing values.

        Returns:
        - list: A list of column names with missing values exceeding the threshold.
        """
        # Calculate the percentage of missing values for each column
        missing_percentages = df.isnull().mean()

        # Filter columns with more than the threshold of missing values
        columns_with_high_missing = missing_percentages[missing_percentages > missing_threshold].index.tolist()

        return columns_with_high_missing


