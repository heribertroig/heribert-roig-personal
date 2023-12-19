from datetime import datetime
from typing import List, Protocol, Tuple

import pandas as pd


class IFeatureEngineer(Protocol):
    def fit(self, df: pd.DataFrame):
        ...

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ...