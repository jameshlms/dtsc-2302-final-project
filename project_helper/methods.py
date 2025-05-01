from pandas import DataFrame, Series
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics

import itertools
from typing import Generator, List, Literal, Sequence, Tuple
from joblib import Parallel, delayed
import math


def trn_vld_tst_split(
    features: DataFrame,
    target: Series,
    train_size: float = 0.8,
) -> Tuple[DataFrame, Series, DataFrame, Series, DataFrame, Series]:
    """Creates a tuple of train, validation, and testings splits for the features and targets provided.

    For index pairs (i, i+1):
        1. (ndarrays at index 0 and 1) contains the features selected for training and the targets selected for training.
        2. (ndarrays at index 2 and 3) contains the features selected for validation and the targets selected for validation.
        3. (ndarrays at index 4 and 5) contains the features selected for testing and the targets selected for testing.

    Args:
        features (ndarray, DataFrame): The features of the data.
        targets (ndarray, Series): The targets of the data.
        train_size (float): What portion of data should be used for training, with the remainder being evenly split for validation and testing. By default, set to 0.8

    Returns:
        splits (tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]): A tuple where
        index 0 contains training features, index 1 contains training targets,
        index 2 contains validation features, index 3 contains validation targets,
        index 4 contains testing features, and index 5 contains testing targets.

    Example:
        ```python
        X_trn, y_trn, X_vld, y_vld, X_tst, y_tst = trn_vld_tst_split(X, y)
        ```
    """
    X = features.copy()
    y = target.copy()

    if len(X) != len(y):
        raise AttributeError(
            f"'features' and 'targets' do not have the same number of rows ({len(X)} against {len(y)})"
        )

    indices = np.arange(length := len(y))
    np.random.shuffle(indices)

    vld_start = int(length * train_size)
    tst_start = int(length * ((1 - train_size) / 2)) + vld_start

    return (
        X.iloc[(trn_indices := indices[:vld_start])],
        y.iloc[trn_indices],
        X.iloc[(vld_indices := indices[vld_start:tst_start])],
        y.iloc[vld_indices],
        X.iloc[(tst_indices := indices[tst_start:])],
        y.iloc[tst_indices],
    )


def get_npa_records(
    data_frame: DataFrame, npa_seq: Sequence[int], col_name: str = "normalized"
) -> Generator[Series, None, None]:
    """
    Provides a generator for this chopped data.

    Args:
        data_frame (DataFrame):
            The dataframe with stacked NPAs and distict features in the same column.
        npa_seq (Sequence[int]):
            Sequence of NPA identifiers to process.
        col_name (str):
            Name of the column in `df` whose values will be yielded.

    Yields:
        Series:
            The values of `col_name` for each NPA, indexed by `"Var_Name_Year"`.

    Example:
        >>> # Build a DataFrame of “normalized” series for NPAs 101, 202:
        >>> result_df = DataFrame.from_records(get_npa_records(stacked_df, npa_list)
    """
    df = data_frame.copy()
    df["Var_Name_Year"] = (
        df["Normalized_Data_Name"].astype(str) + "-" + df["data_year"].astype(str)
    )
    for npa in npa_seq:
        record: DataFrame = df[df["NPA"] == npa].copy()
        record.set_index("Var_Name_Year", inplace=True)
        record["variable_id_adj"] = range(1, len(record) + 1)

        yield record[col_name]


def top_k_models(
    k: int,
    X: DataFrame,
    y: Series,
    *,
    X_vld: DataFrame = None,
    y_vld: Series = None,
    use_metric: Literal["rmse", "r2"] = "rmse",
    n_jobs: int = 1,
):
    total_combins = math.comb(len(X.columns), k)
    interval_in_tenths = max(total_combins // 10, 1)

    print(f"Total possible combination: {total_combins}")

    X_trn, y_trn = X, y

    X_val = X if X_vld is None else X_vld
    y_val = y if y_vld is None else y_vld

    if use_metric == "rmse":
        metric_func = metrics.root_mean_squared_error
        compare = np.less

    else:
        metric_func = metrics.r2_score
        compare = np.greater

    def process_subset(cols):
        X_sub = X_trn[cols]
        model = LinearRegression(fit_intercept=True).fit(X_sub, y_trn)
        y_pred = model.predict(X_val[cols])

        return cols, metric_func(y_val, y_pred)

    best_cols, best_metric = None, None

    processed = 0

    combo_iter = itertools.combinations(X_trn.columns, k)

    for per in range(1, 11):
        combin_chunk = list(itertools.islice(combo_iter, interval_in_tenths))

        if not combin_chunk:
            break

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_subset)(list(col_combin)) for col_combin in combin_chunk
        )

        for cols, metric in results:
            if best_metric is None or compare(metric, best_metric):
                best_cols, best_metric = cols, metric

        processed += len(combin_chunk)
        print(f"top_k_models progress: {min(per * 10, 100)}% done")

    leftover = list(combo_iter)
    if leftover:
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_subset)(list(col_combin)) for col_combin in leftover
        )
        for cols, metric in results:
            if metric is not None and (
                best_metric is None or compare(metric, best_metric)
            ):
                best_metric, best_cols = metric, cols
        processed += len(leftover)
        print("top_k_models progress: finishing up")

    for col_combin in itertools.combinations(X_trn.columns, k):
        cols = list(col_combin)

        X_sub = X_trn[cols]
        X_val_sub = X_val[cols]

        model = LinearRegression(fit_intercept=True).fit(X_sub, y_trn)
        y_pred = model.predict(X_val_sub)

        metric = metric_func(y_val, y_pred)

        if compare(metric, best_metric):
            best_metric, best_cols = metric, cols

    X_trn, y_trn, X_val, y_val, X_sub, X_val_sub = [None] * 6

    return best_metric, best_cols


def top_k_feats(
    k: int,
    X: DataFrame,
    y: Series,
    *,
    X_vld: DataFrame = None,
    y_vld: Series = None,
    use_metric: Literal["rmse", "r2"] = "rmse",
    n_jobs: int = 1,  # One core by default just to be safe
    candidates: int = 1,
):
    num_combins = math.comb(len(X.columns), k)
    interval = max(num_combins // 10, 1)

    print(f"Total possible combinations: {num_combins}")

    X_trn, y_trn = X, y

    X_val = X if X_vld is None else X_vld
    y_val = y if y_vld is None else y_vld

    if use_metric == "r2":
        metric_func = metrics.root_mean_squared_error
        compare = np.less

    else:
        metric_func = metrics.r2_score
        compare = np.greater

    def process_subset(cols):
        X_sub = X_trn[cols]
        model = LinearRegression(fit_intercept=True).fit(X_sub, y_trn)
        y_pred = model.predict(X_val[cols])

        return cols, metric_func(y_val, y_pred)

    best_cols, best_metric = None, None

    processed = 0

    combo_iter = itertools.combinations(X_trn.columns, k)

    for _ in range(1, 11):
        combin_chunk = list(itertools.islice(combo_iter, interval))
        processed += len(combin_chunk)

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_subset)(list(col_combin)) for col_combin in combin_chunk
        )

        for cols, metric in results:
            if best_metric is None or compare(metric, best_metric):
                best_cols, best_metric = cols, metric

        print(f"Progress: {(processed / num_combins):.0%}")

    if leftover := list(combo_iter):
        processed += len(leftover)

        results = Parallel(n_jobs=n_jobs)(
            delayed(process_subset)(list(col_combin)) for col_combin in combin_chunk
        )

        for cols, metric in results:
            if best_metric is None or compare(metric, best_metric):
                best_cols, best_metric = cols, metric

        print("Progress: Finishing up...")

    X_trn, y_trn, X_val, y_val, X_sub, X_val_sub = [None] * 6

    return best_metric, best_cols


def get_recent_cols(dataframe: DataFrame) -> List[str]:
    var_prefix: str = ""
    prev_col: str = ""
    usecols: List[str] = []
    for col in dataframe.columns:
        if len(split := col.split("-")) != 2 and not split[1].isnumeric():
            continue

        prefix: str = split[0]

        if var_prefix != prefix and prev_col:
            usecols.append(prev_col)

        prev_col = col
        var_prefix = prefix

    return usecols


def get_2023_cols(dataframe: DataFrame) -> List[str]:
    usecols = []

    for col in dataframe.columns:
        if len(split := col.split("-")) > 1 and split[-1] == "2023":
            usecols.append(col)

    return usecols


def standardize_df(dataframe: DataFrame) -> DataFrame:
    return (dataframe - dataframe.mean()) / dataframe.std()
