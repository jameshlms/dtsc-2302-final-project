from pandas import DataFrame, Series
from numpy import ndarray
import numpy as np

from typing import Generator, Sequence, Tuple


def trn_vld_tst_split(
    features: ndarray | DataFrame,
    target: ndarray | Series,
    train_size: float = 0.8,
) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
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
    if isinstance(features, DataFrame):
        X = features.to_numpy()
    elif isinstance(features, ndarray):
        X = features.copy()
    else:
        raise TypeError("'features' is not of type DataFrame or ndarray")

    if isinstance(target, Series):
        y = target.to_numpy()
    elif isinstance(target, ndarray):
        y = target.copy()
    else:
        raise TypeError("'target' is not of type Series or ndarray")

    if len(X) != len(y):
        raise AttributeError(
            f"'features' and 'targets' do not have the same number of rows ({len(X)} against {len(y)})"
        )

    indices = np.arange(length := len(y))
    np.random.shuffle(indices)

    vld_start = int(length * train_size)
    tst_start = int(length * ((1 - train_size) / 2)) + vld_start

    return (
        X[(trn_indices := indices[:vld_start])],
        y[trn_indices],
        X[(vld_indices := indices[vld_start:tst_start])],
        y[vld_indices],
        X[(tst_indices := indices[tst_start:])],
        y[tst_indices],
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
