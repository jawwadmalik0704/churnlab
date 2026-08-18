import pandas as pd


def load_data(path):
    """
    Load the churn dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to the dataset CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """
    return pd.read_csv(path)
