import pandas as pd
import numpy as np

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




def train_test_split(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)

    n_samples = X.shape[0]

    indices = np.arange(n_samples)
    rng.shuffle(indices)

    n_test = int(n_samples * test_size)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Select rows using their integer positions
    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    # y is a Pandas Series, so .iloc works here too
    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    return X_train, X_test, y_train, y_test




def encode_categorical(X_train, X_test):
    categories = ["month-to-month", "one-year", "two-year"]

    train_encoded = X_train.copy()
    test_encoded = X_test.copy()

    train_contract = train_encoded["contract_type"]
    test_contract = test_encoded["contract_type"]

    train_encoded = train_encoded.drop(columns=["contract_type"])
    test_encoded = test_encoded.drop(columns=["contract_type"])

    for category in categories:
        train_encoded[category] = (train_contract == category).astype(int) ## 3 more columns are created for each category
        test_encoded[category] = (test_contract == category).astype(int)

    return train_encoded, test_encoded




def calculate_mean(X):
    return np.mean(X, axis=0)
    

def calculate_std(X, mean):
    variance = np.mean((X - mean) ** 2, axis=0)
    return np.sqrt(variance)


def standardize(X, mean, sd):
    """
    Standardize features using mean and standard deviation.
    
    Applies z-score normalization: (X - mean) / std_dev
    
    Parameters
    ----------
    X : numpy.ndarray
        Input features to standardize, shape (n_samples, n_features)
    mean : numpy.ndarray
        Mean values for each feature, typically calculated from training data.
        Shape (n_features,)
    sd : numpy.ndarray
        Standard deviation for each feature, typically calculated from training data.
        Shape (n_features,)
    
    Returns
    -------
    numpy.ndarray
        Standardized features with shape (n_samples, n_features).
        For each feature: (X_feature - mean_feature) / sd_feature
        
    Raises
    ------
    ValueError
        If any standard deviations are zero, indicating zero variance in that feature.
        This typically means the training data has no variation for that feature,
        which would cause division by zero.
    
    Notes
    -----
    - Mean and std should be calculated ONLY from training data
    - The same mean and std should be applied to both training and test data
    - This ensures consistent preprocessing across train and test sets
    """
    # Check for zero standard deviations (features with no variance)
    if np.any(sd == 0):
        zero_std_indices = np.where(sd == 0)[0]
        raise ValueError(
            f"Cannot standardize: feature(s) at index {zero_std_indices.tolist()} "
            f"have zero standard deviation. This indicates no variance in those "
            f"feature(s) within the training data. Standard deviations: {sd}. "
            f"Check your training data for features with constant values."
        )
    
    # Perform z-score standardization
    return (X - mean) / sd
