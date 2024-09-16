import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple, List

def split_data(raw_df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits the raw dataframe into training and validation datasets.

    Args:
        raw_df (pd.DataFrame): The raw input dataframe.
        target_col (str): The name of the target column.

    Returns:
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]: Train inputs, train targets, val inputs, val targets.
    """
    train_df, val_df = train_test_split(raw_df, test_size=0.2, random_state=42, stratify=raw_df[target_col])

    input_cols = list(train_df.columns)
    input_cols.remove(target_col)
    train_inputs, train_targets = train_df[input_cols].copy(), train_df[target_col].copy()
    val_inputs, val_targets = val_df[input_cols].copy(), val_df[target_col].copy()

    return train_inputs, train_targets, val_inputs, val_targets

def select_features(df: pd.DataFrame, num_cols_to_drop: List[str], cat_cols_to_drop: List[str]) -> Tuple[List[str], List[str]]:
    """
    Selects numerical and categorical features from the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe.
        num_cols_to_drop (List[str]): List of numerical columns to drop.
        cat_cols_to_drop (List[str]): List of categorical columns to drop.

    Returns:
        Tuple[List[str], List[str]]: Lists of numerical and categorical column names.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = list(filter(lambda x: x not in num_cols_to_drop, numeric_cols))
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols = list(filter(lambda x: x not in cat_cols_to_drop, categorical_cols))

    return numeric_cols, categorical_cols

def encode_features(train_df: pd.DataFrame, val_df: pd.DataFrame, categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], OneHotEncoder]:
    """
    Encodes categorical features using one-hot encoding.

    Args:
        train_df (pd.DataFrame): The training dataframe.
        val_df (pd.DataFrame): The validation dataframe.
        categorical_cols (List[str]): List of categorical column names.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[str], OneHotEncoder]:
        DataFrames with encoded features, list of encoded column names, and the encoder.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[categorical_cols])
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))

    train_df[encoded_cols] = encoder.transform(train_df[categorical_cols])
    val_df[encoded_cols] = encoder.transform(val_df[categorical_cols])

    return train_df, val_df, encoded_cols, encoder

def scale_features(train_df: pd.DataFrame, val_df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """
    Scales numerical features using Min-Max scaling.

    Args:
        train_df (pd.DataFrame): The training dataframe.
        val_df (pd.DataFrame): The validation dataframe.
        numeric_cols (List[str]): List of numerical column names.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]: DataFrames with scaled numerical features and the scaler.
    """
    scaler = MinMaxScaler()
    scaler.fit(train_df[numeric_cols])

    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])

    return train_df, val_df, scaler

def preprocess_data(raw_df: pd.DataFrame, target_col: str, num_cols_to_drop: List[str], cat_cols_to_drop: List[str], scale_numeric: bool = False) -> Tuple[Dict[str, pd.DataFrame], List[str], MinMaxScaler, OneHotEncoder]:
    """
    Prepares the data for training and validation.

    Args:
        raw_df (pd.DataFrame): The raw input dataframe.
        target_col (str): The name of the target column.
        num_cols_to_drop (List[str]): List of numerical columns to drop.
        cat_cols_to_drop (List[str]): List of categorical columns to drop.
        scale_numeric (bool): Flag to indicate whether to scale numerical features.

    Returns:
        Tuple[Dict[str, pd.DataFrame], List[str], MinMaxScaler, OneHotEncoder]:
        Dictionary containing training and validation inputs and targets, list of input column names, scaler, and encoder.
    """
    train_inputs, train_targets, val_inputs, val_targets = split_data(raw_df, target_col)
    numeric_cols, categorical_cols = select_features(train_inputs, num_cols_to_drop, cat_cols_to_drop)
    train_inputs, val_inputs, encoded_cols, encoder = encode_features(train_inputs, val_inputs, categorical_cols)

    scaler = None
    if scale_numeric:
        train_inputs, val_inputs, scaler = scale_features(train_inputs, val_inputs, numeric_cols)

    input_cols = numeric_cols + encoded_cols

    result = {
        'train_X': train_inputs[input_cols],
        'train_y': train_targets,
        'val_X': val_inputs[input_cols],
        'val_y': val_targets
    }

    return result, input_cols, scaler, encoder

def preprocess_new_data(new_data: pd.DataFrame, input_cols: List[str], scaler: MinMaxScaler, encoder: OneHotEncoder) -> pd.DataFrame:
    """
    Preprocesses new data using the provided scaler and encoder.

    Args:
        new_data (pd.DataFrame): The new input dataframe to preprocess.
        input_cols (List[str]): List of input column names used in training.
        scaler (MinMaxScaler): The scaler fitted on the training data.
        encoder (OneHotEncoder): The encoder fitted on the training data.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    # Select numerical and categorical columns
    numeric_cols = [col for col in input_cols if col in new_data.select_dtypes(include=np.number).columns]
    categorical_cols = [col for col in new_data.columns if col in encoder.feature_names_in_]

    # Encode categorical features
    encoded_cols = list(encoder.get_feature_names_out(categorical_cols))
    new_data[encoded_cols] = encoder.transform(new_data[categorical_cols])

    # Scale numerical features
    if scaler:
        new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])

    return new_data[input_cols]
