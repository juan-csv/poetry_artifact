""" this file contains the data preprocessing functions """
if True:
    import sys
    sys.path.append("../../")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import HashingEncoder
from lifecycle_ml_example.src.utils.logger_class import Logger
from lifecycle_ml_example.src.utils.f_gcs_storage import get_blob_as_dataframe, upload_df_to_gcs


# instance logger
logger = Logger(__name__).logger


def get_gold_data(config: dict, prob_random_error: np.float64 = None) -> tuple:
    """
    Function to get gold data from the input dataframe.

    Args:
        config (dict): The configuration dictionary.

    Returns:
        tuple: A tuple containing the preprocessed training data, preprocessed testing data, 
                training labels, and testing labels.
    """
    # introduce random error

    # get a random value between 0 and 1
    if prob_random_error is None:
        prob_random_error = np.random.rand()
    logger.info(f"Random error: {prob_random_error}")
    if prob_random_error > 0.3:
        raise ValueError("Random error")

    logger.info("Getting gold data")

    bucket_name = config["storage"]["bucket"]
    path_raw_data = config["storage"]["path_raw_data"]
    logger.info(f"Getting raw data from {bucket_name}/{path_raw_data} ...")
    df = get_blob_as_dataframe(bucket_name, path_raw_data)

    # preprocess data
    x, y = preprocess_data(df)

    # Split data into train and test sets
    test_size = config["data"]["test_size"]
    random_state_split = config["data"]["random_state_train_test_split"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state_split)

    # Impute categorical values
    x_train.fillna(x_train.mode().iloc[0], inplace=True)
    x_test.fillna(x_train.mode().iloc[0], inplace=True)

    # encoding features
    x_train_hashing = encode_features(x_train)
    x_test_hashing = encode_features(x_test)

    # Upload preprocessed data to GCS
    logger.info("Uploading preprocessed data to GCS ...")
    path_gold_data = config["storage"]["path_gold_data"]

    upload_df_to_gcs(bucket_name, path_gold_data +
                     "/x_train.csv", x_train_hashing)
    upload_df_to_gcs(bucket_name, path_gold_data +
                     "/x_test.csv", x_test_hashing)
    upload_df_to_gcs(bucket_name, path_gold_data + "/y_train.csv", y_train)
    upload_df_to_gcs(bucket_name, path_gold_data + "/y_test.csv", y_test)

    return True


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the input DataFrame by removing columns, filling missing values, dropping duplicates, 
    creating dummy variables for age and several other features, replacing categorical variables with 
    numerical representations, and separating the input DataFrame into features (x) and target (y).

    Args:
        df: pd.DataFrame: The input DataFrame

    Returns:
        x: pd.DataFrame: The preprocessed feature DataFrame
        y: pd.Series: The preprocessed target Series
    """
    logger.info("Starting data preprocessing ...")

    # check if "Y" column exists
    if "Y" not in df.columns:
        column_y_exists = False
    else:
        column_y_exists = True

    df = df.drop(columns=['car', 'toCoupon_GEQ5min', 'direction_opp'])
    df = df.fillna(df.mode().iloc[0])
    df = df.drop_duplicates()

    df_dummy = df.copy()
    age_list = []
    for i in df['age']:
        if i == 'below21':
            age = '<21'
        elif i in ['21', '26']:
            age = '21-30'
        elif i in ['31', '36']:
            age = '31-40'
        elif i in ['41', '46']:
            age = '41-50'
        else:
            age = '>50'
        age_list.append(age)
    df_dummy['age'] = age_list

    df_dummy['passanger_destination'] = df_dummy['passanger'].astype(
        str) + '-' + df_dummy['destination'].astype(str)
    df_dummy['marital_hasChildren'] = df_dummy['maritalStatus'].astype(
        str) + '-' + df_dummy['has_children'].astype(str)
    df_dummy['temperature_weather'] = df_dummy['temperature'].astype(
        str) + '-' + df_dummy['weather'].astype(str)
    df_dummy = df_dummy.drop(columns=[
                             'passanger', 'destination', 'maritalStatus', 'has_children', 'temperature', 'weather'])

    if column_y_exists:
        df_dummy = df_dummy.drop(columns=['Y'])
        df_dummy = pd.concat([df_dummy, df['Y']], axis=1)

    df_dummy = df_dummy.drop(columns=['gender', 'RestaurantLessThan20'])
    df_le = df_dummy.replace({
        'expiration': {'2h': 0, '1d': 1},
        'age': {'<21': 0, '21-30': 1, '31-40': 2, '41-50': 3, '>50': 4},
        'education': {'Some High School': 0, 'High School Graduate': 1, 'Some college - no degree': 2,
                      'Associates degree': 3, 'Bachelors degree': 4, 'Graduate degree (Masters or Doctorate)': 5},
        'Bar': {'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4},
        'CoffeeHouse': {'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4},
        'CarryAway': {'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4},
        'Restaurant20To50': {'never': 0, 'less1': 1, '1~3': 2, '4~8': 3, 'gt8': 4},
        'income': {'Less than $12500': 0, '$12500 - $24999': 1, '$25000 - $37499': 2, '$37500 - $49999': 3,
                   '$50000 - $62499': 4, '$62500 - $74999': 5, '$75000 - $87499': 6, '$87500 - $99999': 7,
                   '$100000 or More': 8},
        'time': {'7AM': 0, '10AM': 1, '2PM': 2, '6PM': 3, '10PM': 4}
    })

    if column_y_exists:
        x = df_le.drop('Y', axis=1)
        y = df_le.Y

    logger.info("Data preprocessing complete\n")
    if column_y_exists:
        return x, y
    else:
        return df_le


def encode_features(x: pd.DataFrame, n_components: int = 27) -> pd.DataFrame:
    """
    Encode features using hashing encoder and return the transformed DataFrame.

    Args:
        x (pd.DataFrame): The input DataFrame to be encoded.
        n_components (int): The number of components for the hashing encoder. Default is 27.

    Returns:
        pd.DataFrame: The transformed DataFrame after encoding.
    """
    logger.info("Encoding features using hashing encoder ...")

    hashing_ros_enc_instance = HashingEncoder(cols=['passanger_destination', 'marital_hasChildren', 'occupation', 'coupon',
                                                    'temperature_weather'], n_components=n_components, max_process=0)
    # transform avoiding multiprocess
    hashing_ros_enc = hashing_ros_enc_instance.fit(X=x)
    # transform avoiding multiprocess
    x_test_hashing = hashing_ros_enc.transform(x.reset_index(drop=True))
    return x_test_hashing


if __name__ == "__main__":

    # load config
    from lifecycle_ml_example.src.utils.config import load_config

    config = load_config()
    # get gold data
    res = get_gold_data(config)
