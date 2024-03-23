""" this file contains the data preprocessing functions """
if True:
    import sys
    sys.path.append("../../")
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle
from lifecycle_ml_example.src.utils.logger_class import Logger
from lifecycle_ml_example.src.utils.f_gcs_storage import get_blob_as_dataframe, upload_blob


# instance logger
logger = Logger(__name__).logger


def make_training(config) -> bool:
    """
    Train a machine learning model using the specified model name and training data.

    Parameters:
        config (dict): The configuration dictionary.

    Returns:
        object: The trained machine learning model.
    """
    model_name = config["model"]["algorithm"]
    logger.info(f"Get training data for {model_name} model ...")
    bucket_name = config["storage"]["bucket"]
    path_gold_data = config["storage"]["path_gold_data"]

    # get data from GCS
    x_train = get_blob_as_dataframe(
        bucket_name, path_gold_data + "/x_train.csv")
    y_train = get_blob_as_dataframe(
        bucket_name, path_gold_data + "/y_train.csv")

    logger.info(f"Training {model_name} model.")
    if model_name == 'logistic':
        model = LogisticRegression(random_state=42)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(random_state=42)
    elif model_name == 'knn':
        model = KNeighborsClassifier()
    elif model_name == 'xgboost':
        model = XGBClassifier(random_state=42, use_label_encoder=False,
                              learning_rate=0.2, n_estimators=45, max_depth=10)
    else:
        raise ValueError("Invalid model name.")

    model.fit(x_train, y_train)

    # save model in local
    path_save_model = config["model"]["path_save_model"]
    path_save_model_local = save_model(model, path_save_model)

    # save model in GCS
    path_save_model = config["model"]["path_save_model"]
    logger.info(f"Saving {model_name} model to {path_save_model} ...")
    upload_blob(bucket_name, path_save_model_local, path_save_model)
    return True


def save_model(model: object, file_path: str) -> None:
    """
    Save the model to the specified file path.

    Args:
        model: The model to be saved.
        file_path: The file path where the model will be saved.

    Returns:
        None
    """
    # get root path
    root_path = Path(__file__).parent.parent.parent
    # build full path save model
    full_path_save_model = root_path / file_path

    # check if the directory exists
    if not full_path_save_model.parent.exists():
        full_path_save_model.parent.mkdir(parents=True, exist_ok=True)

    pickle.dump(model, open(full_path_save_model, "wb"))

    logger.info(f"Model saved at {full_path_save_model}")

    return full_path_save_model


def load_model(file_path: str) -> object:
    """
    Load the model from the specified file path.

    Args:
        file_path: The file path where the model is saved.

    Returns:
        object: The loaded model.
    """
    # get root path
    root_path = Path(__file__).parent.parent.parent
    # build full path save model
    full_path_save_model = root_path / file_path

    model = pickle.load(open(full_path_save_model, "rb"))

    logger.info(f"Model loaded from {full_path_save_model}")

    return model


if __name__ == "__main__":

    # load config
    from lifecycle_ml_example.src.utils.config import load_config
    from lifecycle_ml_example.src.data.data_manager import read_data
    from lifecycle_ml_example.src.features.data_preprocessing import get_gold_data

    config = load_config()
    # path_data = config["data"]["path"]
    # model_algorithm = config["model"]["algorithm"]
    # path_save_model = config["model"]["path_save_model"]

    # # load data
    # df = read_data(path_data, local=True)

    # # get gold data
    # x_train, x_test, y_train, y_test = get_gold_data(df, config)

    # train model
    model = make_training(config)
