""" this script is used to ingest and retrieve data from the database """
if True:
    import sys
    sys.path.append("../../")

import pandas as pd
from pathlib import Path
from lifecycle_ml_example.src.utils.logger_class import Logger

# instance logger
logger = Logger(__name__).logger


def read_data(path_data: str, local: bool = False) -> pd.DataFrame:
    """
    Read data from either a local or remote store.

    Args:
        local (bool): Flag indicating whether to read data from local store.
        path_data (str): The path to the data.

    Returns:
        pd.DataFrame: The data read from the store.
    """

    if local:
        logger.info("Reading data from local store")
        root_path = Path(__file__).parent.parent.parent
        # full path to data
        full_path_data = root_path / path_data
        df = pd.read_csv(full_path_data)
    else:
        logger.info("Reading data from remote store")
        # TODO: Implement reading from remote store

    return df


if __name__ == "__main__":

    # load config
    from lifecycle_ml_example.src.utils.config import load_config

    config = load_config()
    path_data = config["data"]["path"]

    # load data
    df = read_data(path_data, local=True)
