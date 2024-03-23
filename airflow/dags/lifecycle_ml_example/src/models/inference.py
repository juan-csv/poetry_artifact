
""" this file contains the data preprocessing functions """
if True:
    import sys
    sys.path.append("../../")
import traceback
import pandas as pd
from lifecycle_ml_example.src.utils.logger_class import Logger
from lifecycle_ml_example.src.models.training import load_model
from lifecycle_ml_example.src.features.data_preprocessing import preprocess_data, encode_features


# instance logger
logger = Logger(__name__).logger


class InferenceClassificationModel:
    def __init__(self, config):
        self.config = config
        self.path_model_saved = config["model"]["path_save_model"]

        self.model = load_model(self.path_model_saved)

    def predict(self, input_data: pd.DataFrame | dict) -> list:
        """
        Function to predict the target variable from the input dataframe.

        Args:
            input_data (pd.DataFrame or dict): The input dataframe.

        Returns:
            list: A list containing the predicted target variable.
        """
        logger.info("Predicting target variable")

        if isinstance(input_data, dict):
            input_data = pd.DataFrame(input_data, index=[0])

        # preprocess data
        x_input = preprocess_data(input_data)
        x_hashed = encode_features(x_input)

        # get predictions
        try:
            output = list(self.model.predict(x_hashed))
        except Exception as e:
            logger.error(f"Error predicting target variable: {e}")
            logger.error(traceback.format_exc())
            output = []

        return output


if __name__ == "__main__":

    # load config
    from lifecycle_ml_example.src.utils.config import load_config

    config = load_config()

    # load model
    predictor = InferenceClassificationModel(config)

    input = {'destination': 'No Urgent Place',
             'passanger': 'Alone',
             'weather': 'Sunny',
             'temperature': 55,
             'time': '2PM',
             'coupon': 'Restaurant(<20)',
             'expiration': '1d',
             'gender': 'Female',
             'age': '21',
             'maritalStatus': 'Unmarried partner',
             'has_children': 1,
             'education': 'Some college - no degree',
             'occupation': 'Unemployed',
             'income': '$37500 - $49999',
             'car': None,
             'Bar': 'never',
             'CoffeeHouse': 'never',
             'CarryAway': None,
             'RestaurantLessThan20': '4~8',
             'Restaurant20To50': '1~3',
             'toCoupon_GEQ5min': 1,
             'toCoupon_GEQ15min': 0,
             'toCoupon_GEQ25min': 0,
             'direction_same': 0,
             'direction_opp': 1}

    # predict
    output = predictor.predict(input)
