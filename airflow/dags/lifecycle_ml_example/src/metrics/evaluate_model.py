
""" this file contains the data preprocessing functions """
if True:
    import sys
    sys.path.append("../../")
from lifecycle_ml_example.src.utils.logger_class import Logger
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score


# instance logger
logger = Logger(__name__).logger


def evaluate_model(model, x_test_hashing, y_test, x_train_hashing, y_sm_train):
    y_pred = model.predict(x_test_hashing)
    y_pred_proba = model.predict_proba(x_test_hashing)
    y_pred_train = model.predict(x_train_hashing)
    y_pred_train_proba = model.predict_proba(x_train_hashing)

    metrics = {}
    metrics['accuracy_train'] = accuracy_score(y_sm_train, y_pred_train)
    metrics['precision_train'] = precision_score(y_sm_train, y_pred_train)
    metrics['recall_train'] = recall_score(y_sm_train, y_pred_train)
    metrics['roc_auc_train'] = roc_auc_score(
        y_sm_train, y_pred_train_proba[:, 1])
    metrics['accuracy_test'] = accuracy_score(y_test, y_pred)
    metrics['precision_test'] = precision_score(y_test, y_pred)
    metrics['recall_test'] = recall_score(y_test, y_pred)
    metrics['roc_auc_test'] = roc_auc_score(y_test, y_pred_proba[:, 1])

    print('accuracy (test): ' + str(metrics['accuracy_test']))
    print('precision (test): ' + str(metrics['precision_test']))
    print('recall (test): ' + str(metrics['recall_test']))
    print('roc-auc (train-proba): ' +
          str(metrics['roc_auc_train']))
    print('roc-auc (test-proba): ' +
          str(metrics['roc_auc_test']))

    return metrics


if __name__ == "__main__":

    # load config
    from lifecycle_ml_example.src.utils.config import load_config
    from lifecycle_ml_example.src.models.training import load_model
    from lifecycle_ml_example.src.data.data_manager import read_data
    from lifecycle_ml_example.src.features.data_preprocessing import get_gold_data

    config = load_config()
    path_data = config["data"]["path"]
    path_save_model = config["model"]["path_save_model"]

    # load data
    df = read_data(path_data, local=True)

    # get gold data
    x_train, x_test, y_train, y_test = get_gold_data(df, config)

    # load model
    model = load_model(path_save_model)

    # get metrics
    metrics = evaluate_model(model, x_test, y_test, x_train, y_train)
