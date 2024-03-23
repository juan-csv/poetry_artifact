import pandas as pd
from airflow.models.dag import DAG
from airflow.decorators import dag, task
from datetime import datetime, timedelta
import numpy as np

# Operators; we need this to operate!
from airflow.operators.python import PythonOperator
# import dummy_operator
from airflow.operators.dummy import DummyOperator
from lifecycle_ml_example.src.utils.logger_class import Logger
from lifecycle_ml_example.src.features.data_preprocessing import get_gold_data
from lifecycle_ml_example.src.models.training import make_training, load_model
from lifecycle_ml_example.src.metrics.evaluate_model import evaluate_model
from lifecycle_ml_example.src.utils.f_gcs_storage import download_blob, get_blob_as_dataframe, upload_df_to_gcs
from lifecycle_ml_example.src.utils.config import load_config
from airflow.models import DagBag


def callback_subdag_clear(context):
    """Clears a subdag's tasks on retry."""
    dag_id = "{}.{}".format(
        context['dag'].dag_id,
        context['ti'].task_id
    )
    execution_date = context['execution_date']
    sdag = DagBag().get_dag(dag_id)
    sdag.clear(
        start_date=execution_date,
        end_date=execution_date,
        only_failed=False,
        only_running=False,
        confirm_prompt=False,
        include_subdags=False)


# instance logger
logger = Logger(__name__).logger
config = load_config()


default_args = {
    "depends_on_past": True,
    "retries": 3,
    "retry_delay": timedelta(seconds=10),
    "retry_exponential_backoff": True
}


# args for the DAG
DAG_ID = "dag_ml_life_cycle"
SCHEDULE_INTERVAL = timedelta(minutes=5)
# current time that was created the DAG
START_DATE = datetime.now() - 2 * SCHEDULE_INTERVAL


# Define the DAG
my_dag = DAG(
    dag_id="dag_ml_life_cycle_v2",
    default_args=default_args,
    description="ML lfecycle example DAG v2",
    schedule_interval=SCHEDULE_INTERVAL,
    start_date=START_DATE,
    # catchup=False,  # do not backfill
    tags=["Juan", "ML", "LifeCycle", "Example", "v2"],
    on_failure_callback=callback_subdag_clear,
)
# start task with dummy operator
start = DummyOperator(task_id="start_v2", dag=my_dag)

# load data
prob_random_error = np.random.rand()
task_preprocessing_data = PythonOperator(
    task_id="get_gold_data_v2",
    python_callable=get_gold_data,
    op_args=[config],
    dag=my_dag,
)

# define tasks and to each task associate with the dag my_dag

# training
task_training = PythonOperator(
    task_id="make_training_v2",
    python_callable=make_training,
    op_args=[config],
    dag=my_dag,
)

# evaluation


@task(task_id="evaluate_model_v2", dag=my_dag)
def evaluate_model_task(config: dict):
    bucket_name = config["storage"]["bucket"]
    path_gold_data = config["storage"]["path_gold_data"]
    path_save_model = config["model"]["path_save_model"]

    logger.info(
        f"Getting gold data from {bucket_name}/{path_gold_data} ...")
    x_train = get_blob_as_dataframe(
        bucket_name, path_gold_data + "/x_train.csv")
    y_train = get_blob_as_dataframe(
        bucket_name, path_gold_data + "/y_train.csv")
    x_test = get_blob_as_dataframe(
        bucket_name, path_gold_data + "/x_test.csv")
    y_test = get_blob_as_dataframe(
        bucket_name, path_gold_data + "/y_test.csv")

    # download model
    logger.info(
        f"Downloading model from {bucket_name}/{path_save_model} ...")
    download_blob(bucket_name, path_save_model, "/tmp/model.pkl")

    # load model
    model = load_model("/tmp/model.pkl")

    logger.info(f"Evaluating model ...")
    metrics: dict = evaluate_model(model, x_test, y_test, x_train, y_train)

    # transform metrics to dataframe
    metrics_df = pd.DataFrame(metrics, index=[0])

    # save metrics
    logger.info(f"Uploading metrics to {bucket_name}/{path_gold_data} ...")
    upload_df_to_gcs(
        bucket_name, "metrics.csv", metrics_df)


# ends task with dummy operator
end = DummyOperator(task_id="end_v2", dag=my_dag)

# dag doc
dag.doc_md = __doc__
dag.doc_md = """\
# ML LifeCycle Example
This is a simple example of a ML LifeCycle using Airflow.
"""

# Define the DAG
start >> task_preprocessing_data >> task_training >> evaluate_model_task(
    config) >> end
