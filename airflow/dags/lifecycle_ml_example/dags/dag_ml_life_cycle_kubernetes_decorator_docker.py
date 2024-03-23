import pandas as pd
from airflow.models.dag import DAG
from airflow.decorators import dag, task
from datetime import datetime, timedelta
import numpy as np

# Operators; we need this to operate!
# from airflow.operators.python import PythonOperator
from airflow.operators.docker_operator import DockerOperator
from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator
from lifecycle_ml_example.src.models.training import make_training, load_model


# import dummy_operator
from airflow.operators.dummy import DummyOperator
from lifecycle_ml_example.src.utils.logger_class import Logger
from lifecycle_ml_example.src.features.data_preprocessing import get_gold_data
from lifecycle_ml_example.src.models.training import make_training, load_model
from lifecycle_ml_example.src.metrics.evaluate_model import evaluate_model
from lifecycle_ml_example.src.utils.f_gcs_storage import download_blob, get_blob_as_dataframe, upload_df_to_gcs
from lifecycle_ml_example.src.utils.config import load_config

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
DAG_ID = "dag_ml_life_cycle_docker_decorator"
SCHEDULE_INTERVAL = timedelta(minutes=5)
# current time that was created the DAG
START_DATE = datetime.now() - 2 * SCHEDULE_INTERVAL

# define DAG
with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="ML lfecycle example DAG using custom docker",
    schedule_interval=SCHEDULE_INTERVAL,
    start_date=START_DATE,
    tags=["Juan", "ML", "LifeCycle", "Example", "Docker"]
) as dag:

    # define tasks and use docker operator

    @task.docker(image='gcr.io/testingv0001/worker_airflow', task_id="make_training_in_docker")
    def make_training_in_docker(config):
        logger.info("Training model with docker...")
        result = make_training(config)

    # start task with dummy operator
    start = DummyOperator(task_id="star_docker_decorator")

    # task_processing = KubernetesPodOperator(
    #     name="get_gold_data_KubernetesPodOperator",
    #     task_id="get_gold_data_KubernetesPodOperator",
    #     image='gcr.io/testingv0001/worker_airflow',
    #     cmds=["python", "airflow/dags/lifecycle_ml_example/src/features/data_preprocessing.py"],
    #     dag=dag,
    # )

    # ends task with dummy operator
    end = DummyOperator(task_id="end_docker_decorator")

    # define the order of the tasks
    start >> make_training_in_docker(config) >> end
