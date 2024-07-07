import airflow.utils.dates
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.http.operators.http import SimpleHttpOperator

import json
from datetime import timedelta

#baseapiurl="http://gflapi.glitchflow.svc.cluster.local:8000/"
#apifrzendp="train"

dag = DAG(
    dag_id="Trainpipe",
    start_date=airflow.utils.dates.days_ago(3),
    schedule_interval=None,
    description="DAG implementing the AI training data pipeline.",
    
)

IniTrain = DummyOperator(task_id="start_training", dag=dag)

sign_train = SimpleHttpOperator(
    task_id="send_frz_sign",
    method="POST",
    http_conn_id="testapp",
    endpoint="train",
    data=json.dumps({"user":"airflow","token":"airflow"}),
    headers={"Content-Type": "application/json"},
    xcom_push=True,
    retries=3,
    retry_delay=timedelta(minutes=5),
    dag=dag,
)

Next = DummyOperator(task_id="next", dag=dag)

IniTrain>>sign_train>>Next