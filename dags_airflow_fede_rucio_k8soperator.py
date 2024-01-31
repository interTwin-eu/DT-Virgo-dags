from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator
from airflow import DAG
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2020, 1, 1),
    'tags': ['fede']    
}

dag = DAG('rucio_operator', default_args=default_args, schedule_interval=None)

k = KubernetesPodOperator(
    namespace='airflow',
    image="ubuntu:16.04",
    cmds=["bash", "-cx"],
    arguments=["pwd", "ls"],
    labels={"foo": "bar"},
    name="test-data-access",
    task_id="data-access",
    is_delete_operator_pod=False,
    hostnetwork=False,
    dag=dag
)

(k)
