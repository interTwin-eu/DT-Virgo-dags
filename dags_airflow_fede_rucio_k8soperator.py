from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator
from airflow import DAG
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2020, 1, 1),
}

dag = DAG('kubernetes_sample', default_args=default_args, schedule_interval=None)

k = KubernetesPodOperator(
    namespace='airflow',
    image="ubuntu:16.04",
    cmds=["bash", "-cx"],
    arguments=["echo", "10"],
    labels={"foo": "bar"},
    name="airflow-test-pod",
    task_id="task",
    is_delete_operator_pod=True,
    hostnetwork=False,
    dag=dag
)
