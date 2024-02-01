from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator
from airflow import DAG
from datetime import datetime
from kubernetes.client import models as k8s

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2020, 1, 1),
    'tags': ['fede']    
}

dag = DAG('rucio_operator', default_args=default_args, schedule_interval=None)

repo = "leggerf/rucio-intertwin"
tag = "0.0.0"

k = KubernetesPodOperator(
    namespace='airflow',
    image=f"{repo}:{tag}",  # image="ubuntu:16.04",
    image_pull_secrets=[k8s.V1LocalObjectReference("dockerhub")],
    image_pull_policy="Always",
    cmds=["bash", "-cx"],
    arguments=["pwd", "ls"],
    labels={"foo": "bar"},
    name="test-data-access",
    task_id="data-access",
    is_delete_operator_pod=False,
    hostnetwork=False,
    dag=dag
)

# define DAG pipeline
(k)
