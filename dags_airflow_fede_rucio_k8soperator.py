"""
Create a pod that can access rucio endpoint at CNAF
- Using k8s operator
"""

from datetime import datetime

from airflow import DAG
from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator
from kubernetes.client import models as k8s

default_args = {
    "owner": "airflow",
    "start_date": datetime(2020, 1, 1),
}

dag = DAG(
    "rucio_operator", default_args=default_args, tags=["fede"], schedule_interval=None
)

repo = "leggerf/rucio-intertwin"
tag = "0.0.0"

k = KubernetesPodOperator(
    namespace="airflow",
    image=f"{repo}:{tag}",  # image="ubuntu:16.04",
    image_pull_secrets=[k8s.V1LocalObjectReference("dockerhub")],
    image_pull_policy="Always",
    cmds=["./get-token.sh"],
    # cmds=["bash", "-cx"],
    # arguments=["pwd", "ls"],
    labels={"foo": "bar"},
    name="test-data-access",
    task_id="data-access",
    is_delete_operator_pod=True,  # delete pod after execution
    hostnetwork=False,
    startup_timeout_seconds=900,
    dag=dag,
)

k1 = KubernetesPodOperator(
    namespace="airflow",
    image=f"{repo}:{tag}",  # image="ubuntu:16.04",
    image_pull_secrets=[k8s.V1LocalObjectReference("dockerhub")],
    image_pull_policy="Always",
    cmds=["./get-token.sh"],
    # cmds=["bash", "-cx"],
    # arguments=["pwd", "ls"],
    labels={"foo": "bar"},
    name="test-data-access-1",
    task_id="data-access-1",
    is_delete_operator_pod=True,  # delete pod after execution
    hostnetwork=False,
    startup_timeout_seconds=900,
    dag=dag,
)

# define DAG pipeline
(k >> k1)
