import airflow.utils.dates
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow import DAG
from airflow.operators.dummy import DummyOperator

from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from kubernetes.client import models as k8s

import json
from datetime import timedelta





dag = DAG(
    dag_id="kubeoptest_focal",
    start_date=airflow.utils.dates.days_ago(3),
    schedule_interval=None,
    description="Test for kubernetes operator",
    
)


##DAG CALLABLES


    
### DAG DEFS   
Ini = DummyOperator(task_id="start_training", dag=dag)



Op =KubernetesPodOperator(
        # unique id of the task within the DAG
        task_id="kubeop",
        # the Docker image to launch
        image="debian",
        image_pull_policy="Always",
        cmds=["bash","-cx"],
        arguments=["echo","Hello World"],
        # launch the Pod on the same cluster as Airflow is running on
        in_cluster=True,
        # launch the Pod in the same namespace as Airflow is running in
        namespace="default",
        # Pod configuration
        # name the Pod
        name="airflow_op",
        
        
        # attach labels to the Pod, can be used for grouping
        labels={"app": "preq", "backend": "airflow"},
        # reattach to worker instead of creating a new Pod on worker failure
        reattach_on_restart=True,
        # delete Pod after the task is finished
        is_delete_operator_pod=True,
        # get log stdout of the container as task logs
        get_logs=True,
        # log events in case of Pod failure
        log_events_on_failure=True,
        # enable xcom
        do_xcom_push=True,
        dag=dag,
        
        #env_vars={"NAME_TO_GREET": f"{name}"},
    )

 



Ini>>Op
