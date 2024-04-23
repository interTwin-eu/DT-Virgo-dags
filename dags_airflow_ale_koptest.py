import airflow.utils.dates
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator

dag = DAG(
    dag_id="kop_test",
    start_date=airflow.utils.dates.days_ago(3),
    schedule_interval=None,
    description="A demonstration DAG using KubernetesPodOp.",
    
)

repo="romanoa77/dockdepo"
tag="guniflask.omega"

create_pod=KubernetesPodOperator(
 namespace="airflow",
 image=f"{repo}:{tag}",
 image_pull_policy="Always",
 ##cmds
 name="test_pod",
 task_id="run_pod",
 is_delete_operator_pod=True,
 startup_timeout_seconds=900,
 dag=dag,


)



#create_metrics = DummyOperator(task_id="create_metrics", dag=dag)

create_pod
#>>create_metrics

