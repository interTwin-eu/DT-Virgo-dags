"""
This is an example using a FileSensor



"""

import airflow.utils.dates
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor

dag = DAG(
    dag_id="fsensor_exmp",
    start_date=airflow.utils.dates.days_ago(3),
    schedule_interval=None,
    description="A demonstration DAG using FileSensor.",
    #The execution doesn't proceed if a task fails
    default_args={"depends_on_past": True},
)

create_metrics = DummyOperator(task_id="create_metrics", dag=dag)

for f_id in [1, 2, 3]:
    wait = FileSensor(
        task_id=f"wait_for_file_{f_id}",
        filepath=f"/data/f_{f_id}/data.csv",
        dag=dag,
    )
    copy = DummyOperator(task_id=f"copy_f_{f_id}", dag=dag)
    process = DummyOperator(task_id=f"process_f_{f_id}", dag=dag)
    wait >> copy >> process >> create_metrics
