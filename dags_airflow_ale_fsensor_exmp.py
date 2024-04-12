"""
This is an example using a FileSensor. 
In order to work this DAG needs a connection defined as working directory.
See the airflow UI.

FileSensor expects a file in the local filesystem.

A directory d_1 was put in the temp directory with the data.csv file


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
    
    #DAG execution depends on previous run
    #default_args={"depends_on_past": True},
)



create_metrics = DummyOperator(task_id="create_metrics", dag=dag)

for f_id in [1, 2, 3]:
    wait = FileSensor(
        task_id=f"wait_for_file_{f_id}",
        #define a connection
        fs_conn_id="temp_data",
        filepath=f"d_{f_id}/data.csv",
        #condition will be checked every ten seconds
        poke_interval=10,
        #maximum waiting time
        timeout=180,
        #we can reschedule the task every time the condition is not met
        #mode="reschedule"

        dag=dag,
    )
    copy = DummyOperator(task_id=f"copy_f_{f_id}", dag=dag)
    process = DummyOperator(task_id=f"process_f_{f_id}", dag=dag)
    wait >> copy >> process >> create_metrics
