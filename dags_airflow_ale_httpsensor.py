import airflow.utils.dates
from airflow import DAG
from airflow.operator.dummy import DummyOperator
from airflow.providers.http.sensors.http import HttpSensor

dag = DAG(
    dag_id="httpsens_exmp",
    start_date=airflow.utils.dates.days_ago(3),
    schedule_interval=None,
    description="A demonstration DAG using HttpSensor.",
    
)

def check_response_itm(response):
    js = response.json()
    print("The json var is a ",type(js))

    return True
    

    
    
checkNItems = HttpSensor(task_id="check", 
  http_conn_id="fakeAPIPlaceh", 
  endpoint="users", 
  response_check=lambda response: check_response_itm(response), 
  poke_interval=10, 
  timeout=100,
  
)

create_metrics = DummyOperator(task_id="create_metrics", dag=dag)

checkNItems>>create_metrics


