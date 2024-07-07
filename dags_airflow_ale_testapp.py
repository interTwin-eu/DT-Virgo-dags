import airflow.utils.dates
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.http.sensors.http import HttpSensor

dag = DAG(
    dag_id="testapp",
    start_date=airflow.utils.dates.days_ago(3),
    schedule_interval=None,
    description="A demonstration DAG using HttpSensor.",
    
)

n_cond=10

def check_response(response):
   js = response.json()
   output=js

   flag=None
    

   if(js):
    LoggingMixin().log.info("Read json object")
    

    if(output['buff_size']>=n_cond): 
      flag=True
    else:
      flag=False   
   
    
   return flag



    
checkAppState = HttpSensor(task_id="checkState", 
  http_conn_id="testapp", 
  endpoint="bufstat", 
  response_check=lambda response: check_response(response), 
  poke_interval=10, 
  timeout=3600,
  
)


create_metrics = DummyOperator(task_id="create_metrics", dag=dag) 

checkAppState>>create_metrics