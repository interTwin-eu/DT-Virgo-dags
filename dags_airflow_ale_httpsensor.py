"""
Example code for HttpSensor


"""


import airflow.utils.dates
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.http.sensors.http import HttpSensor


dag = DAG(
    dag_id="httpsens_exmp",
    start_date=airflow.utils.dates.days_ago(3),
    schedule_interval=None,
    description="A demonstration DAG using HttpSensor.",
    
)

def check_response_itm(response):
    
    LoggingMixin().log.info("Callable function start")

    #the object type is List
    js = response.json()
    flag=None
    

    if(js):
     LoggingMixin().log.info("Read json object")
     n_itm=len(js) 
     LoggingMixin().log.info("Json has %s items",n_itm)

     if(n_itm)>=10: 
       flag=True
     else:
      flag=False   
    else:
     flag=False
    
    return flag
    

    
    
checkNItems = HttpSensor(task_id="check", 
  http_conn_id="fakeAPIPlaceh", 
  endpoint="users", 
  response_check=lambda response: check_response_itm(response), 
  poke_interval=10, 
  timeout=100,
  
)

create_metrics = DummyOperator(task_id="create_metrics", dag=dag)

checkNItems>>create_metrics



