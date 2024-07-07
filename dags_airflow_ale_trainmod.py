import airflow.utils.dates
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.operators.python import BranchPythonOperator

import json
from datetime import timedelta

#baseapiurl="http://gflapi.glitchflow.svc.cluster.local:8000/"
#apifrzendp="train"
#apistat="stats"

WAITMSG="WAITING"
FRZMSG="FROZEN"
MAXBUFSZ=2000

dag = DAG(
    dag_id="Trainpipe",
    start_date=airflow.utils.dates.days_ago(3),
    schedule_interval=None,
    description="DAG implementing the AI training data pipeline.",
    
)


##DAG CALLABLES
def pick_branch(**context):

    jsmsg=context["task_instance"].xcom_pull(
        task_ids="send_frz_sign", key="return_value"
    )

    respob=json.loads(jsmsg)

    if(respob["resp"]==WAITMSG):
        return "next_sensor"
    else:
        return "next_metrics"  



def check_response(response):
   js = response.json()
   output=js

   flag=None
    

   if(js):
    LoggingMixin().log.info("Read json object")
    

    if(output['buff_size']>=MAXBUFSZ): 
      flag=True
    else:
      flag=False   
   
    
   return flag       

    
### DAG DEFS   
IniTrain = DummyOperator(task_id="start_training", dag=dag)

sign_train = SimpleHttpOperator(
    task_id="send_frz_sign",
    method="POST",
    http_conn_id="testapp",
    endpoint="train",
    data=json.dumps({"user":"airflow","token":"airflow"}),
    headers={"Content-Type": "application/json"},
    
    
    retries=3,
    retry_delay=timedelta(minutes=5),
    dag=dag,
)



chech_train_resp=BranchPythonOperator(
    task_id="check_frz_sign",
    python_callable=pick_branch,
    dag=dag
)


###SENSOR BRANCH
next_sens = HttpSensor(task_id="next_sensor", 
  http_conn_id="testapp", 
  endpoint="stats", 
  response_check=lambda response: check_response(response), 
  poke_interval=10, 
  timeout=3600,
  mode="reschedule",
  dag=dag
)

freeze= SimpleHttpOperator(
    task_id="freeze_ds",
    method="POST",
    http_conn_id="testapp",
    endpoint="train",
    data=json.dumps({"user":"airflow","token":"airflow"}),
    headers={"Content-Type": "application/json"},
    
    
    retries=3,
    retry_delay=timedelta(minutes=5),
    dag=dag,
)
#######################################


###BEST CASE BRANCH
next_metrics = DummyOperator(task_id="next_metrics", dag=dag)
##########################################################



join_branch = DummyOperator(task_id="join_brc",trigger_rule="none_failed", dag=dag)
list_proc = DummyOperator(task_id="listen_preproc", dag=dag)



IniTrain>>sign_train>>chech_train_resp>>[next_sens,next_metrics]
next_sens>>freeze
[next_metrics,freeze]>>join_branch
join_branch>>list_proc