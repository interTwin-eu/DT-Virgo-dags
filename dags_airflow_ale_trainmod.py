import airflow.utils.dates
from airflow.utils.log.logging_mixin import LoggingMixin
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.providers.http.sensors.http import HttpSensor
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.operators.python import BranchPythonOperator
from airflow.contrib.operators.kubernetes_pod_operator import KubernetesPodOperator
from kubernetes.client import models as k8s

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


preproc =KubernetesPodOperator(
        # unique id of the task within the DAG
        task_id="preQ",
        # the Docker image to launch
        image="romanoa77/preq:0.3.air",
        # launch the Pod on the same cluster as Airflow is running on
        in_cluster=True,
        # launch the Pod in the same namespace as Airflow is running in
        namespace="glitchflow",
        # Pod configuration
        # name the Pod
        name="airflow_preprocessor",
        
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
        volumes = [k8s.V1Volume(
        name="gwdatal",
        persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name="gwdatal"),
        )],
        volume_mounts=[k8s.V1VolumeMount(mount_path="/app/data", name="gwdatal", sub_path=None, read_only=False)
        ],
        dag=dag,
        
        #env_vars={"NAME_TO_GREET": f"{name}"},
    )

 



IniTrain>>sign_train>>chech_train_resp>>[next_sens,next_metrics]
next_sens>>freeze
[next_metrics,freeze]>>join_branch
join_branch>>preproc
