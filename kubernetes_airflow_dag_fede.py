#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
This is an example dag for using a Local Kubernetes Executor Configuration.
"""
from __future__ import annotations

import logging
from datetime import datetime

from airflow.configuration import conf
from airflow.decorators import task
from airflow.example_dags.libs.helper import print_stuff
from airflow.models.dag import DAG

log = logging.getLogger(__name__)

worker_container_repository = conf.get(
    "kubernetes_executor", "worker_container_repository"
)
worker_container_tag = conf.get("kubernetes_executor", "worker_container_tag")

try:
    from kubernetes.client import models as k8s
except ImportError:
    log.warning(
        "Could not import DAGs in fede_local_kubernetes_executor.py", exc_info=True
    )
    log.warning(
        "Install Kubernetes dependencies with: \
         pip install apache-airflow[cncf.kubernetes]"
    )
    k8s = None


# Customize DAG name

if k8s:
    with DAG(
        dag_id="fede_local_kubernetes_executor",
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=["fede"],
    ) as dag:
        # You can use annotations on your kubernetes pods!
        start_task_executor_config = {
            "pod_override": k8s.V1Pod(
                metadata=k8s.V1ObjectMeta(annotations={"test": "fede"})
            )
        }

        ###########################
        #        DEFINE TASKS
        ###########################
 
        # Task with k8s executor
        
        @task(
            executor_config=start_task_executor_config,
            queue="kubernetes",
            task_id="task_with_kubernetes_executor",
        )
        def task_with_executor():
            print_stuff()
            return "Done with task_with_kubernetes_executor"

        # Task with local executor
        
        @task(task_id="task_with_local_executor")
        def task_with_local(ds=None, **kwargs):
            """Print the Airflow context and ds variable from the context."""
            print(kwargs)
            print(ds)
            return "Done with task_with_local_executor"

        # Task with docker image

        @task.docker(image="python:3.9-slim-bookworm", multiple_outputs=True)
        def task_with_docker(order_data_dict: dict):
            """#### Transform task
            A simple Transform task which takes in the collection of order data and computes the total order value.
            """
            total_order_value = 0

            for value in order_data_dict.values():
                total_order_value += value

            return {"total_order_value": total_order_value}

        ###########################
        #        DEFINE TASK ORDER
        ###########################
        
        thisdict =	{
            "brand": 1,
            "model": 2,
            "year": 3
        }
        
        task_with_docker(thisdict) >> task_with_local() >> task_with_executor()
