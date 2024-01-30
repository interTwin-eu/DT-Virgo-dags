"""
Create a pod that can access rucio endpoint at CNAF
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pendulum
from airflow.configuration import conf
from airflow.decorators import task
from airflow.example_dags.libs.helper import print_stuff
from airflow.models.dag import DAG

log = logging.getLogger(__name__)

# Check k8s is there
try:
    from kubernetes.client import models as k8s
except ImportError:
    log.warning(
        "This DAG requires the kubernetes provider."
        " Please install it with: pip install apache-airflow[cncf.kubernetes]"
    )
    k8s = None

# default_queue = "kubernetes"
default_queue = "default"

if k8s:
    with DAG(
        dag_id="rucio_executor",
        schedule=None,
        start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
        catchup=False,
        tags=["fede"],
    ) as dag:

        #############################################################
        # Define config for pod with data access at CNAF and rucio
        #############################################################

        repo = "leggerf/rucio-intertwin"
        tag = "0.0.0"

        kube_exec_config_rucio = {
            "pod_override": k8s.V1Pod(
                spec=k8s.V1PodSpec(
                    containers=[
                        k8s.V1Container(
                            name="base",
                            image=f"{repo}:{tag}",
                        ),
                    ],
                    image_pull_secrets=[
                        k8s.V1LocalObjectReference(
                            name="dockerhub",
                        ),
                    ],
                )
            )
        }

        #############################################################
        # pod with access to rucio
        #############################################################
        @task(
            executor_config=kube_exec_config_rucio,
            queue=default_queue,
            task_id="task_rucio",
        )
        def rucio_task():
            print_stuff()
            log.info("Using image " + f"{repo}:{tag}")

        rucio_task = rucio_task()

        #############################################################
        # Define DAG execution
        #############################################################
        (rucio_task)