"""
Create a pod that can access rucio endpoint at CNAF
- Using k8s executor
"""

from __future__ import annotations

import logging

import pendulum

# from airflow.configuration import conf
from airflow.decorators import task
from airflow.models.dag import DAG

# import os
# from pathlib import Path

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

default_queue = "kubernetes"
# default_queue = "default"

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

        # use image from worker node as example
        # repo = conf.get("kubernetes_executor", "worker_container_repository")
        # tag = conf.get("kubernetes_executor", "worker_container_tag")

        kube_exec_config_rucio = {
            "pod_override": k8s.V1Pod(
                spec=k8s.V1PodSpec(
                    containers=[
                        k8s.V1Container(
                            name="base",  # the image must be named base
                            # image=f"{repo}:{tag}",  # the image must contain airflow
                        ),
                        k8s.V1Container(
                            name="sidecar",
                            image=f"{repo}:{tag}",
                            args=['ls'],
                            command=["bash", "-cx"],
                            # command=["./get-token.sh"],
                            image_pull_policy="Always",
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
            task_id="data_access",
        )
        def data_access_task():
            log.info("Using image " + f"{repo}:{tag}")

        rucio_task = data_access_task()

        #############################################################
        # Define DAG execution
        #############################################################
        (rucio_task)
