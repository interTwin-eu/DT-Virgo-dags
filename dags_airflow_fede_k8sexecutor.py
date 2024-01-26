"""
This is an example dag for using a Kubernetes Executor Configuration. 
It starts the following tasks:
1) pod with annotation
2) pod with mounted volume 
3) pod with sidecar and shared volume
4) pod with label
5) pod with other namespace
6) pod with image
7) pod with resource limits
"""
from __future__ import annotations

import logging
import os
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

if k8s:
    with DAG(
        dag_id="kubernetes_executor",
        schedule=None,
        start_date=pendulum.datetime(2021, 1, 1, tz="UTC"),
        catchup=False,
        tags=["fede"],
    ) as dag:
        
        #############################################################
        # Define config for task with pod annotation
        #############################################################
        executor_config_annotation = {
            "pod_override": k8s.V1Pod(
                metadata=k8s.V1ObjectMeta(annotations={"test": "fede"})
            )
        }

        #############################################################
        # 1) task with pod annotation
        #############################################################
        @task(executor_config=executor_config_annotation)
        def test_annotation():
            print_stuff()

        annotation_task = test_annotation()
        
        #############################################################
        # Define config for task with volume, mount host /tmp/ to /foo/
        #############################################################
        executor_config_volume_mount = {
            "pod_override": k8s.V1Pod(
                spec=k8s.V1PodSpec(
                    containers=[
                        k8s.V1Container(
                            name="base",
                            volume_mounts=[
                                k8s.V1VolumeMount(mount_path="/foo/", name="test-volume")
                            ],
                        )
                    ],
                    volumes=[
                        k8s.V1Volume(
                            name="test-volume",
                            host_path=k8s.V1HostPathVolumeSource(path="/tmp/"),
                        )
                    ],
                )
            ),
        }

        ###########################################################
        # 2) task with mount volume
        ###########################################################
        @task(executor_config=executor_config_volume_mount)
        def test_volume_mount():
            """
            Tests whether the volume has been mounted.
            """

            with open("/foo/volume_mount_test.txt", "w") as foo:
                foo.write("Hello")

            return_code = os.system("cat /foo/volume_mount_test.txt")
            if return_code != 0:
                raise ValueError(f"Error when checking volume mount. Return code {return_code}")

        volume_task = test_volume_mount()

        #############################################################
        # Define config for task with sidecar and shared volume /shared/
        #############################################################
        executor_config_sidecar = {
            "pod_override": k8s.V1Pod(
                spec=k8s.V1PodSpec(
                    containers=[
                        k8s.V1Container(
                            name="base",
                            volume_mounts=[
                                k8s.V1VolumeMount(
                                    mount_path="/shared/", name="shared-empty-dir")],
                        ),
                        k8s.V1Container(
                            name="sidecar",
                            image="ubuntu",
                            args=['echo "retrieved from mount" > /shared/test.txt'],
                            command=["bash", "-cx"],
                            volume_mounts=[
                                k8s.V1VolumeMount(
                                    mount_path="/shared/", name="shared-empty-dir"
                                )
                            ],
                        ),
                    ],
                    volumes=[
                        k8s.V1Volume(
                            name="shared-empty-dir", empty_dir=k8s.V1EmptyDirVolumeSource()
                        ),
                    ],
                )
            ),
        }

        ###########################################################
        # 3) pod with sidecar and shared volumes
        ###########################################################
        @task(executor_config=executor_config_sidecar)
        def test_sharedvolume_mount():
            """
            Tests whether the volume has been mounted.
            """
            for i in range(5):
                try:
                    return_code = os.system("cat /shared/test.txt")
                    if return_code != 0:
                        raise ValueError(f"Error when checking volume mount. Return code {return_code}")
                except ValueError as e:
                    if i > 4:
                        raise e

        sidecar_task = test_sharedvolume_mount()

        #############################################################
        # Define config for task: pod with label
        #############################################################
        executor_config_label = {
            "pod_override": k8s.V1Pod(metadata=k8s.V1ObjectMeta(labels={"release": "stable"}))
        }
        
        #############################################################
        # 4) pod with label
        #############################################################
        @task(executor_config=executor_config_label)
        def test_label():
            print_stuff()

        label_task = test_label()
        
        #############################################################
        # Define config for task: pod with namespace
        #############################################################
        executor_config_other_ns = {
            "pod_override": k8s.V1Pod(
                metadata=k8s.V1ObjectMeta(namespace="test-namespace", labels={"release": "stable"})
            )
        }

        #############################################################
        # 5) pod with other namespace
        #############################################################
        @task(executor_config=executor_config_other_ns)
        def other_namespace_task():
            print_stuff()

        other_ns_task = other_namespace_task()

        #############################################################
        # Define config for task: pod with image
        #############################################################
        
        worker_container_repository = conf.get("kubernetes_executor", "worker_container_repository")
        worker_container_tag = conf.get("kubernetes_executor", "worker_container_tag")

        # You can also change the base image, here we used the worker image for demonstration.
        # Note that the image must have the same configuration as the worker image. 
        # Could be that you want to run this task in a special docker image that has a zip
        # library built-in. You build the special docker image on top your worker image.
        kube_exec_config_image = {
            "pod_override": k8s.V1Pod(
                spec=k8s.V1PodSpec(
                    containers=[
                        k8s.V1Container(
                            name="base", image=f"{worker_container_repository}:{worker_container_tag}"
                        ),
                    ]
                )
            )
        }

        #############################################################
        # 6) pod with image
        #############################################################
        @task(executor_config=kube_exec_config_image)
        def image_override_task():
            print_stuff()

        image_task = image_override_task()

        #############################################################
        # Define config for task: pod with resource limits
        #############################################################
        
        # Use k8s_client.V1Affinity to define node affinity
        k8s_affinity = k8s.V1Affinity(
            pod_anti_affinity=k8s.V1PodAntiAffinity(
                required_during_scheduling_ignored_during_execution=[
                    k8s.V1PodAffinityTerm(
                        label_selector=k8s.V1LabelSelector(
                            match_expressions=[
                                k8s.V1LabelSelectorRequirement(key="app", operator="In", values=["airflow"])
                            ]
                        ),
                        topology_key="kubernetes.io/hostname",
                    )
                ]
            )
        )

        # Use k8s_client.V1Toleration to define node tolerations
        k8s_tolerations = [k8s.V1Toleration(key="dedicated", operator="Equal", value="airflow")]

        # Use k8s_client.V1ResourceRequirements to define resource limits
        k8s_resource_requirements = k8s.V1ResourceRequirements(
            requests={"memory": "512Mi"}, limits={"memory": "512Mi"}
        )

        kube_exec_config_resource_limits = {
            "pod_override": k8s.V1Pod(
                spec=k8s.V1PodSpec(
                    containers=[
                        k8s.V1Container(
                            name="base",
                            resources=k8s_resource_requirements,
                        )
                    ],
                    affinity=k8s_affinity,
                    tolerations=k8s_tolerations,
                )
            )
        }

        #############################################################
        # 7) pod with resource limits
        #############################################################
        @task(executor_config=kube_exec_config_resource_limits)
        def task_with_resource_limits():
            print_stuff()

        resource_task = task_with_resource_limits()

        #############################################################
        # Define DAG execution
        #############################################################
        (
            annotation_task
            >> [volume_task, other_ns_task, sidecar_task]
            >> label_task
            >> [image_task, resource_task]
        )
