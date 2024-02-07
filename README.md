# DAGS for Virgo DT

- This repository contains DAGS for the Virgo DT.
- If you edit a DAG or add one (just be sure to use airflow in the filename),
it is added automatically to the Airflow setup on the k8s cluster at CNAF.
- to trigger DAG execution you can use the [dashboard](http://localhost:8080/) (admin/admin)
- for the dashboard, you need to have access to the cluster, and execute on your local machine:

```bash
kubectl port-forward svc/airflow-webserver 8080:8080 --namespace airflow
```
