import airflow
from datetime import date, datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.contrib.sensors.file_sensor import FileSensor
import os
import shutil
import requests


default_args = {
    "depends_on_past": False,
    "start_date": airflow.utils.dates.days_ago(1),
    "retries": 1,
    "retry_delay": timedelta(hours=1),
}


def _move_new_image(src_folder, dst_folder):
    file_names = os.listdir(src_folder)

    for file_name in file_names:
        shutil.move(os.path.join(src_folder, file_name), dst_folder)


with DAG(dag_id="image_move_pipeline", schedule_interval="@once", default_args=default_args) as dag:
    # start_task = DummyOperator(task_id="start")
    # stop_task = DummyOperator(task_id="stop")
    sensor_task = FileSensor(task_id="file_sensor_task", poke_interval=30, filepath="/bitnami/new_images/")
    move_image = PythonOperator(task_id='move_new_image',
                                python_callable=_move_new_image,
                                op_args=['/bitnami/new_images/', '/bitnami/ml_images/']
                                )
    send_curl = BashOperator(task_id='send_curl', bash_command="curl -XGET 'ml-api:5500/predict'")

# start_task >> sensor_task >> move_image >> send_curl >> stop_task
sensor_task >> move_image >> send_curl
