import airflow
from datetime import date, datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
import os
import shutil

default_args = {
    "depends_on_past": False,
    "start_date": datetime(2021, 8, 20, 15, 50, 0),
    "retries": 1,
    "retry_delay": timedelta(hours=1),
}


def _move_new_image(src_folder, dst_folder):
    file_names = os.listdir(src_folder)
    print(file_names)

    for file_name in file_names:
        path = os.path.join(src_folder, file_name)
        print(src_folder)
        print("file_name", file_name)
        print("path", path)
        print("dst_folder", dst_folder)
        shutil.copyfile(path, "/bitnami/ml_image")
        os.remove(path)


with DAG(dag_id="image_move_pipeline", schedule_interval="@once", default_args=default_args) as dag:
    run_train = BashOperator(task_id='run_train', bash_command="curl -XGET 'model-train:5000/train'")
    sensor_task = FileSensor(task_id="file_sensor", poke_interval=30, filepath="/bitnami/new_images/")
    move_image = PythonOperator(task_id='move_new_image',
                                python_callable=_move_new_image,
                                op_args=['/bitnami/new_images/', '/bitnami/ml_images/'],
                                )
    run_predict = BashOperator(task_id='predict_new_image',
                               bash_command="curl -XGET 'ml-api:5000/predict/bitnami/ml_images'")

run_train >> sensor_task >> move_image >> run_predict
