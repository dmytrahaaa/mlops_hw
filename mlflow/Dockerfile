FROM python:3.9.5
WORKDIR /mlflow

COPY requirements.txt /mlflow/requirements.txt
RUN python -m pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
ENV MLFLOW_TRACKING_URI=http://0.0.0.0:5000
RUN mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow-artifact-root --host 0.0.0.0 --port 5000
RUN mlflow models serve -m $1 -p 1234 -h 0.0.0.0 --no-cond
