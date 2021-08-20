FROM python:3.7-slim-buster
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y build-essential
RUN pip install --upgrade pip setuptools wheel && pip install -r app/requirements.txt
COPY . /app/
WORKDIR /app/
ENV PYTHONUNBUFFERED 1
