FROM python:3.7-slim-buster
COPY requirements.txt /train/requirements.txt
RUN apt-get update && apt-get install -y build-essential
RUN pip install --upgrade pip setuptools wheel && pip install --default-timeout=100 -r train/requirements.txt
COPY . /train/
WORKDIR /train/
ENV PYTHONUNBUFFERED 1
ENTRYPOINT ["python"]
CMD ["main.py"]
EXPOSE 5550:5550