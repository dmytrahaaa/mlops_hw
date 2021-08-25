FROM python:3.7-slim-buster
COPY requirements.txt /app/requirements.txt
RUN apt-get update && apt-get install -y build-essential
RUN pip install --upgrade pip setuptools wheel && pip install --default-timeout=100 -r app/requirements.txt
COPY . /app/
WORKDIR /app/
ENV PYTHONUNBUFFERED 1
ENTRYPOINT ["python"]
CMD ["main.py"]
EXPOSE 5500:5500