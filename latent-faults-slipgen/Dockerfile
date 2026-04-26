FROM apache/airflow:2.7.1-python3.10
USER root
RUN apt-get update && apt-get install -y git libgl1-mesa-glx libglib2.0-0
USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt mlflow dvc
COPY . /app
WORKDIR /app