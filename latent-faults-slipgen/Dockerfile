FROM apache/airflow:2.7.1-python3.10
USER root
RUN apt-get update && apt-get install -y git libgl1-mesa-glx libglib2.0-0
USER airflow
COPY requirements.txt /requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir dvc mlflow imageio pillow scikit-image regex scipy fastjsonschema fsspec pyparsing optuna
# Removed other packages to install fast based on requirements.txt without CUDA headers
