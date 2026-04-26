# MLOps Pipeline Report

## Overview
This document outlines the setup for automating the training of the latent faults modeling application using MLOps principles.

## Configurations & Architecture
- **Git** is used for version control.
- **Docker & Docker Compose** ensure environment parity between local testing and deployment, packaging `apache/airflow` along with the project required dependencies.
- **Apache Airflow** serves as the orchestrator to schedule and run ML tasks. 

## A. Problem Definition & Data Collection
Data resides in local directories (or Git LFS/DVC) which is mounted to the Docker containers to keep the environment consistent.

## B. Data Preprocessing & Feature Engineering
Features extracted (such as text vectors) are stored and versioned to maintain base metrics for drift tracking and evaluations.

## C. Model Development & Training
The automated workflow is built through an Airflow DAG (`train_mapper_decoder_pipeline`) which launches the model training execution inside our unified container environment (`train_mapper_decoder.py`). The setup supports tracking via MLFlow (installed via Docker) to track hyperparameters, iterations, and best-performing weights without manual intervention.

## Instructions
1. Navigate to `MLOPS_Project/latent-faults-slipgen`.
2. Start the Airflow infrastructure with Docker Compose: `docker-compose up --build -d`
3. Access Airflow at `http://localhost:8080`.
4. Trigger the `train_mapper_decoder_pipeline` DAG to execute the training.