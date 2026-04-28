# Latent Faults Slip Generation - MLOps Project

> **A complete MLOps pipeline for stochastic earthquake slip generation using latent-space surrogate models**

This repository implements a production-ready MLOps infrastructure for the **latent-faults-slipgen** project—a machine learning system that generates earthquake slip maps from sparse seismic source parameters using generative AI.

---

## 📋 Quick Navigation

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Detailed Setup Guide](#detailed-setup-guide)
- [Running Components](#running-components)
- [Project Structure](#project-structure)
- [Monitoring & Dashboards](#monitoring--dashboards)
- [Common Issues & Troubleshooting](#common-issues--troubleshooting)
- [Advanced Usage](#advanced-usage)

---

## Project Overview

This project solves the earthquake slip inversion problem using a two-stage deep learning pipeline:

1. **VQ-VAE Representation Learning** - Learns a compressed latent space from earthquake slip images
2. **Conditional Generation** - Maps seismic source parameters → latent space → slip map reconstruction

### Key Components

| Component | Purpose | Technology |
|-----------|---------|-----------|
| **Backend API** | FastAPI inference service | Python FastAPI + PyTorch |
| **Frontend UI** | Interactive parameter adjustment & visualization | React + Vite |
| **ML Pipeline** | Model training & optimization | PyTorch + MLflow |
| **Orchestration** | Workflow scheduling & monitoring | Apache Airflow |
| **Monitoring** | Metrics collection & alerting | Prometheus + Grafana + AlertManager |

### Performance Highlights

- **2D PSD Correlation**: ~0.93
- **Radial PSD Correlation**: ~0.96
- **Dataset**: 200 SRCMOD earthquake events
- **Grid Size**: 50 × 50 standardized slip patches

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User (Git Cloner)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
   ┌────▼────┐   ┌────▼────┐   ┌───▼──────┐
   │ Frontend │   │ Airflow  │   │ Training │
   │ React UI │   │ DAGs     │   │ Scripts  │
   └────┬────┘   └────┬────┘   └───┬──────┘
        │             │             │
        └─────────────┼─────────────┘
                      │
              ┌───────▼────────┐
              │   Backend API  │
              │    FastAPI     │
              └───────┬────────┘
                      │
        ┌─────────────┼──────────────┐
        │             │              │
   ┌────▼────┐   ┌────▼─────┐   ┌──▼────────┐
   │  Models │   │  Assets  │   │ Monitoring│
   │ (PyTorch)   │ (Config) │   │(Prometheus)
   └─────────┘   └──────────┘   └───────────┘
```

---

## Prerequisites

### System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS, or Windows with WSL2
- **RAM**: Minimum 8GB (16GB+ recommended for training)
- **Disk**: 20GB free space for models, data, and Docker images
- **GPU** (Optional but recommended):
  - NVIDIA GPU with CUDA 11.8+ support
  - NVIDIA Driver 525+
  - nvidia-docker2 (for containerized GPU access)

### Software Prerequisites

1. **Python** (3.10+)
   ```bash
   python --version  # Should be 3.10 or higher
   ```

2. **Docker & Docker Compose**
   ```bash
   docker --version        # Docker 24.0+
   docker compose version  # Docker Compose 2.20+
   ```

3. **Git**
   ```bash
   git --version  # Any recent version
   ```

4. **Conda** (Optional, but recommended for environment management)
   ```bash
   conda --version
   ```

### Verify Prerequisites

```bash
# Check all prerequisites in one command
python --version && docker --version && docker compose version && git --version
```

---

## Quick Start

### 1️⃣ Clone & Navigate

```bash
git clone <repository-url>
cd MLOPS_Project
```

### 2️⃣ Set Up Python Environment

```bash
# Option A: Using Conda (recommended)
conda create -n slipgen python=3.10
conda activate slipgen

# Option B: Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install ML Pipeline Dependencies

```bash
cd latent-faults-slipgen
pip install -r requirements.txt
cd ..
```

### 4️⃣ Start All Services with Docker Compose

```bash
cd deploy
docker compose up -d  # Run in background
# Or: docker compose up  # Run in foreground (useful for debugging)
```

### 5️⃣ Verify Services Are Running

```bash
# Check container status
docker compose ps

# Expected output: All containers should show "Up"
# - slip_fastapi          (Backend API)
# - slip_react_vite       (Frontend UI)
# - slip_prometheus       (Metrics)
# - slip_grafana          (Dashboards)
# - slip_airflow          (Orchestration)
# - slip_alertmanager     (Alerts)
```

### 6️⃣ Access the Applications

Once all containers are running:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Frontend UI** | http://localhost:3000 | (None needed) |
| **Backend API** | http://localhost:8000 | (None needed) |
| **API Docs** | http://localhost:8000/docs | (Auto-generated Swagger) |
| **Airflow** | http://localhost:8080 | admin / admin |
| **Prometheus** | http://localhost:9090 | (None needed) |
| **Grafana** | http://localhost:3001 | admin / admin |
| **AlertManager** | http://localhost:9093 | (None needed) |

---

## Detailed Setup Guide

### Step 1: Environment Setup

#### Using Conda (Recommended)

```bash
# Create isolated environment
conda create -n slipgen python=3.10 -y
conda activate slipgen

# Verify activation
python -c "import sys; print(sys.prefix)"
```

#### Using venv

```bash
# Create virtual environment
python -m venv venv

# Activate (choose based on OS)
# Linux/macOS:
source venv/bin/activate

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Windows (CMD):
venv\Scripts\activate.bat

# Verify activation (should show (venv) in terminal prompt)
```

### Step 2: Install Dependencies

#### ML Pipeline Dependencies

```bash
cd latent-faults-slipgen
pip install --upgrade pip setuptools wheel

# Install all ML dependencies
pip install -r requirements.txt

# Install DVC for data versioning (optional but recommended)
pip install dvc dvc[gs]  # Add [gs] for Google Cloud, [s3] for AWS

# Return to project root
cd ..
```

#### Backend API Dependencies

```bash
cd deploy/backend
pip install -r requirements.txt
cd ../..
```

### Step 3: Configure Docker Environment

#### Create Environment File (Optional)

```bash
cd deploy

# Create .env file to customize ports/settings
cat > .env << EOF
API_PORT=8000
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
EOF

cd ..
```

#### Build Docker Images

```bash
cd deploy

# Build images (without cache for clean build)
docker compose build

# Or force rebuild if needed:
docker compose build --no-cache
```

### Step 4: Initialize Airflow Database

```bash
cd deploy

# Start Airflow to initialize database
docker compose up -d airflow-webserver

# Wait 30 seconds for initialization
sleep 30

# Verify Airflow is ready
docker logs slip_airflow | tail -20
```

### Step 5: Train/Prepare Models (First Time Only)

```bash
# Go to ML pipeline directory
cd latent-faults-slipgen

# Train VQ-VAE model
python scripts/train_vqvae.py

# Train mapper & decoder
python scripts/train_mapper_decoder.py

# Verify models were created
ls -la models/
# Expected files:
# - latent_model.pth
# - decoder_model.pth
# - best_hyperparams.json
```

### Step 6: Start All Services

```bash
cd deploy

# Start all containers
docker compose up -d

# Monitor startup (use Ctrl+C to exit)
docker compose logs -f

# Or check individual service health:
docker compose ps
```

---

## Running Components

### 🔄 ML Training Pipeline

#### Training from Scratch

```bash
cd latent-faults-slipgen

# Stage 1: Train VQ-VAE representation model
python scripts/train_vqvae.py
# Expected output:
# - models/vqvae_finetuned.pth
# - Logs in mlruns/0/

# Stage 2: Train mapper & decoder
python scripts/train_mapper_decoder.py
# Expected output:
# - models/latent_model.pth
# - models/decoder_model.pth
# - models/best_hyperparams.json
```

#### Running Inference

```bash
# Single event inference
python scripts/run_inference.py \
  --event-id "s2011TOHOKU01LAYx" \
  --output-dir "predictions/"

# Batch inference
python scripts/run_inference.py \
  --batch \
  --output-dir "predictions/"
```

#### Hyperparameter Tuning

```bash
# Automated tuning
python scripts/tune_mapper.py \
  --n-trials 100 \
  --output-dir "hyperparams/"
```

### 🌐 Backend API Service

#### Start Backend (Containerized)

```bash
cd deploy
docker compose up -d backend
# Verify:
curl http://localhost:8000/health
```

#### Start Backend (Local Development)

```bash
cd deploy/backend

# Activate Python environment
conda activate slipgen  # or source venv/bin/activate

# Run directly (hot-reload on file changes)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Test Backend Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Generate slip map
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "mw": 5.5,
    "lat": 40.0,
    "lon": 141.0,
    "depth": 25.0
  }'

# View all available endpoints
curl http://localhost:8000/docs  # Open in browser
```

### 🎨 Frontend Application

#### Start Frontend (Containerized)

```bash
cd deploy
docker compose up -d frontend
# Access: http://localhost:3000
```

#### Start Frontend (Local Development)

```bash
cd deploy/frontend

# Install dependencies
npm install

# Start dev server (hot-reload)
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### ⚙️ Airflow Orchestration

#### Access Airflow UI

```bash
# Start Airflow
cd deploy
docker compose up -d airflow-webserver

# Open browser: http://localhost:8080
# Login: admin / admin
```

#### Trigger DAGs

```bash
# List available DAGs
docker exec slip_airflow airflow dags list

# Trigger a DAG run
docker exec slip_airflow airflow dags trigger slipgen_retraining_pipeline

# View DAG status
docker exec slip_airflow airflow dags status slipgen_retraining_pipeline
```

#### View Airflow Logs

```bash
# Stream logs from running DAG
docker compose logs -f airflow-webserver

# Or check container logs directly
docker exec slip_airflow cat /opt/airflow/logs/slipgen_retraining_pipeline/*/scheduler.log
```

### 📊 Monitoring & Metrics

#### Prometheus Metrics

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Query metrics manually
curl 'http://localhost:9090/api/v1/query?query=up'

# View metrics UI
# Open: http://localhost:9090
```

#### Grafana Dashboards

```bash
# Access Grafana
# Open: http://localhost:3001
# Login: admin / admin

# Default dashboards:
# - Slip Generation Pipeline (main metrics)
# - FastAPI Performance
# - Infrastructure Health
```

#### AlertManager

```bash
# Check active alerts
curl http://localhost:9093/api/v1/alerts

# View Alert Manager UI
# Open: http://localhost:9093
```

---

## Project Structure

```
MLOPS_Project/
├── README.md                          # This file
├── config.yaml                        # Central configuration for all components
├── interactive_slip_app.py            # Streamlit interactive demo app
│
├── latent-faults-slipgen/             # ML Pipeline Core
│   ├── train_mapper_decoder.py        # Main mapper & decoder training script
│   ├── requirements.txt               # Python dependencies
│   ├── params.yaml                    # DVC parameters
│   ├── dvc.yaml                       # DVC pipeline configuration
│   │
│   ├── scripts/
│   │   ├── train_vqvae.py            # VQ-VAE representation learning
│   │   ├── train_mapper_decoder.py   # Mapper & decoder training
│   │   ├── run_inference.py          # Batch inference runner
│   │   ├── latent_mapper.py          # Mapper model definition
│   │   ├── decoder.py                # Decoder model definition
│   │   └── tune_mapper.py            # Hyperparameter tuning
│   │
│   ├── Dataset/
│   │   ├── filtered_images_train/    # Training slip images
│   │   ├── filtered_images_test/     # Test slip images
│   │   ├── slip_arrays_inference/    # Pre-computed slip arrays
│   │   └── text_vec.npy             # Pre-computed feature vectors
│   │
│   ├── models/
│   │   ├── latent_model.pth         # Trained mapper model
│   │   ├── decoder_model.pth        # Trained decoder model
│   │   └── best_hyperparams.json    # Optimal hyperparameters
│   │
│   ├── assets/
│   │   ├── dz.json                  # Depth zone configuration
│   │   ├── normalizing_slip_range.npy  # Normalization constants
│   │   └── utils.py                 # Shared utilities
│   │
│   └── mlruns/                       # MLflow experiment tracking
│
├── deploy/                            # Production Deployment
│   ├── docker-compose.yml            # Multi-container orchestration
│   │
│   ├── backend/                      # FastAPI Backend
│   │   ├── main.py                  # FastAPI application entry
│   │   ├── inference_service.py     # Inference logic
│   │   ├── config_loader.py         # Configuration management
│   │   ├── requirements.txt         # Backend dependencies
│   │   ├── Dockerfile              # Backend container image
│   │   ├── run_tests.sh            # Test runner script
│   │   └── tests/
│   │       └── test_api.py          # API endpoint tests
│   │
│   ├── frontend/                     # React + Vite Frontend
│   │   ├── index.html               # HTML entry point
│   │   ├── vite.config.js           # Vite bundler config
│   │   ├── package.json             # Frontend dependencies
│   │   ├── Dockerfile              # Frontend container image
│   │   └── src/
│   │       ├── main.jsx             # App entry point
│   │       ├── App.jsx              # Main component
│   │       └── index.css            # Global styles
│   │
│   ├── prometheus/                   # Metrics Collection
│   │   ├── prometheus.yml           # Prometheus config
│   │   └── alert.rules.yml          # Alert rules
│   │
│   ├── grafana/                      # Dashboards & Visualization
│   │   ├── dashboards/
│   │   │   └── slipgen_dashboard.json  # Main dashboard definition
│   │   └── provisioning/
│   │       ├── dashboards/          # Dashboard provisioning
│   │       └── datasources/         # Data source config
│   │
│   └── alertmanager/                 # Alert Routing
│       └── alertmanager.yml         # Alert configuration
│
└── airflow/                           # Workflow Orchestration
    ├── dags/
    │   └── slipgen_pipeline_dag.py  # Main pipeline DAG
    ├── logs/                         # Execution logs
    └── plugins/                      # Custom Airflow plugins
```

---

## Monitoring & Dashboards

### Prometheus Metrics

The backend API exposes metrics on `/metrics` endpoint:

```bash
# View raw metrics
curl http://localhost:8000/metrics | grep slip_generation

# Key metrics:
# - slip_generation_requests_total    # Total requests
# - slip_generation_request_duration_seconds  # Request latency
# - slip_generation_model_accuracy    # Model performance
```

### Grafana Dashboards

Pre-configured dashboards available at http://localhost:3001:

1. **Slip Generation Pipeline** - Main performance dashboard
2. **FastAPI Performance** - API latency and throughput
3. **Infrastructure Health** - Container & system metrics

### Creating Custom Alerts

Edit [deploy/prometheus/alert.rules.yml](deploy/prometheus/alert.rules.yml):

```yaml
groups:
- name: slipgen_alerts
  rules:
  - alert: HighAPILatency
    expr: slip_generation_request_duration_seconds > 5
    for: 5m
    annotations:
      summary: "API latency exceeds 5 seconds"
```

---

## Common Issues & Troubleshooting

### ❌ Docker Compose Build Fails

**Symptom**: `ERROR: failed to solve`

**Solutions**:

```bash
# 1. Update Docker to latest version
docker --version  # Should be 24.0+

# 2. Clean up existing images
docker compose down -v
docker system prune -a

# 3. Rebuild with verbose output
docker compose build --no-cache --progress=plain

# 4. Check disk space
df -h  # Need at least 20GB free
```

### ❌ API Container Crashes

**Symptom**: `slip_fastapi` container exits immediately

```bash
# Check logs
docker logs slip_fastapi

# Common causes:
# - Missing models: verify latent-faults-slipgen/models/ exist
# - Port already in use: change API_PORT in .env

# Restart container
docker compose restart backend
```

### ❌ Models Not Found

**Symptom**: `FileNotFoundError: latent_model.pth`

```bash
# Verify models exist
ls -la latent-faults-slipgen/models/

# If missing, train models
cd latent-faults-slipgen
python scripts/train_mapper_decoder.py
cd ..
```

### ❌ Out of Memory During Training

**Symptom**: Python process killed, or "CUDA out of memory"

```bash
# Reduce batch size in config.yaml
# Change: batch_size: 32 → batch_size: 16

# Or run on CPU:
python scripts/train_mapper_decoder.py --device cpu

# Check system resources
# Linux: free -h
# macOS: vm_stat
# Windows: Task Manager
```

### ❌ Airflow DAGs Not Visible

**Symptom**: No DAGs shown in Airflow UI

```bash
# Restart Airflow scheduler
docker compose restart airflow-webserver

# Check DAG parsing errors
docker exec slip_airflow airflow dags list --report

# View DAG logs
docker logs slip_airflow | grep slipgen_pipeline_dag
```

### ❌ Frontend Can't Connect to Backend

**Symptom**: CORS errors or "Cannot GET /api/..."

```bash
# Verify backend is running
curl http://localhost:8000/health

# Check frontend config
cat deploy/frontend/src/config.js  # Verify API endpoint

# Restart both containers
docker compose restart backend frontend
```

### ✅ Health Check Commands

```bash
# All services
docker compose ps

# Individual health checks
curl http://localhost:8000/health        # Backend API
curl http://localhost:3000               # Frontend
curl http://localhost:8080/home          # Airflow
curl http://localhost:9090/-/healthy     # Prometheus
curl http://localhost:3001/api/health    # Grafana
```

---

## Advanced Usage

### Custom Model Deployment

```bash
# 1. Train new model
cd latent-faults-slipgen
python scripts/train_mapper_decoder.py --output-dir models/custom/

# 2. Copy to backend models directory
cp models/custom/*.pth ../../deploy/backend/models/

# 3. Restart backend
cd ../../deploy
docker compose restart backend
```

### Database Persistence

By default, Airflow uses SQLite (ephemeral). For production:

```yaml
# In docker-compose.yml, replace SQLite with PostgreSQL:
airflow-db:
  image: postgres:15
  environment:
    POSTGRES_DB: airflow
    POSTGRES_USER: airflow
    POSTGRES_PASSWORD: airflow
  volumes:
    - postgres_data:/var/lib/postgresql/data

# Update Airflow config:
airflow-webserver:
  environment:
    - AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@airflow-db:5432/airflow
```

### GPU Support for Inference

```bash
# Check NVIDIA GPU availability
nvidia-smi

# Enable GPU in docker-compose.yml:
backend:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]

# Restart with GPU
docker compose up -d backend
```

### Running Tests

```bash
# Backend API tests
cd deploy/backend
./run_tests.sh

# ML pipeline tests (if available)
cd ../../latent-faults-slipgen
python -m pytest tests/
```

### Accessing Container Shells

```bash
# Backend container
docker exec -it slip_fastapi /bin/bash

# ML environment
docker exec -it slip_fastapi python -c "import torch; print(torch.__version__)"

# Airflow
docker exec -it slip_airflow bash
```

---

## Performance Optimization Tips

| Scenario | Optimization |
|----------|-------------|
| **Slow inference** | Enable GPU: `device: cuda` in config.yaml |
| **High API latency** | Increase worker processes: `workers: 8` in config.yaml |
| **Memory issues** | Reduce batch size in config.yaml or training scripts |
| **Slow dashboard loading** | Increase Prometheus retention: `--storage.tsdb.retention.time=30d` |
| **Pipeline bottlenecks** | Check Airflow logs; parallelize DAG tasks |

---

## Additional Resources

### Documentation

- [ML Pipeline Details](latent-faults-slipgen/README.md) - Detailed model architecture and paper
- [Backend API Docs](http://localhost:8000/docs) - Auto-generated OpenAPI spec
- [Configuration Guide](config.yaml) - All configurable parameters

### External Links

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Apache Airflow Docs](https://airflow.apache.org/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

---

## Support & Contributing

For issues, questions, or contributions:

1. Check [Troubleshooting](#common-issues--troubleshooting) section
2. Review existing issues/documentation
3. Enable debug logging:
   ```bash
   export LOG_LEVEL=DEBUG
   docker compose up
   ```

---

## License

This project is licensed under the MIT License. See LICENSE file for details.

---

## Citation

If you use this MLOps infrastructure, please cite:

```bibtex
@article{latent-faults-slipgen,
  title={Latent Faults: A Latent-Space Surrogate Model for Stochastic Earthquake Slip Generation},
  year={2026}
}
```

---

**Last Updated**: April 28, 2026 | **Version**: 1.0.0
