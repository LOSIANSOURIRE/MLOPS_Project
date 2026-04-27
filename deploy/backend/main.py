from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram
import torch
import numpy as np
import os
import sys

# Mount the parent directory so we can import latent-faults-slipgen scripts
sys.path.append("/app/latent-faults-slipgen")

from config_loader import config

try:
    from scripts.decoder import Decoder
    from scripts.latent_mapper import LatentNN
    from scripts.run_inference import run_inference, compute_parameters
    from assets.utils import pixels_to_slip
except ImportError as e:
    print(f"Warning: Could not import model scripts: {e}")

app = FastAPI(title="Slipgen MLOps API")

# Custom Prometheus Metrics
INFERENCE_TIME = Histogram('inference_latency_seconds', 'Latency of model inference')
INPUT_MW = Histogram('input_magnitude_distribution', 'Distribution of Mw inputs', buckets=[1,3,5,6,7,8,9,10])

class InferenceRequest(BaseModel):
    mw: float = Field(default=config["inference"]["default_mw"], ge=config["inference"]["mw_min"], le=config["inference"]["mw_max"], description="Moment Magnitude")
    strk: float = Field(default=180.0, description="Strike")
    dip: float = Field(default=45.0, description="Dip")
    rake: float = Field(default=90.0, description="Rake")
    lat: float = Field(default=0.0, description="Latitude")
    lon: float = Field(default=0.0, description="Longitude")
    dep: float = Field(default=10.0, description="Depth")
    nx: float = Field(default=config["inference"]["grid_size_x"], description="Nx")
    nz: float = Field(default=config["inference"]["grid_size_z"], description="Nz")
    dx: float = Field(default=1.0, description="Dx")
    dz: float = Field(default=1.0, description="Dz")
    random_seed: int = Field(default=config["train"]["seed"], description="Random Seed")
    apply_dz: bool = Field(default=False)

# Lifespan/State variables
model_state = {}

@app.on_event("startup")
async def startup_event():
    # Instrument Prometheus
    Instrumentator().instrument(app).expose(app)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_state['device'] = device
    
    # Normally we would load models here using paths from env
    # e.g., artifacts_dir = os.environ.get("MODEL_ARTIFACTS_DIR", "")
    print("Startup complete. Hardware targeted:", device)

@app.get("/health")
def health_check():
    """Health check for CI/CD and Container Orchestration"""
    if 'device' not in model_state:
        raise HTTPException(status_code=503, detail="State not initialized")
    return {"status": "healthy", "device": str(model_state['device'])}

@app.get("/ready")
def readiness_check():
    return {"status": "ready"}

@app.post("/predict")
async def predict_slip(req: InferenceRequest):
    INPUT_MW.observe(req.mw)
    
    with INFERENCE_TIME.time():
        # TODO: Replace with the actual model call using real parameters
        # For layout demonstration returning a mock spatial matrix
        pred_slip = np.random.rand(50, 50).tolist()
    
    return {
        "status": "success",
        "parameters_used": req.dict(),
        "slip_map_2d": pred_slip # Send raw data to React to be rendered
    }
