from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram
import os
import sys
import time
from pathlib import Path

from config_loader import config


app = FastAPI(title="Slipgen MLOps API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus instrumentation at app creation time, before startup.
Instrumentator().instrument(app).expose(app)

REQUESTS_TOTAL = Counter(
    "slipgen_api_requests_total",
    "Total API requests processed by the Slipgen backend",
    ["endpoint", "status"],
)
INFERENCE_TIME = Histogram(
    "slipgen_inference_duration_seconds",
    "Latency of Slipgen model inference",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)
INPUT_MW = Histogram(
    "slipgen_input_magnitude_distribution",
    "Distribution of Mw inputs",
    buckets=(1, 3, 5, 6, 7, 8, 9, 10),
)
MODEL_READY = Gauge("slipgen_model_ready", "Whether the model and scaler are loaded")
LAST_INFERENCE = Gauge("slipgen_last_inference_timestamp_seconds", "Unix timestamp of the last successful inference")


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


def _candidate_project_roots() -> list[Path]:
    script_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []

    env_root = os.getenv("SLIPGEN_PROJECT_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    for base_dir in [script_dir, *script_dir.parents]:
        candidate = base_dir / "latent-faults-slipgen"
        if candidate.is_dir():
            candidates.append(candidate)

    return candidates


def _resolve_project_root() -> Path:
    for candidate in _candidate_project_roots():
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the latent-faults-slipgen project root. Checked: "
        + ", ".join(str(path) for path in _candidate_project_roots())
    )


PROJECT_ROOT = _resolve_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

MODEL_ARTIFACTS_DIR = Path(os.getenv("MODEL_ARTIFACTS_DIR", PROJECT_ROOT / "models"))

model_service = None
last_alert_payload: dict | None = None


@app.on_event("startup")
async def startup_event():
    global model_service
    # Ensure the latent-faults-slipgen project is importable before importing the
    # inference service (it imports modules from that project).
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))

    # Import here to avoid import-time failures when the project root isn't on sys.path
    # during module import. This import depends on PROJECT_ROOT being resolvable.
    from inference_service import SlipgenInferenceService

    model_service = SlipgenInferenceService(project_root=PROJECT_ROOT, model_dir=MODEL_ARTIFACTS_DIR)
    MODEL_READY.set(1.0 if model_service.ready else 0.0)

    print("Startup complete.")
    print("Project root resolved to:", PROJECT_ROOT)
    print("Model artifacts directory resolved to:", MODEL_ARTIFACTS_DIR)
    print("Torch available:", model_service.torch_available)


@app.get(
    "/health",
    responses={503: {"description": "Backend not initialized"}},
)
def health_check():
    if model_service is None:
        raise HTTPException(status_code=503, detail="State not initialized")
    return {
        "status": "healthy",
        "torch_available": model_service.torch_available,
        "model_ready": model_service.ready,
        "device": model_service.device_name,
        "project_root": str(PROJECT_ROOT),
        "model_artifacts_dir": str(MODEL_ARTIFACTS_DIR),
    }


@app.get(
    "/ready",
    responses={503: {"description": "Model service is not ready"}},
)
def readiness_check():
    if model_service is None or not model_service.ready:
        raise HTTPException(status_code=503, detail="Model service is not ready")
    return {"status": "ready", "device": model_service.device_name}


@app.post(
    "/predict",
    responses={503: {"description": "Model service is not ready"}, 500: {"description": "Inference failed"}},
)
async def predict_slip(req: InferenceRequest):
    if model_service is None or not model_service.ready:
        REQUESTS_TOTAL.labels(endpoint="predict", status="error").inc()
        raise HTTPException(status_code=503, detail="Model service is not ready")

    INPUT_MW.observe(req.mw)
    started = time.perf_counter()

    try:
        result = model_service.predict(req)
    except Exception as exc:
        REQUESTS_TOTAL.labels(endpoint="predict", status="error").inc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    duration = time.perf_counter() - started
    INFERENCE_TIME.observe(duration)
    LAST_INFERENCE.set(time.time())
    REQUESTS_TOTAL.labels(endpoint="predict", status="success").inc()

    return {
        "status": "success",
        **result,
        "inference_duration_seconds": duration,
        "parameters_used": req.model_dump(),
    }


@app.post("/api/alerts")
async def receive_alerts(request: Request):
    global last_alert_payload
    payload = await request.json()
    last_alert_payload = payload
    alerts = payload.get("alerts", []) if isinstance(payload, dict) else []
    return {"status": "received", "alerts": len(alerts)}


@app.get("/api/alerts/latest")
def latest_alerts():
    return {"status": "ok", "payload": last_alert_payload}
