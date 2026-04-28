from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES: list[str] = [
    "LAT",
    "LON",
    "DEP",
    "STRK",
    "DIP",
    "RAKE",
    "LEN_f",
    "WID",
    "Htop",
    "HypX",
    "HypZ",
    "Nx",
    "Nz",
    "Dx",
    "Dz",
    "Mw",
]

OUTPUT_DIM = 2704


def compute_seismic_moment(mw: float) -> float:
    return 10 ** ((mw + 10.7) * 3 / 2)


def compute_rupture_dimensions(mo: float) -> tuple[float, float]:
    log10_mo = np.log10(mo)
    leff = 10 ** (-3.52 + 0.27 * log10_mo)
    weff = 10 ** (-4.34 + 0.29 * log10_mo)
    return leff, weff


def sample_truncated_normal(mean: float, std: float, lower: float, upper: float, rng: np.random.Generator) -> float:
    if std <= 0:
        return float(np.clip(mean, lower, upper))
    sample = rng.normal(mean, std)
    return float(np.clip(sample, lower, upper))


def compute_parameters(
    mw: float,
    strk: float,
    dip: float,
    rake: float,
    lat: float,
    lon: float,
    dep: float,
    nx: float,
    nz: float,
    dx: float,
    dz: float,
    random_seed: int | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    mo = compute_seismic_moment(mw)
    len_f, wid = compute_rupture_dimensions(mo)
    hypx = sample_truncated_normal(0.30, 0.13, 0.0, 0.5, rng) * len_f
    hypz = sample_truncated_normal(0.52, 0.23, 0.0, 1.0, rng) * wid
    htop = float(rng.uniform(5.0, 15.0))
    return np.array([lat, lon, dep, strk, dip, rake, len_f, wid, htop, hypx, hypz, nx, nz, dx, dz, mw], dtype=float)


@dataclass
class InferenceOutcome:
    image_2d: np.ndarray
    slip_map_2d: np.ndarray
    computed_parameters: np.ndarray
    image_stats: dict[str, float]
    slip_stats: dict[str, float]
    model_info: dict[str, Any]


class SlipgenInferenceService:
    def __init__(self, project_root: Path, model_dir: Path):
        # Import here to ensure sys.path is set up with PROJECT_ROOT
        # before importing modules from the latent-faults-slipgen project.
        from assets.utils import pixels_to_slip
        from scripts.decoder import Decoder
        from scripts.latent_mapper import LatentNN

        self.pixels_to_slip = pixels_to_slip
        self.Decoder = Decoder
        self.LatentNN = LatentNN

        self.project_root = Path(project_root)
        self.model_dir = Path(model_dir)
        self.dataset_path = self.project_root / "Dataset" / "text_vec.npy"
        self.scaler_path = self.project_root / "scaler_x.pkl"
        self.hyperparams_path = self.model_dir / "best_hyperparams.json"
        self.latent_weights_path = self.model_dir / "latent_model.pth"
        self.decoder_weights_path = self.model_dir / "decoder_model.pth"
        self.normalizing_slip_range_path = self.project_root / "assets" / "normalizing_slip_range.npy"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = str(self.device)
        self.ready = False
        self.torch_available = True

        self.latent_model: LatentNN | None = None
        self.decoder: Decoder | None = None
        self.scaler_x: StandardScaler | Any | None = None
        self.hidden_dims: list[int] = []
        self.dropout_prob: float = 0.0

        self._load_artifacts()

    def _load_dataset_matrix(self) -> np.ndarray:
        text_dict = np.load(self.dataset_path, allow_pickle=True).item()
        vectors: list[np.ndarray] = []
        for vec in text_dict.values():
            arr = np.asarray(vec, dtype=float).reshape(-1)
            if arr.size == len(FEATURE_NAMES):
                vectors.append(arr)
        if not vectors:
            raise ValueError(f"No valid feature vectors found in {self.dataset_path}")
        return np.vstack(vectors)

    def _load_scaler(self):
        if self.scaler_path.is_file():
            with open(self.scaler_path, "rb") as handle:
                return pickle.load(handle)

        matrix = self._load_dataset_matrix()
        scaler = StandardScaler()
        scaler.fit(matrix)
        return scaler

    def _load_artifacts(self) -> None:
        if not self.hyperparams_path.is_file():
            raise FileNotFoundError(f"Missing hyperparameters file: {self.hyperparams_path}")
        if not self.latent_weights_path.is_file():
            raise FileNotFoundError(f"Missing latent model weights: {self.latent_weights_path}")
        if not self.decoder_weights_path.is_file():
            raise FileNotFoundError(f"Missing decoder model weights: {self.decoder_weights_path}")

        with open(self.hyperparams_path, "r", encoding="utf-8") as handle:
            hyperparams = json.load(handle)

        self.dropout_prob = float(hyperparams.get("dropout_prob", 0.0))
        self.hidden_dims = [int(hyperparams.get("hidden_layer_1", 192))]
        input_dim = self._load_dataset_matrix().shape[1]

        self.latent_model = self.LatentNN(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=OUTPUT_DIM,
            dropout_prob=self.dropout_prob,
        )
        self.latent_model.load_state_dict(torch.load(self.latent_weights_path, map_location=self.device))
        self.latent_model.to(self.device).eval()

        self.decoder = self.Decoder(model_weights_path=str(self.decoder_weights_path), device=str(self.device))
        self.decoder.to(self.device).eval()

        self.scaler_x = self._load_scaler()
        self.ready = True

    def predict(self, req) -> dict[str, Any]:
        if not self.ready or self.latent_model is None or self.decoder is None or self.scaler_x is None:
            raise RuntimeError("Inference service is not ready")

        computed_parameters = compute_parameters(
            req.mw,
            req.strk,
            req.dip,
            req.rake,
            req.lat,
            req.lon,
            req.dep,
            req.nx,
            req.nz,
            req.dx,
            req.dz,
            req.random_seed,
        )

        scaled_parameters = self.scaler_x.transform(computed_parameters.reshape(1, -1))
        input_tensor = torch.tensor(scaled_parameters, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            latent_img = self.latent_model(input_tensor)
            predicted_image_tensor = self.decoder(latent_img)
            image = predicted_image_tensor[0, 0].detach().cpu().numpy()

        image = np.clip(image, 0.0, 1.0)
        if req.apply_dz:
            slip_map = self.pixels_to_slip(image, req.dz, image_name=None, plot=False)
        else:
            slip_map = image

        image_stats = {
            "min": float(image.min()),
            "max": float(image.max()),
            "mean": float(image.mean()),
            "std": float(image.std()),
        }
        slip_stats = {
            "min": float(np.min(slip_map)),
            "max": float(np.max(slip_map)),
            "mean": float(np.mean(slip_map)),
            "std": float(np.std(slip_map)),
        }

        return {
            "predicted_image_2d": image.tolist(),
            "slip_map_2d": slip_map.tolist(),
            "computed_parameters": {name: float(value) for name, value in zip(FEATURE_NAMES, computed_parameters, strict=True)},
            "image_stats": image_stats,
            "slip_stats": slip_stats,
            "model_info": {
                "project_root": str(self.project_root),
                "model_dir": str(self.model_dir),
                "latent_weights_path": str(self.latent_weights_path),
                "decoder_weights_path": str(self.decoder_weights_path),
                "scaler_path": str(self.scaler_path if self.scaler_path.is_file() else "derived_from_dataset"),
                "torch_device": str(self.device),
            },
        }
