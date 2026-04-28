from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
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


class LatentNN(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dims: list[int] | None = None, output_dim: int = 2704, dropout_prob: float = 0.3):
        super().__init__()
        hidden_dims = hidden_dims or [2048, 1024]
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.embeddings = nn.Parameter(torch.rand(num_embeddings, embedding_dim))

    def forward(self, inputs):
        input_shape = inputs.shape
        flat_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        distances = (
            torch.sum(flat_inputs ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings ** 2, dim=1)
            - 2 * torch.matmul(flat_inputs, self.embeddings.t())
        )
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings[encoding_indices]
        quantized = quantized.view(input_shape[0], input_shape[2], input_shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        commitment_loss = nn.functional.mse_loss(quantized.detach(), inputs)
        codebook_loss = nn.functional.mse_loss(quantized, inputs.detach())
        loss = self.beta * commitment_loss + codebook_loss
        quantized = inputs + (quantized - inputs).detach()
        return quantized, loss


class VQVAE(nn.Module):
    def __init__(self, latent_dim: int = 16, num_embeddings: int = 128, beta: float = 0.25):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_embeddings = num_embeddings
        self.beta = beta
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, latent_dim, kernel_size=1, stride=1, padding=0),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, beta)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss


class DecoderNet(nn.Module):
    def __init__(self, model_weights_path: str | Path, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.model = VQVAE(latent_dim=16, num_embeddings=128)
        if model_weights_path is not None:
            state = torch.load(model_weights_path, map_location=self.device)
            new_state = {}
            for k, v in state.items():
                new_state[k[len("model.") :]] = v if k.startswith("model.") else v
            self.model.load_state_dict(new_state)

    def forward(self, embedding):
        batch_size = embedding.size(0)
        latent = embedding.view(batch_size, 16, 13, 13)
        return self.model.decoder(latent)


def pixels_to_slip(image: np.ndarray, delta_z: float, normalizing_slip_range_path: Path) -> np.ndarray:
    normalizing_slip_range = np.load(normalizing_slip_range_path, allow_pickle=True)
    return (image * normalizing_slip_range) / delta_z


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
        latent_state = torch.load(self.latent_weights_path, map_location=self.device)
        input_dim = self._load_dataset_matrix().shape[1]

        inferred_hidden_dim = None
        if isinstance(latent_state, dict):
            first_linear_weight = latent_state.get("fc.0.weight")
            if first_linear_weight is not None and hasattr(first_linear_weight, "shape") and len(first_linear_weight.shape) == 2:
                inferred_hidden_dim = int(first_linear_weight.shape[0])

        if inferred_hidden_dim is None:
            raise RuntimeError("Could not infer latent hidden width from latent_model.pth")

        self.hidden_dims = [inferred_hidden_dim]

        self.latent_model = LatentNN(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=OUTPUT_DIM,
            dropout_prob=self.dropout_prob,
        )
        self.latent_model.load_state_dict(latent_state)
        self.latent_model.to(self.device).eval()

        self.decoder = DecoderNet(model_weights_path=str(self.decoder_weights_path), device=str(self.device))
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
            slip_plane = pixels_to_slip(image, req.dz, self.normalizing_slip_range_path)
        else:
            slip_plane = image

        image_stats = {
            "min": float(image.min()),
            "max": float(image.max()),
            "mean": float(image.mean()),
            "std": float(image.std()),
        }
        slip_stats = {
            "min": float(np.min(slip_plane)),
            "max": float(np.max(slip_plane)),
            "mean": float(np.mean(slip_plane)),
            "std": float(np.std(slip_plane)),
        }

        return {
            "predicted_image_2d": image.tolist(),
            "slip_plane_2d": slip_plane.tolist(),
            "slip_map_2d": slip_plane.tolist(),
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
