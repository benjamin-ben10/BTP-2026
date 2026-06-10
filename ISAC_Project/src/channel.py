# =========================
# CHANNEL MODULE START
# =========================

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ChannelModel:
    """
    Small container to keep the main channel arrays together.
    """
    h_bi: np.ndarray   # BS -> IRS
    h_iu: np.ndarray   # IRS -> User
    cascaded: np.ndarray  # Element-wise cascaded channel


def _complex_normal(
    size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate CN(0,1) complex Gaussian samples.
    Each entry is (x + j y) / sqrt(2), where x,y ~ N(0,1).
    """
    real = rng.standard_normal(size)
    imag = rng.standard_normal(size)
    return (real + 1j * imag) / np.sqrt(2.0)


def generate_rayleigh_channel(
    num_elements: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a Rayleigh fading channel vector of length num_elements.
    """
    rng = np.random.default_rng(seed)
    return _complex_normal(num_elements, rng)


def generate_rician_channel(
    num_elements: int,
    k_factor: float = 1.0,
    los_phase: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a Rician fading channel vector.

    h = sqrt(K/(K+1)) * h_los + sqrt(1/(K+1)) * h_nlos

    Parameters
    ----------
    num_elements : int
        Number of IRS elements / channel coefficients.
    k_factor : float
        Rician K-factor. Larger values mean stronger LoS.
    los_phase : float
        Common LoS phase in radians.
    seed : Optional[int]
        Random seed for reproducibility.
    """
    if k_factor < 0:
        raise ValueError("k_factor must be non-negative.")

    rng = np.random.default_rng(seed)

    h_nlos = _complex_normal(num_elements, rng)
    h_los = np.exp(1j * los_phase) * np.ones(num_elements, dtype=np.complex128)

    a = np.sqrt(k_factor / (k_factor + 1.0))
    b = np.sqrt(1.0 / (k_factor + 1.0))

    return a * h_los + b * h_nlos


def generate_cascaded_channel(
    h_bi: np.ndarray,
    h_iu: np.ndarray,
) -> np.ndarray:
    """
    Element-wise cascaded channel between BS->IRS and IRS->User links.

    For each IRS element i:
        c[i] = h_bi[i] * h_iu[i]
    """
    if h_bi.shape != h_iu.shape:
        raise ValueError("h_bi and h_iu must have the same shape.")

    return h_bi * h_iu


def generate_irs_channels(
    num_elements: int,
    model: str = "rayleigh",
    k_factor: float = 1.0,
    seed: Optional[int] = None,
) -> ChannelModel:
    """
    Generate BS->IRS, IRS->User and cascaded channels in one call.

    model:
        "rayleigh" -> both links Rayleigh
        "rician"   -> both links Rician
    """
    rng = np.random.default_rng(seed)

    if model.lower() == "rayleigh":
        h_bi = _complex_normal(num_elements, rng)
        h_iu = _complex_normal(num_elements, rng)

    elif model.lower() == "rician":
        # Use independent seeds derived from the same generator
        seed_bi = rng.integers(0, 2**32 - 1)
        seed_iu = rng.integers(0, 2**32 - 1)

        h_bi = generate_rician_channel(
            num_elements=num_elements,
            k_factor=k_factor,
            los_phase=0.0,
            seed=int(seed_bi),
        )
        h_iu = generate_rician_channel(
            num_elements=num_elements,
            k_factor=k_factor,
            los_phase=np.pi / 4,
            seed=int(seed_iu),
        )
    else:
        raise ValueError("model must be either 'rayleigh' or 'rician'.")

    cascaded = generate_cascaded_channel(h_bi, h_iu)
    return ChannelModel(h_bi=h_bi, h_iu=h_iu, cascaded=cascaded)


def channel_magnitude_stats(h: np.ndarray) -> Tuple[float, float]:
    """
    Return mean and standard deviation of the channel magnitude.
    """
    mag = np.abs(h)
    return float(np.mean(mag)), float(np.std(mag))


# =========================
# CHANNEL MODULE END
# =========================

if __name__ == "__main__":
    model = generate_irs_channels(num_elements=8, model="rayleigh", seed=42)
    print("h_bi:", model.h_bi)
    print("h_iu:", model.h_iu)
    print("cascaded:", model.cascaded)