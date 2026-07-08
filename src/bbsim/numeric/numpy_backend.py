"""NumPy implementation of BBSim numeric operations."""

from __future__ import annotations

import hashlib
import re

import numpy as np

from bbsim.core.config import SeedConfig
from bbsim.core.seed import PersonalSeed, SeedMetrics

_SEED_VERSION_PREFIX = "BBSimSeed:v1|"


def canonicalize_phrase(phrase: str) -> str:
    """Normalize a user phrase into a stable seed phrase.

    Args:
        phrase: User-provided seed phrase.

    Returns:
        Canonical lowercase phrase with collapsed whitespace.
    """

    normalized = phrase.casefold().strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized[:128] or "player"


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seed_int(root_hash_hex: str, stream: str) -> int:
    digest = hashlib.sha256(f"{root_hash_hex}|{stream}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _public_seed_code(phrase: str, root_hash_hex: str) -> str:
    letters = "".join(ch for ch in phrase.upper() if ch.isalnum())[:4] or "SEED"
    return f"{letters}-{root_hash_hex[:4].upper()}-{root_hash_hex[4:8].upper()}-{root_hash_hex[8:12].upper()}"


class NumpyBackend:
    """NumPy-backed implementation of the numeric backend protocol."""

    def create_personal_seed(self, config: SeedConfig) -> tuple[PersonalSeed, np.ndarray]:
        """Create deterministic seed metadata and a primordial fluctuation field."""

        phrase_canonical = canonicalize_phrase(config.phrase)
        root_hash_hex = _sha256_hex(_SEED_VERSION_PREFIX + phrase_canonical)
        rng = np.random.default_rng(_seed_int(root_hash_hex, "density"))

        field = self._generate_gaussian_field(config=config, rng=rng)
        metrics = self._measure_seed_field(field)
        seed = PersonalSeed(
            phrase_display=config.phrase,
            phrase_canonical=phrase_canonical,
            root_hash_hex=root_hash_hex,
            public_code=_public_seed_code(phrase_canonical, root_hash_hex),
            metrics=metrics,
        )
        return seed, field.astype(np.float32, copy=False)

    def normalize_field(self, field: np.ndarray) -> np.ndarray:
        """Return a zero-mean, unit-variance float32 field copy."""

        result = np.asarray(field, dtype=np.float32).copy()
        result -= float(result.mean())
        std = float(result.std())
        if std > 1.0e-8:
            result /= std
        return result

    def diffuse(self, field: np.ndarray, amount: float) -> np.ndarray:
        """Apply a small neighbor-average diffusion step.

        Args:
            field: Input field.
            amount: Diffusion amount in [0, 1]. Values outside the range are clamped.

        Returns:
            Diffused field copy.
        """

        alpha = float(np.clip(amount, 0.0, 1.0))
        base = np.asarray(field, dtype=np.float32)
        neighbors = (
            np.roll(base, 1, axis=0)
            + np.roll(base, -1, axis=0)
            + np.roll(base, 1, axis=1)
            + np.roll(base, -1, axis=1)
        ) * 0.25
        return ((1.0 - alpha) * base + alpha * neighbors).astype(np.float32)

    def _generate_gaussian_field(self, config: SeedConfig, rng: np.random.Generator) -> np.ndarray:
        size = config.grid_size
        if size < 16:
            raise ValueError("grid_size must be at least 16")

        kx = np.fft.fftfreq(size)
        ky = np.fft.fftfreq(size)
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
        k = np.sqrt(kx_grid * kx_grid + ky_grid * ky_grid)
        k[0, 0] = 1.0

        scale = max(float(config.fluctuation_scale), 1.0e-3)
        tilt = float(config.spectral_tilt)
        power = np.power(k, tilt) * np.exp(-np.square(k / scale))
        power[0, 0] = 0.0

        phases = rng.uniform(0.0, 2.0 * np.pi, size=(size, size))
        complex_field = np.sqrt(power) * (np.cos(phases) + 1j * np.sin(phases))
        field = np.fft.ifft2(complex_field).real
        field = self.normalize_field(field)
        field *= float(config.fluctuation_amplitude)
        return field.astype(np.float32)

    def _measure_seed_field(self, field: np.ndarray) -> SeedMetrics:
        normalized = self.normalize_field(field)
        spectrum = np.abs(np.fft.fft2(normalized)) ** 2
        size = normalized.shape[0]
        kx = np.fft.fftfreq(size)
        ky = np.fft.fftfreq(size)
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
        k = np.sqrt(kx_grid * kx_grid + ky_grid * ky_grid)

        low_band = spectrum[k < 0.08]
        high_band = spectrum[k > 0.20]
        total_power = float(spectrum.mean()) + 1.0e-8

        ripple_contrast = float(np.clip(field.std() / 0.75, 0.0, 1.0))
        large_scale_power = float(np.clip(low_band.mean() / total_power, 0.0, 1.0))
        fine_grain_power = float(np.clip(high_band.mean() / total_power, 0.0, 1.0))
        void_potential = float(np.clip(abs(float(field.min())) / 2.0, 0.0, 1.0))
        collapse_risk = float(np.clip(float(field.max()) / 2.0, 0.0, 1.0))
        isotropy = float(np.clip(1.0 - abs(large_scale_power - fine_grain_power), 0.0, 1.0))

        return SeedMetrics(
            ripple_contrast=ripple_contrast,
            large_scale_power=large_scale_power,
            fine_grain_power=fine_grain_power,
            isotropy=isotropy,
            void_potential=void_potential,
            collapse_risk=collapse_risk,
        )
