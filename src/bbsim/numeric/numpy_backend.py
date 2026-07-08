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

    def apply_inflation_smoothing(self, field: np.ndarray, smoothing: float) -> np.ndarray:
        """Suppress fine fluctuation noise while keeping the large-scale seed pattern.

        Inflation in the prototype is represented in comoving coordinates: the grid
        size does not grow, but the physical scale of each cell does. This helper
        creates the checkpoint visual field by damping high-frequency power instead
        of resampling the array into a larger texture.
        """

        base = self.normalize_field(field)
        size = base.shape[0]
        strength = float(np.clip(smoothing, 0.0, 1.0))

        spectrum = np.fft.rfft2(base)
        kx = np.fft.fftfreq(size)
        ky = np.fft.rfftfreq(size)
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
        k = np.sqrt(kx_grid * kx_grid + ky_grid * ky_grid)

        # Keep the strongest large-scale modes and increasingly damp fine modes as
        # smoothing grows. The blend preserves some texture so the result is not a
        # featureless cloud after inflation.
        cutoff = 0.035 + 0.030 * (1.0 - strength)
        low_pass = np.exp(-np.power(k / cutoff, 4.0))
        low_pass[0, 0] = 0.0

        smoothed = np.fft.irfft2(spectrum * low_pass, s=base.shape).real
        smoothed = self.normalize_field(smoothed)
        blend = 0.18 * (1.0 - strength)
        result = (1.0 - blend) * smoothed + blend * base
        result = self.normalize_field(result)
        result *= float(np.std(field))
        return result.astype(np.float32)

    def _generate_gaussian_field(self, config: SeedConfig, rng: np.random.Generator) -> np.ndarray:
        size = config.grid_size
        if size < 16:
            raise ValueError("grid_size must be at least 16")

        # Start from real white noise, then shape it in Fourier space. This keeps
        # the field deterministic and real-valued while avoiding the raw
        # television-static look of unfiltered pixel noise. The result is still a
        # prototype Gaussian random field, but with visible large-scale structure.
        white_noise = rng.normal(loc=0.0, scale=1.0, size=(size, size))
        spectrum = np.fft.rfft2(white_noise)

        kx = np.fft.fftfreq(size)
        ky = np.fft.rfftfreq(size)
        kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
        k = np.sqrt(kx_grid * kx_grid + ky_grid * ky_grid)

        scale = float(np.clip(config.fluctuation_scale, 0.05, 1.0))
        cutoff = 0.025 + 0.075 * scale
        tilt = float(np.clip(config.spectral_tilt, 0.1, 2.0))
        k_floor = 1.0 / float(size)

        large_scale_filter = np.exp(-np.square(k / cutoff))
        tilt_filter = np.power(k + k_floor, -0.5 * tilt)
        field_filter = large_scale_filter * tilt_filter
        field_filter[0, 0] = 0.0

        filtered_spectrum = spectrum * field_filter
        field = np.fft.irfft2(filtered_spectrum, s=(size, size)).real

        # Add a small deterministic fine component so the seed does not look like
        # one over-smoothed cloud. The fine layer is deliberately weak; the player
        # should see future voids and dense regions first, not raw static.
        fine_noise = rng.normal(loc=0.0, scale=1.0, size=(size, size))
        field = self.normalize_field(field) + 0.08 * self.normalize_field(fine_noise)
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

        total_power = float(spectrum.sum()) + 1.0e-8
        large_power = float(spectrum[k < 0.08].sum()) / total_power
        fine_power = float(spectrum[k > 0.20].sum()) / total_power
        mid_power = max(0.0, 1.0 - large_power - fine_power)

        low_percentile = float(np.percentile(field, 5.0))
        high_percentile = float(np.percentile(field, 95.0))
        ripple_contrast = float(np.clip(field.std() / 0.75, 0.0, 1.0))
        large_scale_power = float(np.clip(large_power + 0.35 * mid_power, 0.0, 1.0))
        fine_grain_power = float(np.clip(fine_power, 0.0, 1.0))
        void_potential = float(np.clip(abs(low_percentile) / 0.75, 0.0, 1.0))
        collapse_risk = float(np.clip(high_percentile / 0.75, 0.0, 1.0))
        isotropy = float(np.clip(1.0 - abs(large_scale_power - 0.65), 0.0, 1.0))

        return SeedMetrics(
            ripple_contrast=ripple_contrast,
            large_scale_power=large_scale_power,
            fine_grain_power=fine_grain_power,
            isotropy=isotropy,
            void_potential=void_potential,
            collapse_risk=collapse_risk,
        )
