from __future__ import annotations

import numpy as np

from bbsim.core.config import SeedConfig
from bbsim.numeric.numpy_backend import NumpyBackend, canonicalize_phrase


def test_canonicalize_phrase_collapses_spaces_and_case() -> None:
    assert canonicalize_phrase("  DiMas   COSMOS  ") == "dimas cosmos"


def test_personal_seed_is_deterministic() -> None:
    backend = NumpyBackend()
    config = SeedConfig(phrase="Dimas", grid_size=64)

    seed_a, field_a = backend.create_personal_seed(config)
    seed_b, field_b = backend.create_personal_seed(config)

    assert seed_a.public_code == seed_b.public_code
    assert np.array_equal(field_a, field_b)


def test_different_phrases_produce_different_fields() -> None:
    backend = NumpyBackend()
    _, field_a = backend.create_personal_seed(SeedConfig(phrase="Dimas", grid_size=64))
    _, field_b = backend.create_personal_seed(SeedConfig(phrase="Pelevin", grid_size=64))

    assert not np.array_equal(field_a, field_b)


def test_seed_field_prefers_large_scale_structure() -> None:
    backend = NumpyBackend()
    seed, _ = backend.create_personal_seed(SeedConfig(phrase="Dimas", grid_size=128))

    assert seed.metrics.large_scale_power > seed.metrics.fine_grain_power
