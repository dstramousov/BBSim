from __future__ import annotations

import pytest

from bbsim.core.config import CosmologyConfig
from bbsim.core.expansion import compute_hubble_gyr_inv, detect_era


def test_hubble_requires_positive_scale_factor() -> None:
    with pytest.raises(ValueError):
        compute_hubble_gyr_inv(0.0, CosmologyConfig())


def test_hubble_is_positive_for_default_config() -> None:
    assert compute_hubble_gyr_inv(1.0, CosmologyConfig()) > 0.0


def test_detect_era_changes_with_scale_factor() -> None:
    config = CosmologyConfig()
    assert detect_era(1.0e-6, config) == "radiation"
    assert detect_era(1.0, config) == "dark_energy"
