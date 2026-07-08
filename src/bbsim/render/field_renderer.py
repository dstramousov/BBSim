"""Helpers for preparing numeric fields for visualization."""

from __future__ import annotations

import numpy as np


def field_to_display(field: np.ndarray) -> np.ndarray:
    """Normalize a field into the [0, 1] range for UI display.

    Args:
        field: Input numeric field.

    Returns:
        Float32 display array in the range [0, 1].
    """

    data = np.asarray(field, dtype=np.float32)
    min_value = float(data.min())
    max_value = float(data.max())
    span = max(max_value - min_value, 1.0e-8)
    return ((data - min_value) / span).astype(np.float32)
