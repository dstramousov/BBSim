"""Small dependency-free PNG writer for generated RGB frames."""

from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import zlib

import numpy as np


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def save_rgb_png(path: Path, rgb: np.ndarray) -> None:
    """Save an RGB float or uint8 array as a PNG file.

    Args:
        path: Destination PNG path. Parent directories are created automatically.
        rgb: RGB image with shape ``(height, width, 3)``. Floating point arrays are
            interpreted as the range ``[0, 1]``. Unsigned byte arrays are written as-is.

    Raises:
        ValueError: If the input array is not a finite RGB image.
        OSError: If the file cannot be written.
    """

    image = _to_uint8_rgb(rgb)
    height, width, _ = image.shape
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    chunks = [
        _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        _png_chunk(b"IDAT", zlib.compress(_scanlines(image), level=6)),
        _png_chunk(b"IEND", b""),
    ]

    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(_PNG_SIGNATURE)
        for chunk in chunks:
            handle.write(chunk)

    tmp_path.replace(path)


def _to_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    data = np.asarray(rgb)
    if data.ndim != 3 or data.shape[2] != 3:
        raise ValueError(f"expected RGB image with shape (height, width, 3), got {data.shape}")
    if data.shape[0] <= 0 or data.shape[1] <= 0:
        raise ValueError("expected a non-empty RGB image")

    if data.dtype == np.uint8:
        return np.ascontiguousarray(data)

    float_data = np.asarray(data, dtype=np.float32)
    if not np.isfinite(float_data).all():
        raise ValueError("RGB image contains non-finite values")
    return np.ascontiguousarray(np.clip(float_data, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _scanlines(image: np.ndarray) -> bytes:
    # PNG filter type 0 keeps the writer simple and deterministic.
    rows = [b"\x00" + image[row].tobytes() for row in range(image.shape[0])]
    return b"".join(rows)


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)
