"""ctypes bindings for the Mojo kernels."""

from __future__ import annotations

import ctypes
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJODATASKETCH_LIB") or os.path.join(
    ROOT, "dist", "libmojo-datasketch.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "md_minhash_affine32": ([I, I, I, I, I, I], None),
    "md_minhash_affine64": ([I, I, I, I, I, I], None),
    "md_minhash_legacy": ([I, I, I, I, I, I], None),
    "md_jaccard32": ([I, I, I], F),
    "md_jaccard64": ([I, I, I], F),
    "md_merge32": ([I, I, I], None),
    "md_merge64": ([I, I, I], None),
    "md_hll_update32": ([I, I, I, I], None),
    "md_hll_update64": ([I, I, I, I], None),
    "md_hll_merge": ([I, I, I], None),
    "md_score32": ([I, I, I, I, I], None),
    "md_score64": ([I, I, I, I, I], None),
}

_lib: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not os.path.exists(LIB):
            raise RuntimeError(
                f"Mojo library not found at {LIB}; run `pixi run build` first"
            )
        _lib = ctypes.CDLL(LIB)
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_lib, name)
            function.argtypes = argtypes
            function.restype = restype
    return _lib


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def require_buffer(
    array,
    *,
    dtype,
    size: int,
    label: str,
    writable: bool = False,
) -> np.ndarray:
    """Validate an array immediately before exposing its address to Mojo."""
    expected = np.dtype(dtype)
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{label} must be a numpy.ndarray")
    if array.ndim != 1 or array.size != size:
        raise ValueError(f"{label} must be one-dimensional with length {size}")
    if array.dtype != expected:
        raise ValueError(f"{label} must have dtype {expected}")
    if not array.flags.c_contiguous:
        raise ValueError(f"{label} must be C-contiguous")
    if writable and not array.flags.writeable:
        raise ValueError(f"{label} must be writable")
    if size and addr(array) == 0:
        raise ValueError(f"{label} has a null data pointer")
    return array
