"""Mojo-backed MinHash compatible with datasketch 2.x."""

from __future__ import annotations

import copy
import operator
import warnings
from collections.abc import Generator, Iterable
from typing import Callable, Literal

import numpy as np

from ._lib import addr, lib, require_buffer
from .hashfunc import sha1_hash32, sha1_hash64

_MERSENNE_PRIME = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 32) - 1)
_HASH_RANGE = 1 << 32
_VALID_SCHEMES = ("affine32", "affine64", "legacy")
_DTYPES = {
    "affine32": np.dtype(np.uint32),
    "affine64": np.dtype(np.uint64),
    "legacy": np.dtype(np.uint64),
}
_MAX_HASHES = {
    "affine32": np.uint32((1 << 32) - 1),
    "affine64": np.uint64((1 << 64) - 1),
    "legacy": _MAX_HASH,
}


def _unsigned_array(values, dtype: np.dtype, label: str) -> np.ndarray:
    """Convert integer values without NumPy's implicit wraparound narrowing."""
    maximum = int(np.iinfo(dtype).max)
    if isinstance(values, np.ndarray) and values.dtype == dtype:
        return np.ascontiguousarray(values)
    materialized = list(values)
    candidate = np.asarray(materialized)
    if candidate.dtype.kind in "iu":
        if candidate.size and (
            int(candidate.min()) < 0 or int(candidate.max()) > maximum
        ):
            raise ValueError(f"{label} out of range for dtype {dtype}")
        return np.ascontiguousarray(candidate, dtype=dtype)
    checked = []
    for value in materialized:
        try:
            integer = operator.index(value)
        except TypeError as error:
            raise ValueError(f"{label} must contain integers") from error
        if integer < 0 or integer > maximum:
            raise ValueError(f"{label} out of range for dtype {dtype}")
        checked.append(integer)
    return np.ascontiguousarray(checked, dtype=dtype)


def _check_scheme_consistency(known: str | None, minhash) -> str | None:
    scheme = getattr(minhash, "scheme", None)
    if scheme is None:
        return known
    if known is None:
        return scheme
    if scheme != known:
        raise ValueError(
            f"MinHash scheme {scheme!r} does not match scheme {known!r} "
            "of the MinHash previously given to this index"
        )
    return known


class MinHash:
    def __init__(
        self,
        num_perm: int = 128,
        seed: int = 1,
        gpu_mode: Literal["disable", "detect", "always"] = "disable",
        hashfunc: Callable | None = None,
        hashobj: object | None = None,
        hashvalues=None,
        permutations=None,
        scheme: Literal["affine32", "affine64", "legacy"] | None = None,
    ) -> None:
        if scheme is None:
            if hashvalues is not None or permutations is not None:
                raise ValueError(
                    "scheme must be specified explicitly when initializing from "
                    "existing hash values or permutations: pass the scheme of the "
                    "MinHash they came from, or scheme='legacy' for values created "
                    "by datasketch before 2.0.0."
                )
            scheme = "affine32"
        elif scheme not in _VALID_SCHEMES:
            raise ValueError(
                "scheme must be one of affine32, affine64, legacy, "
                f"got {scheme!r}"
            )
        if hashvalues is not None:
            num_perm = len(hashvalues)
        if num_perm < 1:
            raise ValueError("num_perm must be positive")
        if num_perm > _HASH_RANGE:
            raise ValueError(
                f"Cannot have more than {_HASH_RANGE} number of permutation functions"
            )
        if gpu_mode not in ("disable", "detect", "always"):
            raise ValueError("gpu_mode must be 'disable', 'detect', or 'always'")
        if gpu_mode == "always":
            raise RuntimeError("GPU mode is not available in mojo-datasketch")
        self.scheme = scheme
        self.seed = seed
        self.num_perm = num_perm
        self._gpu_mode = gpu_mode
        if hashfunc is None:
            hashfunc = sha1_hash64 if scheme == "affine64" else sha1_hash32
        if not callable(hashfunc):
            raise ValueError("The hashfunc must be a callable.")
        self.hashfunc = hashfunc
        if hashobj is not None:
            warnings.warn(
                "hashobj is deprecated, use hashfunc instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.hashvalues = (
            self._parse_values(hashvalues, "hash values")
            if hashvalues is not None
            else np.full(num_perm, _MAX_HASHES[scheme], dtype=_DTYPES[scheme])
        )
        self._hashvalues_ref = self.hashvalues
        self._hashvalues_addr = addr(self.hashvalues)
        self._hashvalues_nbytes = self.hashvalues.nbytes
        self._hashvalues_zero_copy = (
            self.hashvalues.ndim == 1
            and self.hashvalues.dtype == _DTYPES[scheme]
            and self.hashvalues.flags.c_contiguous
            and not self.hashvalues.flags.owndata
        )
        root = self.hashvalues
        while isinstance(root.base, np.ndarray):
            root = root.base
        self._hashvalues_root = root
        self.permutations = (
            self._parse_permutations(permutations)
            if permutations is not None
            else self._init_permutations(num_perm)
        )
        if len(self.hashvalues) != len(self.permutations[0]):
            raise ValueError("Numbers of hash values and permutations mismatch")

    def _parse_values(self, values, label: str) -> np.ndarray:
        dtype = _DTYPES[self.scheme]
        parsed = _unsigned_array(values, dtype, label)
        if parsed.ndim != 1:
            raise ValueError(f"{label} must be one-dimensional")
        return parsed

    def _parse_permutations(self, permutations) -> np.ndarray:
        dtype = _DTYPES[self.scheme]
        source = np.asarray(permutations, dtype=object)
        if source.ndim != 2 or source.shape != (2, self.num_perm):
            raise ValueError("permutations must have shape (2, num_perm)")
        parsed = _unsigned_array(source.ravel(), dtype, "permutations").reshape(
            source.shape
        )
        if parsed.ndim != 2 or parsed.shape != (2, self.num_perm):
            raise ValueError("permutations must have shape (2, num_perm)")
        if self.scheme != "legacy" and np.any(
            parsed[0] & dtype.type(1) == 0
        ):
            raise ValueError(
                f"all `a` permutation parameters must be odd for scheme {self.scheme!r}"
            )
        return parsed

    def _init_permutations(self, num_perm: int) -> np.ndarray:
        generator = np.random.RandomState(self.seed)
        if self.scheme == "legacy":
            return np.ascontiguousarray(
                np.array(
                    [
                        (
                            generator.randint(
                                1, _MERSENNE_PRIME, dtype=np.uint64
                            ),
                            generator.randint(
                                0, _MERSENNE_PRIME, dtype=np.uint64
                            ),
                        )
                        for _ in range(num_perm)
                    ],
                    dtype=np.uint64,
                ).T
            )
        width = 32 if self.scheme == "affine32" else 64
        dtype = _DTYPES[self.scheme].type
        a = (
            generator.randint(0, 1 << (width - 1), num_perm, dtype=dtype)
            * dtype(2)
            + dtype(1)
        )
        b = generator.randint(0, 1 << width, num_perm, dtype=dtype)
        return np.ascontiguousarray(np.array([a, b]))

    def update(self, b) -> None:
        self.update_batch((b,))

    def update_batch(self, b: Iterable) -> None:
        hashes = _unsigned_array(
            (self.hashfunc(value) for value in b),
            _DTYPES[self.scheme],
            f"hashfunc results for scheme {self.scheme!r}",
        )
        if not hashes.size:
            return
        values = require_buffer(
            self.hashvalues,
            dtype=_DTYPES[self.scheme],
            size=len(self),
            label="hashvalues",
            writable=True,
        )
        permutations = self.permutations
        if (
            not isinstance(permutations, np.ndarray)
            or permutations.shape != (2, len(self))
            or permutations.dtype != _DTYPES[self.scheme]
            or not permutations.flags.c_contiguous
        ):
            raise ValueError(
                "permutations must be a C-contiguous array with shape "
                "(2, num_perm) and the scheme dtype"
            )
        function = getattr(lib(), f"md_minhash_{self.scheme}")
        function(
            addr(hashes),
            hashes.size,
            addr(permutations[0]),
            addr(permutations[1]),
            addr(values),
            len(self),
        )

    def jaccard(self, other: MinHash) -> float:
        self._check_compatible(other, "compute Jaccard given")
        left = self._ffi_hashvalues()
        right = other._ffi_hashvalues()
        suffix = "32" if self.scheme == "affine32" else "64"
        return float(
            getattr(lib(), f"md_jaccard{suffix}")(
                addr(left), addr(right), len(self)
            )
        )

    def count(self) -> float:
        return float(len(self)) / np.sum(
            self.hashvalues / float(_MAX_HASHES[self.scheme])
        ) - 1.0

    def merge(self, other: MinHash) -> None:
        self._check_compatible(other, "merge")
        left = self._ffi_hashvalues(writable=True)
        right = other._ffi_hashvalues()
        suffix = "32" if self.scheme == "affine32" else "64"
        getattr(lib(), f"md_merge{suffix}")(
            addr(left), addr(right), len(self)
        )

    def _ffi_hashvalues(self, writable: bool = False) -> np.ndarray:
        return require_buffer(
            self.hashvalues,
            dtype=_DTYPES[self.scheme],
            size=self.num_perm,
            label="hashvalues",
            writable=writable,
        )

    def _check_compatible(self, other: MinHash, operation: str) -> None:
        if getattr(other, "scheme", None) != self.scheme:
            raise ValueError(
                f"Cannot {operation} MinHash with different permutation schemes"
            )
        if other.seed != self.seed:
            raise ValueError(f"Cannot {operation} MinHash with different seeds")
        if len(self) != len(other):
            raise ValueError(
                f"Cannot {operation} MinHash with different numbers of "
                "permutation functions"
            )

    def digest(self) -> np.ndarray:
        return copy.copy(self.hashvalues)

    def is_empty(self) -> bool:
        return not np.any(self.hashvalues != _MAX_HASHES[self.scheme])

    def clear(self) -> None:
        self.hashvalues.fill(_MAX_HASHES[self.scheme])

    def copy(self) -> MinHash:
        return MinHash(
            seed=self.seed,
            hashfunc=self.hashfunc,
            hashvalues=self.digest(),
            permutations=self.permutations,
            gpu_mode=self._gpu_mode,
            scheme=self.scheme,
        )

    def __len__(self) -> int:
        return len(self.hashvalues)

    def __eq__(self, other) -> bool:
        return (
            type(self) is type(other)
            and self.scheme == other.scheme
            and self.seed == other.seed
            and np.array_equal(self.hashvalues, other.hashvalues)
        )

    @classmethod
    def union(cls, *mhs: MinHash) -> MinHash:
        if len(mhs) < 2:
            raise ValueError("Cannot union less than 2 MinHash")
        first = mhs[0]
        if any(
            first.seed != item.seed
            or len(first) != len(item)
            or first.scheme != item.scheme
            for item in mhs
        ):
            raise ValueError(
                "The unioning MinHash must have the same seed, number of "
                "permutation functions and scheme"
            )
        result = first.copy()
        for item in mhs[1:]:
            result.merge(item)
        return result

    @classmethod
    def bulk(cls, b: Iterable, **minhash_kwargs) -> list[MinHash]:
        return list(cls.generator(b, **minhash_kwargs))

    @classmethod
    def generator(
        cls, b: Iterable, **minhash_kwargs
    ) -> Generator[MinHash, None, None]:
        template = cls(**minhash_kwargs)
        for values in b:
            item = template.copy()
            item.update_batch(values)
            yield item


def jaccard_many(query: MinHash, sketches: Iterable[MinHash]) -> np.ndarray:
    """Score many compatible sketches without a Python loop."""
    items = sketches if isinstance(sketches, (list, tuple)) else list(sketches)
    if not items:
        return np.empty(0, dtype=np.float64)
    scheme = query.scheme
    seed = query.seed
    count = len(query)
    dtype = _DTYPES[scheme]
    query_values = query._ffi_hashvalues()
    shared_root = None
    matrix_addr = 0
    next_addr = 0
    zero_copy = True
    for item in items:
        native = item.__class__ is MinHash
        item_scheme = item.scheme if native else getattr(item, "scheme", None)
        if item_scheme != scheme:
            raise ValueError(
                "Cannot compute Jaccard given MinHash with different "
                "permutation schemes"
            )
        if item.seed != seed:
            raise ValueError(
                "Cannot compute Jaccard given MinHash with different seeds"
            )
        values = item.hashvalues
        cached = native and values is item._hashvalues_ref
        if not cached and not isinstance(values, np.ndarray):
            raise ValueError("hashvalues must be a numpy.ndarray")
        if values.ndim != 1 or values.size != count:
            raise ValueError(
                "Cannot compute Jaccard given MinHash with different numbers "
                "of permutation functions"
            )
        if values.dtype != dtype:
            raise ValueError(
                f"hashvalues must be a one-dimensional array with dtype {dtype}"
            )
        if not values.flags.c_contiguous:
            raise ValueError("hashvalues must be C-contiguous")
        item_addr = item._hashvalues_addr if native else 0
        item_root = item._hashvalues_root if native else None
        item_nbytes = item._hashvalues_nbytes if native else 0
        if (
            not cached
            or not item._hashvalues_zero_copy
            or values.nbytes != item_nbytes
            or (
                shared_root is not None
                and (item_root is not shared_root or item_addr != next_addr)
            )
        ):
            zero_copy = False
        if shared_root is None:
            shared_root = item_root
            matrix_addr = item_addr
        next_addr = item_addr + item_nbytes
    if not zero_copy:
        matrix = np.ascontiguousarray(
            np.stack([item.hashvalues for item in items]), dtype=dtype
        )
        matrix_addr = addr(matrix)
    scores = np.empty(len(items), dtype=np.float64)
    suffix = "32" if scheme == "affine32" else "64"
    getattr(lib(), f"md_score{suffix}")(
        matrix_addr,
        addr(query_values),
        addr(scores),
        len(items),
        count,
    )
    return scores
