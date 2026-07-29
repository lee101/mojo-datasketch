"""In-memory MinHash LSH compatible with datasketch."""

from __future__ import annotations

import pickle
from collections.abc import Hashable
from typing import Callable

from scipy.integrate import quad

from .minhash import MinHash, _check_scheme_consistency


def _false_positive_probability(threshold, b, r):
    return quad(lambda similarity: 1 - (1 - similarity**r) ** b, 0, threshold)[0]


def _false_negative_probability(threshold, b, r):
    return quad(lambda similarity: (1 - similarity**r) ** b, threshold, 1)[0]


def _optimal_param(threshold, num_perm, false_positive_weight, false_negative_weight):
    minimum = float("inf")
    optimum = (0, 0)
    for bands in range(1, num_perm + 1):
        for rows in range(1, int(num_perm / bands) + 1):
            error = (
                _false_positive_probability(threshold, bands, rows)
                * false_positive_weight
                + _false_negative_probability(threshold, bands, rows)
                * false_negative_weight
            )
            if error < minimum:
                minimum = error
                optimum = (bands, rows)
    return optimum


class MinHashLSH:
    def __init__(
        self,
        threshold: float = 0.9,
        num_perm: int = 128,
        weights: tuple[float, float] = (0.5, 0.5),
        params: tuple[int, int] | None = None,
        storage_config: dict | None = None,
        prepickle: bool | None = None,
        hashfunc: Callable[[bytes], bytes] | None = None,
    ) -> None:
        storage_config = storage_config or {"type": "dict"}
        if storage_config.get("type", "dict") != "dict":
            raise NotImplementedError("only in-memory dict storage is supported")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if num_perm < 2:
            raise ValueError("Too few permutation functions")
        if any(not 0.0 <= weight <= 1.0 for weight in weights):
            raise ValueError("Weight must be in [0.0, 1.0]")
        if sum(weights) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        self.h = num_perm
        if params is None:
            self.b, self.r = _optimal_param(threshold, num_perm, *weights)
        else:
            self.b, self.r = params
            if self.b * self.r > num_perm:
                raise ValueError(
                    f"The product of b and r in params is {self.b} * {self.r} "
                    f"= {self.b * self.r} -- it must be less than num_perm "
                    f"{num_perm}. Did you forget to specify num_perm?"
                )
        if self.b < 2:
            raise ValueError("The number of bands are too small (b < 2)")
        self.prepickle = bool(prepickle)
        self.hashfunc = hashfunc
        self.hashranges = [
            (index * self.r, (index + 1) * self.r) for index in range(self.b)
        ]
        self.hashtables: list[dict[bytes, set]] = [
            {} for _ in range(self.b)
        ]
        self.keys: dict[Hashable, list[bytes]] = {}
        self._minhash_scheme: str | None = None
        self._buffer_size = 50000

    @property
    def buffer_size(self) -> int:
        return self._buffer_size

    @buffer_size.setter
    def buffer_size(self, value: int) -> None:
        self._buffer_size = value

    def _H(self, values) -> bytes:
        raw = bytes(values.byteswap().data)
        return self.hashfunc(raw) if self.hashfunc else raw

    def insert(
        self, key: Hashable, minhash: MinHash, check_duplication: bool = True
    ) -> None:
        if len(minhash) != self.h:
            raise ValueError(
                f"Expecting minhash with length {self.h}, got {len(minhash)}"
            )
        self._minhash_scheme = _check_scheme_consistency(
            self._minhash_scheme, minhash
        )
        stored_key = pickle.dumps(key) if self.prepickle else key
        if check_duplication and stored_key in self.keys:
            raise ValueError("The given key already exists")
        bands = [
            self._H(minhash.hashvalues[start:end])
            for start, end in self.hashranges
        ]
        self.keys[stored_key] = bands
        for band, table in zip(bands, self.hashtables):
            table.setdefault(band, set()).add(stored_key)

    def query(self, minhash) -> list[Hashable]:
        if len(minhash) != self.h:
            raise ValueError(
                f"Expecting minhash with length {self.h}, got {len(minhash)}"
            )
        _check_scheme_consistency(self._minhash_scheme, minhash)
        candidates = set()
        for (start, end), table in zip(self.hashranges, self.hashtables):
            candidates.update(table.get(self._H(minhash.hashvalues[start:end]), ()))
        if self.prepickle:
            return [pickle.loads(key) for key in candidates]
        return list(candidates)

    def remove(self, key: Hashable) -> None:
        stored_key = pickle.dumps(key) if self.prepickle else key
        if stored_key not in self.keys:
            raise ValueError("The given key does not exist")
        for band, table in zip(self.keys.pop(stored_key), self.hashtables):
            table[band].remove(stored_key)
            if not table[band]:
                del table[band]

    def merge(self, other: MinHashLSH, check_overlap: bool = False):
        if (
            type(self) is not type(other)
            or self.h != other.h
            or self.b != other.b
            or self.r != other.r
        ):
            raise ValueError(
                "Cannot merge MinHashLSH with different initialization parameters."
            )
        if (
            self._minhash_scheme is not None
            and other._minhash_scheme is not None
            and self._minhash_scheme != other._minhash_scheme
        ):
            raise ValueError("Cannot merge indexes with different MinHash schemes")
        if check_overlap and set(self.keys).intersection(other.keys):
            raise ValueError("The keys are overlapping, duplicate key exists.")
        if self._minhash_scheme is None:
            self._minhash_scheme = other._minhash_scheme
        for key, bands in other.keys.items():
            self.keys[key] = list(bands)
            for band, table in zip(bands, self.hashtables):
                table.setdefault(band, set()).add(key)

    def insertion_session(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        return _InsertionSession(self)

    def deletion_session(self, buffer_size: int = 50000):
        self.buffer_size = buffer_size
        return _DeletionSession(self)

    def __contains__(self, key: Hashable) -> bool:
        stored_key = pickle.dumps(key) if self.prepickle else key
        return stored_key in self.keys

    def is_empty(self) -> bool:
        return any(not table for table in self.hashtables)

    def get_counts(self) -> list[dict[bytes, int]]:
        return [
            {band: len(keys) for band, keys in table.items()}
            for table in self.hashtables
        ]

    def get_subset_counts(self, *keys: Hashable) -> list[dict[bytes, int]]:
        counts: list[dict[bytes, int]] = [{} for _ in range(self.b)]
        for key in set(keys):
            stored = pickle.dumps(key) if self.prepickle else key
            for band, result in zip(self.keys.get(stored, ()), counts):
                result[band] = result.get(band, 0) + 1
        return counts


class _InsertionSession:
    def __init__(self, lsh: MinHashLSH):
        self.lsh = lsh

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def close(self) -> None:
        pass

    def insert(self, key, minhash, check_duplication=True) -> None:
        self.lsh.insert(key, minhash, check_duplication)


class _DeletionSession:
    def __init__(self, lsh: MinHashLSH):
        self.lsh = lsh

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def close(self) -> None:
        pass

    def remove(self, key) -> None:
        self.lsh.remove(key)
