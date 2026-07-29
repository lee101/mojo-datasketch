"""Sorted-array MinHash LSH forest."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Hashable

import numpy as np

from .minhash import MinHash, _check_scheme_consistency


class MinHashLSHForest:
    def __init__(self, num_perm: int = 128, l: int = 8) -> None:
        if l <= 0 or num_perm <= 0:
            raise ValueError("num_perm and l must be positive")
        if l > num_perm:
            raise ValueError("l cannot be greater than num_perm")
        self.l = l
        self.k = int(num_perm / l)
        self.hashtables = [defaultdict(list) for _ in range(l)]
        self.hashranges = [(i * self.k, (i + 1) * self.k) for i in range(l)]
        self.keys: dict[Hashable, list[bytes]] = {}
        self.sorted_hashtables: list[list[bytes]] = [[] for _ in range(l)]
        self._minhash_scheme: str | None = None

    @staticmethod
    def _H(values) -> bytes:
        return bytes(values.byteswap().data)

    def add(self, key: Hashable, minhash: MinHash) -> None:
        if len(minhash) < self.k * self.l:
            raise ValueError("The num_perm of MinHash out of range")
        self._minhash_scheme = _check_scheme_consistency(
            self._minhash_scheme, minhash
        )
        if key in self.keys:
            raise ValueError("The given key has already been added")
        bands = [
            self._H(minhash.hashvalues[start:end])
            for start, end in self.hashranges
        ]
        self.keys[key] = bands
        for band, table in zip(bands, self.hashtables):
            table[band].append(key)

    def index(self) -> None:
        for index, table in enumerate(self.hashtables):
            self.sorted_hashtables[index] = sorted(table)

    def _query(self, minhash, r, b):
        if r > self.k or r <= 0 or b > self.l or b <= 0:
            raise ValueError("parameter outside range")
        prefixes = [
            self._H(minhash.hashvalues[start : start + r])
            for start, _ in self.hashranges
        ]
        size = len(prefixes[0])
        for sorted_table, prefix, table in zip(
            self.sorted_hashtables[:b],
            prefixes[:b],
            self.hashtables[:b],
        ):
            position = bisect_left(sorted_table, prefix)
            while (
                position < len(sorted_table)
                and sorted_table[position][:size] == prefix
            ):
                yield from table[sorted_table[position]]
                position += 1

    def query(self, minhash: MinHash, k: int) -> list[Hashable]:
        if k <= 0:
            raise ValueError("k must be positive")
        if len(minhash) < self.k * self.l:
            raise ValueError("The num_perm of MinHash out of range")
        _check_scheme_consistency(self._minhash_scheme, minhash)
        results = set()
        depth = self.k
        while depth > 0:
            for key in self._query(minhash, depth, self.l):
                results.add(key)
                if len(results) >= k:
                    return list(results)
            depth -= 1
        return list(results)

    def get_minhash_hashvalues(self, key: Hashable) -> np.ndarray:
        if key not in self.keys:
            raise KeyError(
                f"The provided key does not exist in the LSHForest: {key}"
            )
        segments = self.keys[key]
        segment_size = len(segments[0])
        bytes_per_value, remainder = divmod(segment_size, self.k)
        if remainder or bytes_per_value not in (4, 8):
            raise ValueError("Stored hash segment has an unsupported width")
        dtype = np.uint32 if bytes_per_value == 4 else np.uint64
        return np.concatenate(
            [np.frombuffer(item, dtype=dtype).byteswap() for item in segments]
        )

    def is_empty(self) -> bool:
        return any(not table for table in self.sorted_hashtables)

    def __contains__(self, key: Hashable) -> bool:
        return key in self.keys
