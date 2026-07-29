"""Mojo-backed HyperLogLog sketches."""

from __future__ import annotations

import copy
import math
import struct
import warnings
from collections.abc import Iterable
from typing import Callable

import numpy as np

from ._lib import addr, lib, require_buffer
from .minhash import _unsigned_array
from .hashfunc import sha1_hash32, sha1_hash64


class HyperLogLog:
    _hash_range_bit = 32
    _hash_range_byte = 4

    def __init__(
        self,
        p: int = 8,
        reg: np.ndarray | None = None,
        hashfunc: Callable = sha1_hash32,
        hashobj: object | None = None,
    ):
        if reg is None:
            self._get_alpha(p)
            self.p = p
            self.m = 1 << p
            self.reg = np.zeros(self.m, dtype=np.int8)
        else:
            if not isinstance(reg, np.ndarray):
                raise ValueError("The imported register must be a numpy.ndarray.")
            self.m = reg.size
            self.p = self.m.bit_length() - 1
            if 1 << self.p != self.m:
                raise ValueError(
                    "The imported register has incorrect size. Expect a power of 2."
                )
            if reg.dtype != np.int8:
                raise ValueError("The imported register must have dtype int8.")
            if reg.ndim != 1 or np.any(reg < 0):
                raise ValueError(
                    "The imported register must be a one-dimensional, "
                    "non-negative int8 array."
                )
            self.reg = np.ascontiguousarray(reg)
        if not callable(hashfunc):
            raise ValueError("The hashfunc must be a callable.")
        if hashobj is not None:
            warnings.warn(
                "hashobj is deprecated, use hashfunc instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        self.hashfunc = hashfunc
        self.alpha = self._get_alpha(self.p)
        self.max_rank = self._hash_range_bit - self.p

    @staticmethod
    def _get_alpha(p: int) -> float:
        if not 4 <= p <= 16:
            raise ValueError(f"p={p} should be in range [4 : 16]")
        if p == 4:
            return 0.673
        if p == 5:
            return 0.697
        if p == 6:
            return 0.709
        return 0.7213 / (1.0 + 1.079 / (1 << p))

    def update(self, b) -> None:
        self.update_batch((b,))

    def update_batch(self, values: Iterable) -> None:
        dtype = np.dtype(
            np.uint32 if self._hash_range_bit == 32 else np.uint64
        )
        hashes = _unsigned_array(
            (self.hashfunc(value) for value in values),
            dtype,
            "hashfunc results",
        )
        if hashes.size:
            registers = self._ffi_registers(writable=True)
            getattr(lib(), f"md_hll_update{self._hash_range_bit}")(
                addr(hashes), hashes.size, addr(registers), self.p
            )

    def count(self) -> float:
        estimate = self.alpha * float(self.m**2) / np.sum(
            2.0 ** (-self.reg.astype(np.int16))
        )
        threshold = 2.5 * self.m
        if abs(estimate - threshold) / threshold < 0.15:
            warnings.warn(
                "Warning: estimate is close to error correction threshold. "
                "Output may not satisfy HyperLogLog accuracy guarantee.",
                stacklevel=2,
            )
        if estimate <= threshold:
            num_zero = self.m - np.count_nonzero(self.reg)
            return self._linearcounting(num_zero)
        if estimate <= (1.0 / 30.0) * (1 << 32):
            return float(estimate)
        return -(1 << 32) * math.log(1.0 - estimate / (1 << 32))

    def merge(self, other: HyperLogLog) -> None:
        if self.m != other.m or self.p != other.p:
            raise ValueError("Cannot merge HyperLogLog with different precisions.")
        left = self._ffi_registers(writable=True)
        right = other._ffi_registers()
        lib().md_hll_merge(addr(left), addr(right), self.m)

    def _ffi_registers(self, writable: bool = False) -> np.ndarray:
        return require_buffer(
            self.reg,
            dtype=np.int8,
            size=self.m,
            label="registers",
            writable=writable,
        )

    def digest(self) -> np.ndarray:
        return copy.copy(self.reg)

    def copy(self) -> HyperLogLog:
        return self.__class__(reg=self.digest(), hashfunc=self.hashfunc)

    def is_empty(self) -> bool:
        return not np.any(self.reg)

    def clear(self) -> None:
        self.reg.fill(0)

    def __len__(self) -> int:
        return len(self.reg)

    def __eq__(self, other) -> bool:
        return (
            type(self) is type(other)
            and self.p == other.p
            and self.m == other.m
            and np.array_equal(self.reg, other.reg)
        )

    def _linearcounting(self, num_zero: int) -> float:
        return self.m * np.log(self.m / float(num_zero))

    @classmethod
    def union(cls, *hyperloglogs: HyperLogLog) -> HyperLogLog:
        if len(hyperloglogs) < 2:
            raise ValueError("Cannot union less than 2 HyperLogLog sketches")
        if not all(item.m == hyperloglogs[0].m for item in hyperloglogs):
            raise ValueError(
                "Cannot union HyperLogLog sketches with different precisions"
            )
        result = hyperloglogs[0].copy()
        for item in hyperloglogs[1:]:
            result.merge(item)
        return result

    def bytesize(self) -> int:
        return 1 + self.m

    def serialize(self, buf) -> None:
        if len(buf) < self.bytesize():
            raise ValueError(
                "The buffer does not have enough space for holding this HyperLogLog."
            )
        struct.pack_into(f"B{self.m}B", buf, 0, self.p, *self.reg)

    @classmethod
    def deserialize(cls, buf):
        p = struct.unpack_from("B", memoryview(buf), 0)[0]
        result = cls(p)
        result.reg = np.array(
            struct.unpack_from(f"{result.m}B", memoryview(buf), 1),
            dtype=np.int8,
        )
        return result

    def __getstate__(self):
        buf = bytearray(self.bytesize())
        self.serialize(buf)
        return buf

    def __setstate__(self, buf):
        restored = self.deserialize(buf)
        self.__init__(p=restored.p, reg=restored.reg)


class HyperLogLogPlusPlus(HyperLogLog):
    _hash_range_bit = 64
    _hash_range_byte = 8
    _thresholds = [
        10,
        20,
        40,
        80,
        220,
        400,
        900,
        1800,
        3100,
        6500,
        11500,
        20000,
        50000,
    ]

    def __init__(
        self,
        p: int = 8,
        reg: np.ndarray | None = None,
        hashfunc: Callable = sha1_hash64,
        hashobj: object | None = None,
    ):
        super().__init__(p=p, reg=reg, hashfunc=hashfunc, hashobj=hashobj)

    def count(self) -> float:
        num_zero = self.m - np.count_nonzero(self.reg)
        if num_zero > 0:
            linear = self._linearcounting(num_zero)
            if linear <= self._thresholds[self.p - 4]:
                return linear
        return self.alpha * float(self.m**2) / np.sum(
            2.0 ** (-self.reg.astype(np.int16))
        )
