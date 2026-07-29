"""A Mojo-backed subset of datasketch."""

from .hashfunc import sha1_hash32, sha1_hash64
from .hyperloglog import HyperLogLog, HyperLogLogPlusPlus
from .lsh import MinHashLSH
from .lshforest import MinHashLSHForest
from .minhash import MinHash, jaccard_many

__version__ = "0.1.0"

__all__ = [
    "HyperLogLog",
    "HyperLogLogPlusPlus",
    "MinHash",
    "MinHashLSH",
    "MinHashLSHForest",
    "jaccard_many",
    "sha1_hash32",
    "sha1_hash64",
]
