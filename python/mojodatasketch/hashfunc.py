"""Hash functions used by the sketches."""

import hashlib
import struct


def sha1_hash32(data) -> int:
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def sha1_hash64(data) -> int:
    return struct.unpack("<Q", hashlib.sha1(data).digest()[:8])[0]
