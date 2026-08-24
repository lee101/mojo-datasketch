# mojo-datasketch

`mojo-datasketch` is a standalone Mojo port of the compute-heavy core of
[datasketch](https://github.com/ekzhu/datasketch): MinHash signatures,
HyperLogLog registers, and in-memory MinHash LSH indexes. The Python package is
named `mojodatasketch`, so it can be installed beside upstream `datasketch` for
parity testing. Covered classes retain upstream's names, constructor arguments,
attributes, and common methods.

This is an implementation, not a binding to datasketch. Hash functions and
Python object bookkeeping remain in Python; permutation reduction, register
updates, signature comparison, and merging execute in the compiled Mojo
library.

## Covered subset

| upstream area | implemented |
| --- | --- |
| `MinHash` | `affine32`, `affine64`, and `legacy` schemes; default and custom hashes; incremental/batch updates; Jaccard, cardinality, merge, union, bulk/generator, copy, clear, digest, and state validation |
| `HyperLogLog` | 32-bit default or custom hashing; incremental and batch updates; count corrections, merge, union, copy, clear, digest, byte serialization, and pickle |
| `HyperLogLogPlusPlus` | exact 64-bit register updates and the small-cardinality linear-counting path |
| `MinHashLSH` | optimized or explicit band parameters, in-memory insertion/query/removal, merge, sessions, pickled keys, compressed band keys, and bucket counts |
| `MinHashLSHForest` | add/index/top-k query, membership, emptiness, and signature reconstruction |
| batch scoring | `jaccard_many` scores compatible MinHash objects with one Mojo call |

The 47 tests compare directly against datasketch 2.0.0 or exercise FFI safety
checks. They assert exact
permutation arrays, exact hash values for all three current schemes, exact HLL
registers and estimates, optimized LSH parameters, bucket behavior, candidate
sets, forest results, merging, and serialization. They also cover SIMD tail
lengths, contiguous row-view scoring, invalid widths and strides, read-only
outputs, and integer values that would otherwise narrow silently.

## Not covered

WeightedMinHash, LeanMinHash, b-bit MinHash, LSH Ensemble, LSH Bloom, HNSW,
Redis/Cassandra storage, asynchronous APIs, and GPU execution are outside this
repository. `storage_config` currently accepts only the default in-memory
dictionary backend.

Upstream HLL++ uses an empirical bias table after linear counting. That table
is not bundled here, so `HyperLogLogPlusPlus.count()` is
upstream-identical only while its linear-counting path applies; at larger
cardinalities this port returns the uncorrected 64-bit HLL estimate. The
ordinary `HyperLogLog` estimator is fully covered, including all correction
ranges.

## Install

The repository pins the Mojo nightly and includes upstream datasketch only in
the Pixi development environment for parity tests:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-datasketch.so`. The Python package
reports a clear error if the library has not been built.

## Usage

This example runs with `pixi run python` after the build:

```python
from mojodatasketch import MinHash, MinHashLSH


def signature(words):
    sketch = MinHash(num_perm=128)
    sketch.update_batch(word.encode() for word in words)
    return sketch


documents = {
    "a": signature({"minhash", "similarity", "sets"}),
    "b": signature({"minhash", "similarity", "search"}),
    "c": signature({"unrelated", "document"}),
}

index = MinHashLSH(threshold=0.5, num_perm=128)
for key, sketch in documents.items():
    index.insert(key, sketch)

query = signature({"minhash", "similarity", "sets"})
print(sorted(index.query(query)))
print(query.jaccard(documents["a"]))
```

`HyperLogLog` also keeps the upstream API and adds a batch update that avoids
one FFI call per value:

```python
from mojodatasketch import HyperLogLog

hll = HyperLogLog(p=12)
hll.update_batch(str(value).encode() for value in range(100_000))
print(round(hll.count()))
```

## Benchmarks

Measured with `pixi run bench`, which holds the repository's machine-wide
benchmark lock. Times are best-of-three public-API calls (best-of-20 for the
short LSH query) on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux x86-64, Python
3.13.14, comparing with datasketch 2.0.0:

| case | mojo-datasketch | datasketch | result |
| --- | ---: | ---: | ---: |
| MinHash.update_batch (100k x 128) | 50.61 ms | 83.59 ms | 1.65x faster |
| HyperLogLog update (500k) | 125.16 ms | 376.69 ms | 3.01x faster |
| Jaccard score 50k signatures | 36.29 ms | 103.83 ms | 2.86x faster |
| MinHashLSH.query (20k index) | 0.02 ms | 0.03 ms | 1.21x faster |

The MinHash win comes from fusing affine permutation application with the
column minima. Upstream materializes a `batch_size × num_perm` intermediate;
the Mojo kernel scans hashes per permutation and retains only one minimum.
HLL batch update hashes values in Python once, then updates every register in
one FFI call instead of crossing Python method dispatch for each item. Integer
hash results are range-checked in bulk and converted with one contiguous NumPy
allocation rather than a second Python list.

`jaccard_many` detects signature row views that form one contiguous matrix and
passes that NumPy storage to Mojo without copying. Independent signature arrays
fall back to one contiguous stack. The scoring kernel compares a native SIMD
width at a time, handles the remainder with a scalar tail, and parallelizes
matrices with at least one million signature elements; smaller inputs stay
serial to avoid thread-pool overhead. LSH lookup itself is dictionary-bound rather than
arithmetic-bound, so its small timing difference should not be treated as a
substantial kernel speedup.

No GPU path was added or benchmarked. The targeted HLL register update and
signature scoring kernels have low arithmetic intensity, irregular writes or
host/device transfer overhead, so neither justifies occupying a shared GPU.

Run the benchmark on another machine with:

```bash
pixi run bench
```

## How it works

All kernels live in one Mojo compilation unit, `src/kernels.mojo`. Exports use
`@export("name")` with `abi("C")`; ctypes passes each NumPy buffer as an integer
address, and Mojo reconstructs it as an
`UnsafePointer[..., AnyOrigin[mut=True]]`. Mojo allocates no cross-language
memory. Each binding validates shape, dtype, contiguity, writability, length,
and non-null data before the synchronous call, and Python keeps every NumPy
owner alive until that call returns.

MinHash signatures and permutation arrays are C-contiguous `uint32` for
`affine32` and `uint64` for `affine64` and `legacy`. Batch reduction divides
permutations into disjoint worker ranges, so each thread writes independent
signature slots without synchronization. The legacy kernel deliberately
preserves upstream NumPy's wrapped uint64 multiply before its Mersenne-prime
reduction.

HLL registers are contiguous signed bytes, matching upstream's `int8` storage.
Hashing remains callable Python so custom hash functions retain upstream
semantics; the resulting contiguous uint32/uint64 hash vector crosses the FFI
once for register rank updates. LSH band tables and arbitrary Python keys stay
in Python dictionaries, while their source signatures are produced and
compared by Mojo.

## License

MIT
