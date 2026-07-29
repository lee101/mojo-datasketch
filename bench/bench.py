"""Public-API benchmarks against datasketch."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import datasketch as upstream  # noqa: E402
import mojodatasketch as mojo  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def identity(value):
    return int(value)


def minhash_case(package, values):
    def run():
        sketch = package.MinHash(
            num_perm=128, seed=7, hashfunc=identity, scheme="affine32"
        )
        sketch.update_batch(values)
        return sketch

    return run


def hll_case(package, values):
    def run():
        sketch = package.HyperLogLog(p=14, hashfunc=identity)
        if package is mojo:
            sketch.update_batch(values)
        else:
            for value in values:
                sketch.update(value)
        return sketch

    return run


def lsh_case(package, sketches, query):
    index = package.MinHashLSH(num_perm=128, params=(16, 8))
    for key, sketch in enumerate(sketches):
        index.insert(key, sketch)
    return lambda: index.query(query)


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    values = np.arange(100_000, dtype=np.uint64)
    minhash_case(mojo, values[:10])()

    rows = []
    ours = timeit(minhash_case(mojo, values))
    reference = timeit(minhash_case(upstream, values))
    rows.append(("MinHash.update_batch (100k x 128)", ours, reference))

    hll_values = np.arange(500_000, dtype=np.uint64)
    ours = timeit(hll_case(mojo, hll_values))
    reference = timeit(hll_case(upstream, hll_values))
    rows.append(("HyperLogLog update (500k)", ours, reference))

    rng = np.random.default_rng(4)
    matrix = rng.integers(0, 2**32, size=(50_000, 128), dtype=np.uint32)
    query = mojo.MinHash(
        hashvalues=matrix[0], permutations=np.vstack([np.ones(128, np.uint32), np.zeros(128, np.uint32)]), scheme="affine32"
    )
    sketches = [
        mojo.MinHash(
            hashvalues=row,
            permutations=query.permutations,
            scheme="affine32",
        )
        for row in matrix
    ]
    reference_query = upstream.MinHash(
        hashvalues=matrix[0],
        permutations=query.permutations,
        scheme="affine32",
    )
    reference_sketches = [
        upstream.MinHash(
            hashvalues=row,
            permutations=query.permutations,
            scheme="affine32",
        )
        for row in matrix
    ]
    ours = timeit(lambda: mojo.jaccard_many(query, sketches))
    reference = timeit(
        lambda: np.array(
            [reference_query.jaccard(sketch) for sketch in reference_sketches]
        )
    )
    rows.append(("Jaccard score 50k signatures", ours, reference))

    documents = [
        mojo.MinHash(
            num_perm=128,
            hashfunc=identity,
            scheme="affine32",
        )
        for _ in range(20_000)
    ]
    for index, sketch in enumerate(documents):
        sketch.update_batch(range(index, index + 30))
    reference_documents = [
        upstream.MinHash(
            hashvalues=sketch.hashvalues,
            permutations=sketch.permutations,
            scheme="affine32",
        )
        for sketch in documents
    ]
    ours_fn = lsh_case(mojo, documents, documents[10_000])
    reference_fn = lsh_case(
        upstream, reference_documents, reference_documents[10_000]
    )
    ours = timeit(ours_fn, repeat=20)
    reference = timeit(reference_fn, repeat=20)
    rows.append(("MinHashLSH.query (20k index)", ours, reference))

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}; Python {platform.python_version()}")
    print()
    print("| case | mojo-datasketch | datasketch | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, reference in rows:
        ratio = reference / ours
        result = (
            f"{ratio:.2f}x faster"
            if ratio >= 1
            else f"{1 / ratio:.2f}x slower"
        )
        print(
            f"| {name} | {ours * 1e3:.2f} ms | "
            f"{reference * 1e3:.2f} ms | {result} |"
        )


if __name__ == "__main__":
    main()
