# ANALYZE phase — where the time goes in mojo-datasketch

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30 GHz, Linux x86_64, Python 3.13.14. All numbers below were measured in this session under `flock /tmp/mojo-bench.lock` (`pixi run bench` for the headline table, ad-hoc timed probes for the splits). No code was changed in this phase; every byte/flop figure is derived from the measured input shapes.

"Flops" for these kernels are integer ops. Bytes = dtype width × element count of the input the kernel actually streams. `glue` = everything outside the Mojo kernel: `hashfunc` Python calls, `_unsigned_array` materialization, buffer validation, `addr()`/`lib()` lookups, per-sketch Python loops, and `np.stack` fallbacks.

## Headline bench (from `pixi run bench` this session)

| case | mojo-datasketch | datasketch | vs upstream |
| --- | ---: | ---: | --- |
| MinHash.update_batch (100k × 128) | 49.14 ms | 80.52 ms | 1.64x faster |
| HyperLogLog update (500k) | 132.43 ms | 363.51 ms | 2.74x faster |
| Jaccard score 50k signatures | 34.15 ms | 95.15 ms | 2.79x faster |
| MinHashLSH.query (20k index) | 0.02 ms | 0.02 ms | 1.28x faster |

## Ranked hotspots

Ranked by absolute time at the bench sizes, worst first. "SLOWER" marks rows where the port loses to upstream.

| # | function | input size | time | where | bytes touched | flops (ops) | flops/byte | bound | upstream (same shape) | next phase |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | --- |
| 1 | HLL `update_batch` (500k, p=14) | 500k hashes | 132.4 ms | 95.6% glue, 0.5% kernel, rest ctor | 2.0 MB in, 16 KB regs | ~5 M | ~2.5 | glue-bound | 363.5 ms (2.74x faster) | none |
| 2 | MinHash `update_batch` (100k × 128) | 100k hashes × 128 perms = 12.8 M pairs | 49.1 ms | 53% kernel, 44% glue | 51.2 MB re-read (hashes streamed per perm; 0.40 MB unique) | ~102 M | ~2.0 per streamed byte, ~255 per unique byte | split | 80.5 ms (1.64x faster) | parallel (+ hoist fmix) |
| 3 | `jaccard_many` (50k signatures × 128) | 50k × 128 = 6.4 M elems | 34.2 ms | 96% glue, 4% kernel | 25.6 MB in | ~13 M | ~0.5 | latency/glue-bound | 95.2 ms (2.79x faster) | parallel (kernel) / none (glue) |
| 4 | MinHash `update(v)` per element | n=1, 128 perms | 0.018 ms/call | ~100% glue | 4 B + 1 KB perms | ~1 k | — | glue-bound | 0.0098 ms/call (**SLOWER 1.83x**) | none |
| 5 | HLL `update(v)` per element | n=1, p=14 | 0.012 ms/call | ~100% glue | 4 B | ~10 | — | glue-bound | 0.0006 ms/call (**SLOWER 20x**) | none |
| 6 | MinHash `jaccard` pair | 128 elems | 0.0088 ms | 58% glue (raw kernel 0.0037 ms) | 1 KB | ~256 | ~0.5 | glue-bound | 0.0015 ms (**SLOWER 5.9x**) | none |
| 7 | MinHash `merge` / HLL `merge` | 128 / 16,384 elems | 0.008 / 0.013 ms | 59% / 29% glue (raw 0.0033 / 0.0095 ms) | 1 KB / 32 KB | ~256 / ~32 k | ~0.5 / ~1.0 | glue-bound | 0.0009 / 0.0016 ms (**SLOWER 8.9x / 8.3x**) | none |
| 8 | MinHash `copy` (+ `union`) | 128 perms | 0.054 ms (union×3: 0.084 ms) | 100% glue (re-parses permutations) | 4 KB moved | ~0 | — | glue-bound | 0.013 ms (**SLOWER 4.3x**) | none |
| 9 | MinHash `count`, HLL `count` | 128 / 16,384 regs | 0.004 / 0.39 ms | 100% numpy glue (same code as upstream) | 1 KB / 16 KB | ~1 k / ~30 k | ~1–2 | parity | 0.0036 / 0.34 ms (equal, slower by ≤12% on HLL which is numpy noise) | none |
| 10 | MinHashLSH `query`, `insert` | 20k index, 128 perms | 0.018 / 0.018 ms | 75% `_H` byteswap | 512 B sig | ~0 | ~0 | dict/latency-bound | 0.022 / 0.025 ms (faster) | none |
| 11 | LSH ctor w/ threshold | 128 perms | 18 ms | 100% `scipy.integrate.quad` | 0 | small | — | pure Python/scipy | same | none |
| 12 | LSHForest `query` | 2k index, 128 perms | 0.08–0.2 ms | 100% Python/bisect | 512 B | small | — | dict-bound | 0.20 ms (faster) | none |
| 13 | HLL `serialize`/`deserialize` | 16,385 B | 0.87 / 0.67 ms | 100% `struct` | 16 KB | 0 | 0 | copy-bound | 0.88 / 0.75 ms (equal) | none |

## Glue-vs-kernel detail (the numbers behind rows 1–3)

### Row 1 — HLL `update_batch`, 500k values (132.4 ms total)

- `_unsigned_array((self.hashfunc(v) for v in values), uint32)`: **126.7 ms (96%)**. The path is generator → `list(...)` of 500k Python ints → `np.asarray` → min/max range scan → `np.ascontiguousarray`. Three copies of the same 2 MB of data happen in glue alone.
- Raw `md_hll_update32` kernel: **0.77 ms (0.6%)** streaming 2 MB → ~2.6 GB/s effective, scattered `int8` register writes.
- Ctor (`np.zeros(16,384)` regs): ~0.02 ms, negligible.
- A plain NumPy vectorized HLL baseline (mask + `np.maximum.at`) does the whole update in **12.6 ms** — the glue is 10× slower than the algorithm, not the kernel.
- No per-element ctypes re-entry here: `update_batch` is one FFI call. The per-element path exists only via row 5.
- Verdict: **none** — this is a binding-layer problem (feed the kernel a contiguous hash vector without a Python-int list), not a kernel problem. Fpb ~2.5 with irregular scatter writes rules out both `parallel` (only 0.6% of time) and `gpu` (data-dependent rank scan + scatter to 16 KB of registers; <2–3 flops/byte before the Python hash, and the hash itself is outside the kernel).

### Row 2 — MinHash `update_batch`, 100k × 128 (49.1 ms total)

- Kernel `md_minhash_affine32`: **28.2 ms** (re-measured 25–27 ms across runs; same input → 100k hashes at perm=512 costs 106.6 ms, linear in pairs).
- Glue (`hashfunc` genexp + `_unsigned_array` + ctor): **~22.6 ms (46%)**.
- Kernel workspace: hashes (400 KB) stay in L2 — they are re-read 128× (51.2 MB streamed) — so the kernel is **cache/compute limited at ~1.9 GB/s effective**, not DRAM-bound. Flops/pair ≈ 8 (5 for `fmix32`, 2 affine, 1 min) → 102 M ops over 12.8 M pairs.
- **Redundant work found (candidate, not changed):** `fmix32(hashes[item])` depends only on `item`, yet is recomputed for every permutation — 12.7 M fmix calls where 100k would do. Hoisting it out of the permutation loop is an `unroll` fix, not an algorithmic change.
- The kernel also walks `for task in range(TASKS)` **serially**: the per-permutation ranges are disjoint and lock-free, so this is parallelizable with `max.algorithm.parallelize` behind a size threshold (per MOJO_NOTES.md §4), with `parallel` as the next phase.
- **GPU verdict: no at this size.** Arithmetic intensity is high on unique bytes (~255 f/unique-byte) which would favor a GPU at scale, but (a) the 400 KB input already lives in L2 and the real limit is single-threaded issue rate, and (b) at 100k×128 a launch + 400 KB H2D round trip buys little over a parallel CPU kernel; revisiting GPU would need n·perm ≥ ~100 M elements. Fpb ≈ 2.0 per streamed (L2) byte / 255 per unique byte.

### Row 3 — `jaccard_many`, 50k × 128 (34.2 ms total)

- Raw `md_score32` kernel (zero-copy matrix): **1.5–2.8 ms** — 25.6 MB streamed at ~9–17 GB/s, already SIMD (`simd_width_of`) with scalar tail.
- Python validation loop over 50k sketches (shape/dtype/contiguity/addr/root checks): **≈33 ms (96%)** — the kernel is idle 96% of the time.
- Fallback `np.stack` on non-contiguous signatures copies the whole 25.6 MB before the kernel even starts (measured 6.5 ms at just 5k signatures).
- Upstream’s per-sketch Python loop is slower (95 ms), but both spend >90% in Python, not Mojo.
- Next phase verdict: **parallel** for the kernel at 6.4 M elements (≥1 M-element threshold — the kernel splits rows into disjoint task ranges already; README claims threading there but the kernel loop is serial today); **none** for the glue (per-sketch metadata loop needs a vectorized/batched admission step — again a binding change, not a kernel phase). Fpb ~0.5 and a 25.6 MB contiguous matrix rule out **gpu** (transfer cost exceeds streaming cost; ~0.5 f/byte means the GPU would be a very expensive memcpy).

### Row 4–8 — the per-element and small-arg paths: the port's real regressions

- **HLL `update(v)` is 20x slower than upstream** (12.0 µs vs 0.6 µs/call). Each call re-enters the full batch path: generator → `_unsigned_array` (measured 3.9 µs even for n=1) → `lib()` lookup → three `addr()` int conversions + `require_buffer` validation → one ctypes call. A streaming caller making 500k single updates pays ~6 s where upstream pays ~0.3 s.
- Same shape for MinHash `update(v)` (1.83x slower), `jaccard` pair (5.9x slower, 0.0088 vs 0.0015 ms), `merge` (8.9x slower), HLL `merge` (8.3x slower). Raw kernel floors are 0.0033–0.0095 ms; the rest is validation + ctypes address plumbing.
- `MinHash.copy` is 4.3× slower (0.054 vs 0.013 ms) because it re-runs `_parse_permutations` (object-array materialization + odd-`a` check) instead of copying the array.
- All of these are **glue latency**, not kernel work. Verdict: **none** — no kernel phase (`unroll`/`simd`/`parallel`/`gpu`) touches them; they need a binding-layer fast path (cached `lib()` handle, skip re-validation of invariants already proven at construction), which is out of scope for the named kernel phases.

### Rows 9–13 — at parity, no action

- `count()` is identical NumPy in both ports (≤12% jitter = noise). LSH query/insert, LSHForest query, HLL serialize/deserialize are all dict/`struct`-bound and at or ahead of upstream. LSH construction with a threshold does ~16k `scipy.integrate.quad` calls (~18 ms) — pure Python, shared with upstream, **none** for the Mojo port.

## Consolidated per-call allocation / copy inventory

| call site | per-call allocations or copies | when they hurt |
| --- | --- | --- |
| `MinHash.update_batch` | `list(...)` of hashes (1 Python int each) + `np.asarray` + `np.ascontiguousarray` = 3 materializations; ~22.6 ms at 100k | every batch call, always |
| `HyperLogLog.update_batch` | same 3 copies; ~126.7 ms at 500k (96% of the function) | every batch call, always |
| `MinHash.update` / `HLL.update` | above, **plus** buffer re-validation + 3 `addr()` conversions per element | 20x slower than upstream HLL at n=1 |
| `jaccard_many` | `np.stack` full-matrix copy on the non-zero-copy path (25.6 MB at bench size) + 50k-iteration validation loop | ~33 ms glue at 50k sketches |
| `MinHash.copy` / `union` | permutation re-parse per copy | 4.3x slower copy |
| LSH `insert`/`query` | 16 `byteswap().data` copies per sketch (small) | ~75% of LSH query |

## Python loops that re-enter ctypes or rebuild per element

1. `MinHash.update` → `update_batch((b,))`: one element goes through the whole generator → list → asarray → 3× `addr()` → validation path. Measured 12–20 µs/call in glue vs ~2 µs of kernel work at 128 perms.
2. Same for `HyperLogLog.update` (20x slower than upstream; worst regression in the repo).
3. `jaccard_many`'s 50k-iteration metadata loop never touches Mojo until the end; it is 96% of that function's time.
