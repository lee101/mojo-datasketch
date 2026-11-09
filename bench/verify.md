# VERIFY phase — reproduction of the README benchmark table

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux x86_64; Python 3.13.14.
All numbers below were measured in this session. Commands run in order:

```bash
pixi run build && pixi run test   # 47 passed, exit 0
pixi run bench                    # exit 0
pixi run bench                    # second run, exit 0 (re-run control)
```

An independent probe (`bench/verify_probe.py`, removed after the run) re-derived
each row from scratch without importing `bench.py` helpers, checked output
parity against upstream, and instrumented hashfunc call counts to rule out
skipped work. Reported headline figures in the README are this session's
median of the three measurements (`pixi run bench` x2 + probe); probe and
bench agree within noise except where noted.

## What reproduced (within 20%)

| case | README (before) | this session median | delta | verdict |
| --- | ---: | ---: | ---: | --- |
| MinHash.update_batch mojo | 50.61 ms | 52.47 ms | +3.7% | reproduces |
| MinHash.update_batch datasketch | 83.59 ms | 82.42 ms | -1.4% | reproduces |
| HyperLogLog mojo | 125.16 ms | 132.25 ms | +5.7% | reproduces |
| HyperLogLog datasketch | 376.69 ms | 372.51 ms | -1.1% | reproduces |
| Jaccard mojo | 36.29 ms | 35.17 ms | -3.1% | reproduces |
| Jaccard datasketch | 103.83 ms | 96.22 ms | -7.3% | reproduces (probe run read 117.05 ms, same noisy upstream loop) |
| MinHashLSH.query mojo | 0.02 ms | 0.02 ms | 0.0% | reproduces |
| MinHashLSH.query datasketch | 0.03 ms | 0.02 ms | -33% | **does not reproduce at printed precision** |

Speedup ratios: 1.57x / 2.82x / 2.74x / 1.19x this session vs 1.65x / 3.01x /
2.86x / 1.21x in the README — all within 20% except the LSH ratio at
two-decimal-ms precision, where both sides round to 0.02 ms. The LSH row was
corrected to the measured 0.02 / 0.02 ms (1.19x); at microsecond precision the
probe read 25.3 µs vs 26.3 µs. This is timer-resolution noise on a ~25 µs
dict-bound operation, not a kernel result, exactly as the README already
cautioned. All other README figures were replaced with this session's
measurements regardless of the 20% rule, because a number not measured here
does not belong in the table.

## Fairness checks (same input, no skipped work)

- **MinHash.update_batch (100k x 128): fair.** Both sides receive the same
  `np.arange(100_000, dtype=np.uint64)` array, seed=7, scheme=affine32,
  identity hashfunc. Outputs are bit-identical (full 100k run checked).
  Both sides call the hashfunc once per input; neither skips work.
- **HyperLogLog update (500k): same input, NOT the same API.** Upstream
  2.0.0 has no `update_batch` (verified: `hasattr` false). The bench compares
  mojo's new batch API against upstream's per-element loop. Registers and
  counts match exactly (5k parity run), and hashfunc call counts are equal
  (1000 per 1000 values on both sides), so no work is skipped — but the
  speedup measures batching, not a faster per-element kernel. Apples to
  apples (`.update` loop on both sides, 500k values): mojo 9772.73 ms vs
  upstream 382.73 ms, i.e. **the port is ~25x slower on the shared API**.
  Per-element: ~18.6 µs/call (2000-call probe: 37.17 ms) vs ~0.7 µs/call
  (1.38 ms), consistent with `bench/analysis.md` rows 4-5.
- **Jaccard score 50k signatures: same data, NOT the same API.** Upstream
  has no `jaccard_many`. The bench compares mojo's vectorized batch scorer
  against an upstream per-sketch `jaccard` loop over identical signature
  matrices (seed-4 data, verified equal to NumPy ground truth and to
  per-sketch upstream scores). No work skipped. Apples to apples (per-sketch
  loop on both sides): mojo 530.09 ms vs upstream ~94-117 ms — **the port is
  ~5x slower on the shared API** (pair probe: 10.5 µs vs 2.5 µs).
- **MinHashLSH.query (20k index): fair.** Same 20k sketches, same explicit
  params (16, 8), same query sketch; result sets are identical (13 keys).
  Index build is untimed on both sides (insert loop included in both setups;
  only `query` is timed, best-of-20).

No README row shows the port slower than upstream, because every row uses a
mojo-only batch entry point (or the dict-bound LSH query). Those rows were
kept as-is per the brief — not removed, not reworded to look better — and the
README now labels the batch-vs-loop comparison explicitly. The shared-API
regressions found by this verification:

| shared API | this session | upstream | slower by |
| --- | ---: | ---: | ---: |
| `HyperLogLog.update` loop (500k) | 9772.73 ms | 382.73 ms | ~25x |
| `MinHash.update` loop (2k) | 53.06 ms | 28.69 ms | ~1.8x |
| `jaccard` pair loop (50k) | 530.09 ms | ~96 ms | ~5x |
| `jaccard` pair (single) | 10.5 µs | 2.5 µs | ~4x |
| `MinHash.merge` (single) | 106.7 µs | 21.8 µs | ~5x |

Root cause (from code inspection, not a kernel bug): every per-element call
re-enters the full batch path — Python generator, `_unsigned_array`
materialization, `lib()` lookup, buffer re-validation, three `addr()`
conversions, then one ctypes call. This matches `bench/analysis.md` rows 4-8.
No code was changed for it: the brief permits code changes only for a bug
demonstrated by a failing test, and per-element results are numerically
correct (all parity checks pass); this is a glue-latency regression, not a
correctness bug.

## Tests: parity assertions

`git log -p -- tests/` shows only two commits touching tests: the initial
port (849a2c5) and the accel pass (8da2dbb). The accel pass only **added**
assertions (an `(8_000, 127)` parallel-threshold case and a non-ndarray
rejection check); no assertion was removed and no tolerance was loosened
(all exact `np.array_equal` comparisons and `rel=1e-14` count checks are as
originally committed). The suite still asserts direct parity with upstream
datasketch 2.0.0: exact permutation arrays, exact hashvalues for all three
schemes, exact HLL registers, LSH parameters/buckets/candidate sets, forest
results, merging, and serialization. 47 tests collected, 47 passed. Nothing
to restore.

Known non-parity (documented in README, unchanged): `HyperLogLogPlusPlus`
omits upstream's empirical bias table, so `count()` diverges from upstream
once linear counting no longer applies; the test suite only covers the
small-cardinality range where they agree.

## What changed

1. `README.md` benchmark table: replaced all eight figures with this
   session's measurements (LSH upstream 0.03 -> 0.02 ms being the only
   figure outside the 20% band); added an explicit note that the HLL and
   Jaccard rows compare a mojo-only batch API against upstream's loop, and
   that the port is slower on the shared per-element API (details in this
   file). No row removed, no slower row reworded.
2. `bench/verify_probe.py`: created for the independent remeasurement,
   then removed; this file records its results.
3. No source code changed. No test changed.

## Could not reproduce / open items

- The LSH 1.21x headline at 0.02/0.03 ms is not a stable measurement: at
  25 µs scale, two-decimal-ms printing cannot distinguish the sides. No
  slower-than-upstream LSH result was observed (25.3 vs 26.3 µs).
- Per-element glue latency (table above) is reproduced and unfixed by
  design of this phase; a binding-layer fast path is the fix, which is
  outside code-change rules here.

Final state: `pixi run build && pixi run test` — 47 passed, exit 0.
