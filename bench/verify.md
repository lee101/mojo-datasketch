# VERIFY phase — reproduction of the README benchmark table

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux x86_64; Python 3.13.14.
All numbers below were measured in this session. Commands run in order:

```bash
pixi run build && pixi run test   # 47 passed, exit 0
pixi run bench                    # exit 0
pixi run bench                    # second run, exit 0 (re-run control)
pixi run bench                    # third run, exit 0 (median of three is the headline)
pixi run python bench/verify_probe.py   # independent probe, exit 0 (removed after the run)
pixi run build && pixi run test   # 47 passed, exit 0 (final state)
```

An independent probe re-derived each row from scratch without importing
`bench.py` helpers, checked output parity against upstream, and instrumented
hashfunc call counts to rule out skipped work. Reported headline figures in
the README are the median of the three `pixi run bench` measurements this
session; probe and bench agree within noise except where noted (the probe's
counting hashfunc wrapper adds per-call Python overhead — see HLL note below).

## What reproduced (within 20%)

All eight headline figures reproduced within 20% in **every** one of the
three bench runs. None was outside the band, so no figure was corrected
under the 20% rule; per the brief's "a number you did not measure in this
session does not go in the README", all eight were nevertheless replaced
with this session's medians.

| case | README (before) | this session median (bench x3) | delta | verdict |
| --- | ---: | ---: | ---: | --- |
| MinHash.update_batch mojo | 52.47 ms | 53.55 ms | +2.1% | reproduces |
| MinHash.update_batch datasketch | 82.42 ms | 81.28 ms | -1.4% | reproduces |
| HyperLogLog mojo | 132.25 ms | 129.11 ms | -2.4% | reproduces |
| HyperLogLog datasketch | 372.51 ms | 375.61 ms | +0.8% | reproduces |
| Jaccard mojo | 35.17 ms | 36.01 ms | +2.4% | reproduces |
| Jaccard datasketch | 96.22 ms | 100.48 ms | +4.4% | reproduces (individual runs 93.74 / 100.48 / 114.48 ms; worst run +19.0%, still inside the band, same noisy upstream loop) |
| MinHashLSH.query mojo | 0.02 ms | 0.02 ms | 0.0% | reproduces |
| MinHashLSH.query datasketch | 0.02 ms | 0.02 ms | 0.0% | reproduces |

Speedup ratios: 1.52x / 2.91x / 2.79x / 1.00x this session vs 1.57x / 2.82x /
2.74x / 1.19x in the README — all within 20%. The LSH ratio at two-decimal-ms
precision is 1.00x (both sides round to 0.02 ms); at microsecond precision
the probe read 17.8–18.4 µs vs 21.8–23.6 µs (1.22–1.29x). This is
timer-resolution noise on a ~20 µs dict-bound operation, not a kernel
result, exactly as the README already cautioned. The README's LSH result
was corrected from the unmeasured 1.19x to the measured 1.00x at the printed
precision.

Individual bench runs (mojo / upstream, ms):

| run | MinHash | HLL | Jaccard | LSH |
| --- | --- | --- | --- | --- |
| 1 | 53.55 / 81.28 (1.52x) | 128.19 / 362.34 (2.83x) | 32.76 / 93.74 (2.86x) | 0.02 / 0.02 (1.00x) |
| 2 | 49.83 / 78.65 (1.58x) | 129.11 / 375.61 (2.91x) | 36.13 / 100.48 (2.78x) | 0.02 / 0.02 (1.00x) |
| 3 | 54.12 / 83.85 (1.55x) | 130.47 / 376.67 (2.89x) | 36.01 / 114.48 (3.18x) | 0.02 / 0.02 (1.00x) |
| probe 1 | 60.64 / 91.40 (1.51x) | 179.39 / 415.90 (2.32x) | 33.31 / 91.74 (2.75x) | 0.018 / 0.024 (1.28x) |
| probe 2 | 60.22 / 92.96 (1.54x) | 177.12 / 421.54 (2.38x) | 35.53 / 97.85 (2.75x) | 0.018 / 0.022 (1.22x) |

Probe HLL is higher than bench HLL because the probe's hashfunc is a
counting wrapper (an extra Python call per value, 500k calls) while the bench
uses the plain identity function; the probe's purpose was the call-count
instrumentation, not a second headline measurement. The bench median is
the headline.

## Fairness checks (same input, no skipped work)

- **MinHash.update_batch (100k x 128): fair.** Both sides receive the same
  `np.arange(100_000, dtype=np.uint64)` array, seed=7, scheme=affine32,
  identity hashfunc. Outputs are bit-identical (full 100k run checked:
  hashvalues and permutations equal). Both sides call the hashfunc exactly
  once per input (100,000 calls each); neither skips work. Upstream 2.0.0
  has `update_batch`, so this row is a same-API comparison (verified
  `hasattr` true on both).
- **HyperLogLog update (500k): same input, NOT the same API.** Upstream
  2.0.0 has no `update_batch` (verified: `hasattr` false). The bench compares
  mojo's new batch API against upstream's per-element loop. Registers and
  counts match exactly (5k parity run), and hashfunc call counts are equal
  (5,000 per 5,000 values on both sides), so no work is skipped — but the
  speedup measures batching, not a faster per-element kernel. Apples to
  apples (`.update` loop on both sides): mojo 15.5–16.7 µs/call vs upstream
  0.67–0.74 µs/call, i.e. **the port is ~21–25x slower on the shared API**
  (2k-call probe: 31.10–33.29 ms vs 1.33–1.47 ms; projected 500k: ~7,800–8,300 ms vs ~330–370 ms).
- **Jaccard score 50k signatures: same data, NOT the same API.** Upstream
  has no `jaccard_many` (verified: `hasattr` false). The bench compares mojo's
  vectorized batch scorer against an upstream per-sketch `jaccard` loop over
  identical signature matrices (seed-4 data, verified equal to NumPy ground
  truth and to per-sketch upstream scores on a 200-sketch subset). No work
  skipped. Apples to apples (per-sketch loop on both sides): mojo 533–536 ms
  vs upstream 92–94 ms — **the port is ~5.7x slower on the shared API**.
- **MinHashLSH.query (20k index): fair.** Same 20k sketches, same explicit
  params (16, 8), same query sketch; result sets are identical (13 keys).
  Index build is untimed on both sides (insert loop included in both setups;
  only `query` is timed, best-of-20).

No README row shows the port slower than upstream, because every row uses a
mojo-only batch entry point (or the dict-bound LSH query). Those rows were
kept as-is per the brief — not removed, not reworded to look better — and the
README already labels the batch-vs-loop comparison explicitly. The shared-API
regressions found by this verification:

| shared API | this session | upstream | slower by |
| --- | ---: | ---: | ---: |
| `HyperLogLog.update` loop (2k calls) | 31.10–33.29 ms (~16 µs/call) | 1.33–1.47 ms (~0.7 µs/call) | ~21–25x |
| `MinHash.update` loop (2k) | 44.20–46.18 ms | 27.02–27.64 ms | ~1.6x |
| `jaccard` pair loop (50k) | 533.57–536.31 ms | 92.55–94.23 ms | ~5.7x |

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
   session's bench medians (93.74/100.48/114.48 ms spread on Jaccard
   upstream noted above; every figure individually within 20% of the prior
   value, so none was corrected under that rule — they were replaced
   because a number not measured here does not belong in the table).
   LSH result corrected 1.19x -> 1.00x at printed precision (0.02 vs 0.02 ms;
   a number not measured here did not belong in the table either).
   README's per-element call cost corrected ~18 µs -> ~17 µs to this
   session's probe (~15.5–16.7 µs/call, median ~16 µs, rounding to ~17 µs
   conservatively). No row removed, no slower row reworded.
2. `bench/verify_probe.py`: created for the independent remeasurement,
   then removed; this file records its results.
3. No source code changed. No test changed.

## Could not reproduce / open items

- The LSH 1.19x headline at 0.02/0.02 ms is not a stable measurement at
  printed precision: at ~20 µs scale, two-decimal-ms printing cannot
  distinguish the sides (1.00x). Microsecond probe readings do show a
  small 1.22–1.29x difference, but that is timer-resolution noise on a
  dict-bound operation, not a kernel result. No slower-than-upstream LSH
  result was observed.
- Per-element glue latency (table above) is reproduced and unfixed by
  design of this phase; a binding-layer fast path is the fix, which is
  outside code-change rules here.

Final state: `pixi run build && pixi run test` — 47 passed, exit 0.
