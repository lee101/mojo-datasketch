import pickle
import warnings

import numpy as np
import pytest

import datasketch as upstream
import mojodatasketch as mojo


@pytest.mark.parametrize("scheme", ["affine32", "affine64", "legacy"])
@pytest.mark.parametrize("num_perm", [1, 17, 128, 257])
def test_minhash_batch_state_matches_upstream(scheme, num_perm):
    values = [f"token-{index}".encode() for index in range(513)]
    reference = upstream.MinHash(num_perm=num_perm, seed=19, scheme=scheme)
    result = mojo.MinHash(num_perm=num_perm, seed=19, scheme=scheme)
    reference.update_batch(values)
    result.update_batch(values)
    assert result.hashvalues.dtype == reference.hashvalues.dtype
    assert np.array_equal(result.permutations, reference.permutations)
    assert np.array_equal(result.hashvalues, reference.hashvalues)
    assert result.count() == pytest.approx(reference.count(), rel=1e-14)


@pytest.mark.parametrize("scheme", ["affine32", "affine64", "legacy"])
def test_minhash_incremental_update_matches_upstream(scheme):
    reference = upstream.MinHash(num_perm=64, scheme=scheme)
    result = mojo.MinHash(num_perm=64, scheme=scheme)
    for value in [b"alpha", b"beta", b"gamma", b"alpha"]:
        reference.update(value)
        result.update(value)
    assert np.array_equal(result.hashvalues, reference.hashvalues)


def test_minhash_custom_hash_merge_jaccard_union_and_clear():
    hashfunc = lambda value: int(value)
    left_ref = upstream.MinHash(96, seed=3, hashfunc=hashfunc)
    right_ref = upstream.MinHash(96, seed=3, hashfunc=hashfunc)
    left = mojo.MinHash(96, seed=3, hashfunc=hashfunc)
    right = mojo.MinHash(96, seed=3, hashfunc=hashfunc)
    left_values = range(1_000)
    right_values = range(400, 1_400)
    left_ref.update_batch(left_values)
    right_ref.update_batch(right_values)
    left.update_batch(left_values)
    right.update_batch(right_values)
    assert left.jaccard(right) == left_ref.jaccard(right_ref)
    expected = upstream.MinHash.union(left_ref, right_ref)
    combined = mojo.MinHash.union(left, right)
    assert np.array_equal(combined.hashvalues, expected.hashvalues)
    left.merge(right)
    assert np.array_equal(left.hashvalues, expected.hashvalues)
    left.clear()
    assert left.is_empty()


def test_minhash_bulk_copy_and_constructor_validation():
    batches = [[b"a", b"b"], [b"b", b"c"], []]
    reference = upstream.MinHash.bulk(batches, num_perm=32, scheme="affine32")
    result = mojo.MinHash.bulk(batches, num_perm=32, scheme="affine32")
    assert all(
        np.array_equal(got.hashvalues, expected.hashvalues)
        for got, expected in zip(result, reference)
    )
    copied = result[0].copy()
    assert copied == result[0] and copied is not result[0]
    with pytest.raises(ValueError, match="scheme must be specified"):
        mojo.MinHash(hashvalues=result[0].hashvalues)
    with pytest.raises(ValueError, match="positive"):
        mojo.MinHash(num_perm=0)


def test_jaccard_many_matches_individual_upstream_scores():
    sketches = mojo.MinHash.bulk(
        [[str(index + offset).encode() for offset in range(30)] for index in range(300)],
        num_perm=128,
    )
    scores = mojo.jaccard_many(sketches[0], sketches)
    expected = np.array([sketches[0].jaccard(item) for item in sketches])
    assert np.array_equal(scores, expected)


@pytest.mark.parametrize("scheme,dtype", [("affine32", np.uint32), ("affine64", np.uint64)])
@pytest.mark.parametrize("rows,num_perm", [(9, 3), (9, 127), (800, 127)])
def test_jaccard_many_simd_tail_and_parallel_threshold(
    scheme, dtype, rows, num_perm
):
    rng = np.random.default_rng(23)
    matrix = rng.integers(
        0, np.iinfo(dtype).max, size=(rows, num_perm), dtype=dtype
    )
    template = mojo.MinHash(num_perm=num_perm, seed=11, scheme=scheme)
    sketches = [
        mojo.MinHash(
            hashvalues=row,
            permutations=template.permutations,
            seed=template.seed,
            scheme=scheme,
        )
        for row in matrix
    ]
    scores = mojo.jaccard_many(sketches[0], sketches)
    expected = np.count_nonzero(matrix == matrix[0], axis=1) / num_perm
    assert np.array_equal(scores, expected)


def test_jaccard_many_uses_contiguous_shared_rows_without_stacking(monkeypatch):
    matrix = np.arange(257 * 127, dtype=np.uint32).reshape(257, 127)
    template = mojo.MinHash(num_perm=127, seed=5)
    sketches = [
        mojo.MinHash(
            hashvalues=row,
            permutations=template.permutations,
            seed=template.seed,
            scheme=template.scheme,
        )
        for row in matrix
    ]

    def fail_stack(*args, **kwargs):
        raise AssertionError("contiguous shared rows should remain zero-copy")

    monkeypatch.setattr(np, "stack", fail_stack)
    scores = mojo.jaccard_many(sketches[0], sketches)
    expected = np.count_nonzero(matrix == matrix[0], axis=1) / matrix.shape[1]
    assert np.array_equal(scores, expected)


def test_ffi_rejects_reassigned_wrong_dtype_strides_and_readonly_outputs():
    left = mojo.MinHash(num_perm=16)
    right = mojo.MinHash(num_perm=16)
    left.hashvalues = np.zeros(16, dtype=np.uint64)
    with pytest.raises(ValueError, match="dtype uint32"):
        left.jaccard(right)

    left.hashvalues = np.zeros(32, dtype=np.uint32)[::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        left.merge(right)

    left.hashvalues = np.zeros(16, dtype=np.uint32)
    left.hashvalues.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        left.update(b"value")

    hll = mojo.HyperLogLog(p=4)
    hll.reg = np.zeros(32, dtype=np.int8)[::2]
    with pytest.raises(ValueError, match="C-contiguous"):
        hll.update(b"value")


@pytest.mark.parametrize("bad_hash", [-1, 2**32, 1.5])
def test_hash_results_cannot_silently_narrow(bad_hash):
    sketch = mojo.MinHash(hashfunc=lambda _: bad_hash, scheme="affine32")
    with pytest.raises(ValueError, match="hashfunc results"):
        sketch.update(b"value")
    hll = mojo.HyperLogLog(hashfunc=lambda _: bad_hash)
    with pytest.raises(ValueError, match="hashfunc results"):
        hll.update(b"value")


def test_constructor_arrays_cannot_silently_narrow():
    with pytest.raises(ValueError, match="hash values out of range"):
        mojo.MinHash(
            hashvalues=[-1],
            permutations=[[1], [0]],
            scheme="affine32",
        )
    with pytest.raises(ValueError, match="permutations out of range"):
        mojo.MinHash(
            hashvalues=[0],
            permutations=[[1], [2**32]],
            scheme="affine32",
        )
    with pytest.raises(ValueError, match="range"):
        mojo.HyperLogLog(p=17)
    with pytest.raises(ValueError, match="dtype int8"):
        mojo.HyperLogLog(reg=np.zeros(16, dtype=np.uint8))


@pytest.mark.parametrize("p", [4, 8, 12, 16])
def test_hyperloglog_registers_and_count_match_upstream(p):
    values = [f"value-{index}".encode() for index in range(8_000)]
    reference = upstream.HyperLogLog(p=p)
    result = mojo.HyperLogLog(p=p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for value in values:
            reference.update(value)
        result.update_batch(values)
        assert np.array_equal(result.reg, reference.reg)
        assert result.count() == pytest.approx(reference.count(), rel=1e-14)


def test_hyperloglog_merge_union_clear_and_serialization():
    first = mojo.HyperLogLog(p=10)
    second = mojo.HyperLogLog(p=10)
    first.update_batch(str(index).encode() for index in range(1_000))
    second.update_batch(str(index).encode() for index in range(500, 1_500))
    expected = np.maximum(first.reg, second.reg)
    united = mojo.HyperLogLog.union(first, second)
    assert np.array_equal(united.reg, expected)
    first.merge(second)
    assert np.array_equal(first.reg, expected)
    restored = mojo.HyperLogLog.deserialize(bytearray(first.__getstate__()))
    assert restored == first
    assert pickle.loads(pickle.dumps(first)) == first
    first.clear()
    assert first.is_empty()
    with pytest.raises(ValueError, match="different precisions"):
        first.merge(mojo.HyperLogLog(p=8))


@pytest.mark.parametrize("p,count", [(8, 100), (12, 2_000), (16, 20_000)])
def test_hyperloglogplusplus_small_range_parity(p, count):
    values = [f"value-{index}".encode() for index in range(count)]
    reference = upstream.HyperLogLogPlusPlus(p=p)
    result = mojo.HyperLogLogPlusPlus(p=p)
    for value in values:
        reference.update(value)
    result.update_batch(values)
    assert np.array_equal(result.reg, reference.reg)
    assert result.count() == pytest.approx(reference.count(), rel=1e-14)


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7, 0.9])
def test_lsh_optimized_parameters_match_upstream(threshold):
    reference = upstream.MinHashLSH(threshold=threshold, num_perm=128)
    result = mojo.MinHashLSH(threshold=threshold, num_perm=128)
    assert (result.b, result.r) == (reference.b, reference.r)


def _documents():
    return [
        {f"word-{index}" for index in range(start, start + 30)}
        for start in range(0, 200, 5)
    ]


def _sketches(package):
    result = []
    for document in _documents():
        sketch = package.MinHash(num_perm=128, seed=7)
        sketch.update_batch(value.encode() for value in document)
        result.append(sketch)
    return result


def test_lsh_insert_query_remove_and_counts_match_upstream():
    reference_sketches = _sketches(upstream)
    sketches = _sketches(mojo)
    reference = upstream.MinHashLSH(threshold=0.5, num_perm=128)
    result = mojo.MinHashLSH(threshold=0.5, num_perm=128)
    for key, (ref_sketch, sketch) in enumerate(zip(reference_sketches, sketches)):
        reference.insert(key, ref_sketch)
        result.insert(key, sketch)
    for key in (0, 7, 20, 39):
        assert set(result.query(sketches[key])) == set(
            reference.query(reference_sketches[key])
        )
    assert [
        sorted(counts.values()) for counts in result.get_counts()
    ] == [sorted(counts.values()) for counts in reference.get_counts()]
    subset = (0, 3, 5)
    assert [
        sorted(counts.values()) for counts in result.get_subset_counts(*subset)
    ] == [
        sorted(counts.values())
        for counts in reference.get_subset_counts(*subset)
    ]
    result.remove(3)
    assert 3 not in result
    with pytest.raises(ValueError, match="does not exist"):
        result.remove(3)


def test_lsh_sessions_pickle_keys_custom_bands_and_merge():
    sketches = _sketches(mojo)
    left = mojo.MinHashLSH(
        num_perm=128, params=(16, 8), prepickle=True, hashfunc=lambda data: data[:8]
    )
    right = mojo.MinHashLSH(
        num_perm=128, params=(16, 8), prepickle=True, hashfunc=lambda data: data[:8]
    )
    with left.insertion_session() as session:
        session.insert(("left", 0), sketches[0])
        session.insert(("left", 1), sketches[1])
    right.insert(("right", 0), sketches[2])
    left.merge(right)
    assert ("left", 0) in left and ("right", 0) in left
    assert ("left", 0) in left.query(sketches[0])
    with left.deletion_session() as session:
        session.remove(("left", 1))
    assert ("left", 1) not in left


def test_lsh_forest_query_and_reconstruction_match_upstream():
    reference_sketches = _sketches(upstream)
    sketches = _sketches(mojo)
    reference = upstream.MinHashLSHForest(num_perm=128, l=8)
    result = mojo.MinHashLSHForest(num_perm=128, l=8)
    for key, (ref_sketch, sketch) in enumerate(zip(reference_sketches, sketches)):
        reference.add(key, ref_sketch)
        result.add(key, sketch)
    reference.index()
    result.index()
    for key in (0, 10, 39):
        assert set(result.query(sketches[key], 10)) == set(
            reference.query(reference_sketches[key], 10)
        )
        assert np.array_equal(
            result.get_minhash_hashvalues(key),
            reference.get_minhash_hashvalues(key),
        )
    assert not result.is_empty()
    assert 5 in result
