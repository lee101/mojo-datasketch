"""Compute kernels for MinHash and HyperLogLog."""

from std.sys.info import simd_width_of

comptime U8Ptr = Pointer[UInt8, AnyOrigin[mut=True]]
comptime I8Ptr = Pointer[Int8, AnyOrigin[mut=True]]
comptime U32Ptr = Pointer[UInt32, AnyOrigin[mut=True]]
comptime U64Ptr = Pointer[UInt64, AnyOrigin[mut=True]]
comptime F64Ptr = Pointer[Float64, AnyOrigin[mut=True]]
comptime TASKS = 64
comptime MERSENNE61: UInt64 = (UInt64(1) << 61) - 1
comptime MAX32: UInt64 = (UInt64(1) << 32) - 1


def u8p(addr: Int) -> U8Ptr:
    return U8Ptr(unsafe_from_address=addr)


def i8p(addr: Int) -> I8Ptr:
    return I8Ptr(unsafe_from_address=addr)


def u32p(addr: Int) -> U32Ptr:
    return U32Ptr(unsafe_from_address=addr)


def u64p(addr: Int) -> U64Ptr:
    return U64Ptr(unsafe_from_address=addr)


def f64p(addr: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=addr)


@always_inline
def fmix32(value_in: UInt32) -> UInt32:
    var value = value_in
    value ^= value >> 16
    value *= UInt32(0x85EBCA6B)
    value ^= value >> 13
    value *= UInt32(0xC2B2AE35)
    return value ^ (value >> 16)


@always_inline
def fmix64(value_in: UInt64) -> UInt64:
    var value = value_in
    value ^= value >> 33
    value *= UInt64(0xFF51AFD7ED558CCD)
    value ^= value >> 33
    value *= UInt64(0xC4CEB9FE1A85EC53)
    return value ^ (value >> 33)


@export("md_minhash_affine32")
def md_minhash_affine32(
    hashes_addr: Int,
    count: Int,
    a_addr: Int,
    b_addr: Int,
    values_addr: Int,
    num_perm: Int,
) abi("C"):
    var hashes = u32p(hashes_addr)
    var a = u32p(a_addr)
    var b = u32p(b_addr)
    var values = u32p(values_addr)

    def update_range(task: Int) {imm}:
        var begin = num_perm * task // TASKS
        var end = num_perm * (task + 1) // TASKS
        for permutation in range(begin, end):
            var av = a[unsafe_offset=permutation]
            var bv = b[unsafe_offset=permutation]
            var smallest = values[unsafe_offset=permutation]
            for item in range(count):
                var candidate = av * fmix32(hashes[unsafe_offset=item]) + bv
                if candidate < smallest:
                    smallest = candidate
            values[unsafe_offset=permutation] = smallest

    for task in range(TASKS):
        update_range(task)


@export("md_minhash_affine64")
def md_minhash_affine64(
    hashes_addr: Int,
    count: Int,
    a_addr: Int,
    b_addr: Int,
    values_addr: Int,
    num_perm: Int,
) abi("C"):
    var hashes = u64p(hashes_addr)
    var a = u64p(a_addr)
    var b = u64p(b_addr)
    var values = u64p(values_addr)

    def update_range(task: Int) {imm}:
        var begin = num_perm * task // TASKS
        var end = num_perm * (task + 1) // TASKS
        for permutation in range(begin, end):
            var av = a[unsafe_offset=permutation]
            var bv = b[unsafe_offset=permutation]
            var smallest = values[unsafe_offset=permutation]
            for item in range(count):
                var candidate = av * fmix64(hashes[unsafe_offset=item]) + bv
                if candidate < smallest:
                    smallest = candidate
            values[unsafe_offset=permutation] = smallest

    for task in range(TASKS):
        update_range(task)


@export("md_minhash_legacy")
def md_minhash_legacy(
    hashes_addr: Int,
    count: Int,
    a_addr: Int,
    b_addr: Int,
    values_addr: Int,
    num_perm: Int,
) abi("C"):
    var hashes = u64p(hashes_addr)
    var a = u64p(a_addr)
    var b = u64p(b_addr)
    var values = u64p(values_addr)

    def update_range(task: Int) {imm}:
        var begin = num_perm * task // TASKS
        var end = num_perm * (task + 1) // TASKS
        for permutation in range(begin, end):
            var av = a[unsafe_offset=permutation]
            var bv = b[unsafe_offset=permutation]
            var smallest = values[unsafe_offset=permutation]
            for item in range(count):
                var candidate = (
                    (av * hashes[unsafe_offset=item] + bv) % MERSENNE61
                ) & MAX32
                if candidate < smallest:
                    smallest = candidate
            values[unsafe_offset=permutation] = smallest

    for task in range(TASKS):
        update_range(task)


@export("md_jaccard32")
def md_jaccard32(
    left_addr: Int, right_addr: Int, count: Int
) abi("C") -> Float64:
    var left = u32p(left_addr)
    var right = u32p(right_addr)
    var equal = 0
    for i in range(count):
        if left[unsafe_offset=i] == right[unsafe_offset=i]:
            equal += 1
    return Float64(equal) / Float64(count)


@export("md_jaccard64")
def md_jaccard64(
    left_addr: Int, right_addr: Int, count: Int
) abi("C") -> Float64:
    var left = u64p(left_addr)
    var right = u64p(right_addr)
    var equal = 0
    for i in range(count):
        if left[unsafe_offset=i] == right[unsafe_offset=i]:
            equal += 1
    return Float64(equal) / Float64(count)


@export("md_merge32")
def md_merge32(left_addr: Int, right_addr: Int, count: Int) abi("C"):
    var left = u32p(left_addr)
    var right = u32p(right_addr)
    for i in range(count):
        if right[unsafe_offset=i] < left[unsafe_offset=i]:
            left[unsafe_offset=i] = right[unsafe_offset=i]


@export("md_merge64")
def md_merge64(left_addr: Int, right_addr: Int, count: Int) abi("C"):
    var left = u64p(left_addr)
    var right = u64p(right_addr)
    for i in range(count):
        if right[unsafe_offset=i] < left[unsafe_offset=i]:
            left[unsafe_offset=i] = right[unsafe_offset=i]


@always_inline
def rank32(bits_in: UInt32, max_rank: Int) -> Int:
    var bits = bits_in
    var length = 0
    while bits != 0:
        bits >>= 1
        length += 1
    return max_rank - length + 1


@always_inline
def rank64(bits_in: UInt64, max_rank: Int) -> Int:
    var bits = bits_in
    var length = 0
    while bits != 0:
        bits >>= 1
        length += 1
    return max_rank - length + 1


@export("md_hll_update32")
def md_hll_update32(
    hashes_addr: Int, count: Int, reg_addr: Int, precision: Int
) abi("C"):
    var hashes = u32p(hashes_addr)
    var reg = i8p(reg_addr)
    var mask = (UInt32(1) << UInt32(precision)) - 1
    var max_rank = 32 - precision
    for i in range(count):
        var value = hashes[unsafe_offset=i]
        var index = Int(value & mask)
        var rank = rank32(value >> UInt32(precision), max_rank)
        if rank > Int(reg[unsafe_offset=index]):
            reg[unsafe_offset=index] = Int8(rank)


@export("md_hll_update64")
def md_hll_update64(
    hashes_addr: Int, count: Int, reg_addr: Int, precision: Int
) abi("C"):
    var hashes = u64p(hashes_addr)
    var reg = i8p(reg_addr)
    var mask = (UInt64(1) << UInt64(precision)) - 1
    var max_rank = 64 - precision
    for i in range(count):
        var value = hashes[unsafe_offset=i]
        var index = Int(value & mask)
        var rank = rank64(value >> UInt64(precision), max_rank)
        if rank > Int(reg[unsafe_offset=index]):
            reg[unsafe_offset=index] = Int8(rank)


@export("md_hll_merge")
def md_hll_merge(left_addr: Int, right_addr: Int, count: Int) abi("C"):
    var left = i8p(left_addr)
    var right = i8p(right_addr)
    for i in range(count):
        if right[unsafe_offset=i] > left[unsafe_offset=i]:
            left[unsafe_offset=i] = right[unsafe_offset=i]


@export("md_score32")
def md_score32(
    matrix_addr: Int,
    query_addr: Int,
    scores_addr: Int,
    rows: Int,
    cols: Int,
) abi("C"):
    comptime W = simd_width_of[DType.uint32]()
    var matrix = u32p(matrix_addr)
    var query = u32p(query_addr)
    var scores = f64p(scores_addr)

    def score_range(task: Int) {imm}:
        var begin = rows * task // TASKS
        var end = rows * (task + 1) // TASKS
        for row in range(begin, end):
            var equal = 0
            var base = row * cols
            var vector_end = cols - cols % W
            for col in range(0, vector_end, W):
                var matches = matrix.unsafe_load[width=W, alignment=1](
                    base + col
                ).eq(query.unsafe_load[width=W, alignment=1](col))
                equal += Int(matches.cast[DType.uint32]().reduce_add())
            for col in range(vector_end, cols):
                if matrix[unsafe_offset=base + col] == query[unsafe_offset=col]:
                    equal += 1
            scores[unsafe_offset=row] = Float64(equal) / Float64(cols)

    for task in range(TASKS):
        score_range(task)


@export("md_score64")
def md_score64(
    matrix_addr: Int,
    query_addr: Int,
    scores_addr: Int,
    rows: Int,
    cols: Int,
) abi("C"):
    comptime W = simd_width_of[DType.uint64]()
    var matrix = u64p(matrix_addr)
    var query = u64p(query_addr)
    var scores = f64p(scores_addr)

    def score_range(task: Int) {imm}:
        var begin = rows * task // TASKS
        var end = rows * (task + 1) // TASKS
        for row in range(begin, end):
            var equal = 0
            var base = row * cols
            var vector_end = cols - cols % W
            for col in range(0, vector_end, W):
                var matches = matrix.unsafe_load[width=W, alignment=1](
                    base + col
                ).eq(query.unsafe_load[width=W, alignment=1](col))
                equal += Int(matches.cast[DType.uint64]().reduce_add())
            for col in range(vector_end, cols):
                if matrix[unsafe_offset=base + col] == query[unsafe_offset=col]:
                    equal += 1
            scores[unsafe_offset=row] = Float64(equal) / Float64(cols)

    for task in range(TASKS):
        score_range(task)
