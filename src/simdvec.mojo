"""SIMD forms of the elementwise `Vec` kernels.

`ported.mojo` keeps the upstream naming, argument order and branch structure of
every routine and delegates the array sweep to this module.  Each routine here
has two bodies computing the same expression, one `SW` elements wide and a
scalar tail; nothing here allocates and every routine reads its operands once
and writes its result once.

What is deliberately *not* delegated is anything whose result depends on the
order the entries are visited in: `VecMinMax_Seq`'s first-index-wins
tie-breaking and the infinity norm's first-NaN rule both stay scalar, because a
vector reduction cannot reproduce them.  The minimum and maximum *values*,
which no reordering can change, are found with the vector sweep and the index
is then taken with the upstream scan.

Elementwise `exp` and `log` are one libm call per element.  `std.math.exp`
vectorises on this target, so those two sweep four elements per call; if a
target ever reported that as a compile-time constant fallback, the loops would
degrade to four scalar calls rather than to a wrong answer.

`reciprocal_inplace` and `chop_inplace` stay scalar: both need a per-lane
select, which this dialect does not have, and a group-wide test cannot be exact
for a subnormal (which reciprocates to infinity) or for an entry at `tol`.


Part of mojo-petsc4py (MIT); the kernels it exports are transcribed from
PETSc 3.25.5 / petsc4py 3.25.5 (BSD-2-Clause), whose notices are in NOTICE.
"""

from std.math import abs, exp, log, sqrt

from simd import BLK, SW

comptime Ptr = Pointer[Float64, AnyOrigin[mut=True]]


# ------------------------------------------------------------------ maps


def copy(y: Ptr, x: Ptr, n: Int):
    var i = 0
    while i + SW <= n:
        y.unsafe_store(i, x.unsafe_load[width=SW](i))
        i += SW
    while i < n:
        y.unsafe_store(i, x.unsafe_load(i))
        i += 1


def map1(w: Ptr, x: Ptr, n: Int, alpha: Float64):
    """w[i] = x[i] * alpha."""
    var i = 0
    while i + SW <= n:
        w.unsafe_store(i, x.unsafe_load[width=SW](i) * alpha)
        i += SW
    while i < n:
        w.unsafe_store(i, x.unsafe_load(i) * alpha)
        i += 1


def shift(w: Ptr, x: Ptr, n: Int, alpha: Float64):
    """w[i] = x[i] + alpha."""
    var i = 0
    while i + SW <= n:
        w.unsafe_store(i, x.unsafe_load[width=SW](i) + alpha)
        i += SW
    while i < n:
        w.unsafe_store(i, x.unsafe_load(i) + alpha)
        i += 1


def axpy(y: Ptr, x: Ptr, n: Int, alpha: Float64):
    """y[i] += alpha * x[i].  `x` may alias `y`, as upstream allows."""
    var i = 0
    while i + SW <= n:
        y.unsafe_store(
            i, y.unsafe_load[width=SW](i) + alpha * x.unsafe_load[width=SW](i)
        )
        i += SW
    while i < n:
        y.unsafe_store(i, y.unsafe_load(i) + alpha * x.unsafe_load(i))
        i += 1


def aypx(y: Ptr, x: Ptr, n: Int, alpha: Float64):
    """y[i] = x[i] + alpha * y[i]."""
    var i = 0
    while i + SW <= n:
        y.unsafe_store(
            i, x.unsafe_load[width=SW](i) + alpha * y.unsafe_load[width=SW](i)
        )
        i += SW
    while i < n:
        y.unsafe_store(i, x.unsafe_load(i) + alpha * y.unsafe_load(i))
        i += 1


def axpby(y: Ptr, x: Ptr, n: Int, alpha: Float64, beta: Float64):
    """y[i] = alpha * x[i] + beta * y[i]."""
    var i = 0
    while i + SW <= n:
        y.unsafe_store(
            i, alpha * x.unsafe_load[width=SW](i) + beta * y.unsafe_load[width=SW](i)
        )
        i += SW
    while i < n:
        y.unsafe_store(i, alpha * x.unsafe_load(i) + beta * y.unsafe_load(i))
        i += 1


def axpbypcz(z: Ptr, x: Ptr, y: Ptr, n: Int, alpha: Float64, beta: Float64,
             gamma: Float64):
    """z[i] = alpha*x[i] + beta*y[i] + gamma*z[i], one pass over four arrays."""
    var i = 0
    while i + SW <= n:
        z.unsafe_store(
            i,
            alpha * x.unsafe_load[width=SW](i)
            + beta * y.unsafe_load[width=SW](i)
            + gamma * z.unsafe_load[width=SW](i),
        )
        i += SW
    while i < n:
        z.unsafe_store(
            i,
            alpha * x.unsafe_load(i) + beta * y.unsafe_load(i) + gamma * z.unsafe_load(i),
        )
        i += 1


def waxpy(w: Ptr, x: Ptr, y: Ptr, n: Int, alpha: Float64):
    """w[i] = alpha * x[i] + y[i].  `w` must not alias `y`, as upstream says."""
    var i = 0
    while i + SW <= n:
        w.unsafe_store(
            i, alpha * x.unsafe_load[width=SW](i) + y.unsafe_load[width=SW](i)
        )
        i += SW
    while i < n:
        w.unsafe_store(i, alpha * x.unsafe_load(i) + y.unsafe_load(i))
        i += 1


def pointwise_mult(w: Ptr, x: Ptr, y: Ptr, n: Int):
    var i = 0
    while i + SW <= n:
        w.unsafe_store(i, x.unsafe_load[width=SW](i) * y.unsafe_load[width=SW](i))
        i += SW
    while i < n:
        w.unsafe_store(i, x.unsafe_load(i) * y.unsafe_load(i))
        i += 1


def pointwise_divide(w: Ptr, x: Ptr, y: Ptr, n: Int):
    var i = 0
    while i + SW <= n:
        w.unsafe_store(i, x.unsafe_load[width=SW](i) / y.unsafe_load[width=SW](i))
        i += SW
    while i < n:
        w.unsafe_store(i, x.unsafe_load(i) / y.unsafe_load(i))
        i += 1


def pointwise_min(w: Ptr, x: Ptr, y: Ptr, n: Int):
    var i = 0
    while i + SW <= n:
        w.unsafe_store(
            i, min(x.unsafe_load[width=SW](i), y.unsafe_load[width=SW](i))
        )
        i += SW
    while i < n:
        w.unsafe_store(i, min(x.unsafe_load(i), y.unsafe_load(i)))
        i += 1


def pointwise_max(w: Ptr, x: Ptr, y: Ptr, n: Int):
    var i = 0
    while i + SW <= n:
        w.unsafe_store(
            i, max(x.unsafe_load[width=SW](i), y.unsafe_load[width=SW](i))
        )
        i += SW
    while i < n:
        w.unsafe_store(i, max(x.unsafe_load(i), y.unsafe_load(i)))
        i += 1


def pointwise_maxabs(w: Ptr, x: Ptr, y: Ptr, n: Int):
    var i = 0
    while i + SW <= n:
        w.unsafe_store(
            i, max(abs(x.unsafe_load[width=SW](i)), abs(y.unsafe_load[width=SW](i)))
        )
        i += SW
    while i < n:
        w.unsafe_store(i, max(abs(x.unsafe_load(i)), abs(y.unsafe_load(i))))
        i += 1


def abs_inplace(x: Ptr, n: Int):
    var i = 0
    while i + SW <= n:
        x.unsafe_store(i, abs(x.unsafe_load[width=SW](i)))
        i += SW
    while i < n:
        x.unsafe_store(i, abs(x.unsafe_load(i)))
        i += 1


def sqrtabs_inplace(x: Ptr, n: Int):
    var i = 0
    while i + SW <= n:
        x.unsafe_store(i, sqrt(abs(x.unsafe_load[width=SW](i))))
        i += SW
    while i < n:
        x.unsafe_store(i, sqrt(abs(x.unsafe_load(i))))
        i += 1


def conjugate_inplace(x: Ptr, n: Int):
    """PetscConj is the identity for a real build.  The loop survives because
    upstream keeps it and that is where a complex build would differ."""
    copy(x, x, n)


def zero(x: Ptr, n: Int):
    var i = 0
    while i + SW <= n:
        x.unsafe_store(i, SIMD[DType.float64, SW](0.0))
        i += SW
    while i < n:
        x.unsafe_store(i, 0.0)
        i += 1


def set_const(x: Ptr, n: Int, alpha: Float64):
    var i = 0
    while i + SW <= n:
        x.unsafe_store(i, SIMD[DType.float64, SW](alpha))
        i += SW
    while i < n:
        x.unsafe_store(i, alpha)
        i += 1


def reciprocal_inplace(x: Ptr, n: Int):
    """x <- 1/x with upstream's exact-zero guard: 0 maps to 0, not inf.

    The guard is an exact comparison, not a tolerance: PETSc reciprocates
    1e-20 to 1e20, so an epsilon test would diverge.  Only a zero is guarded,
    so a subnormal still reciprocates to infinity; the value cannot be selected
    per lane (no select in this dialect) and a group-wide test cannot be exact
    for every such entry, so the sweep stays scalar and per-entry exact.
    """
    var i = 0
    while i < n:
        var xi = x.unsafe_load(i)
        x.unsafe_store(i, 0.0 if xi == 0.0 else 1.0 / xi)
        i += 1


def chop_inplace(x: Ptr, n: Int, tol: Float64):
    """Zero the entries whose magnitude is strictly below `tol`.

    Same reason as `reciprocal_inplace`: the body is a compare and a
    conditional store, and this dialect has no per-lane select to vectorise it
    with; an entry at or above `tol` must be left untouched, not rewritten.
    """
    var i = 0
    while i < n:
        if abs(x.unsafe_load(i)) < tol:
            x.unsafe_store(i, 0.0)
        i += 1


def exp_inplace(x: Ptr, n: Int):
    var i = 0
    while i + SW <= n:
        x.unsafe_store(i, exp(x.unsafe_load[width=SW](i)))
        i += SW
    while i < n:
        x.unsafe_store(i, exp(x.unsafe_load(i)))
        i += 1


def log_inplace(x: Ptr, n: Int):
    var i = 0
    while i + SW <= n:
        x.unsafe_store(i, log(x.unsafe_load[width=SW](i)))
        i += SW
    while i < n:
        x.unsafe_store(i, log(x.unsafe_load(i)))
        i += 1


def normalize_inplace(x: Ptr, n: Int, z: Float64):
    """x <- x / z; the caller has already established `z > 0`."""
    var i = 0
    while i + SW <= n:
        x.unsafe_store(i, x.unsafe_load[width=SW](i) / z)
        i += SW
    while i < n:
        x.unsafe_store(i, x.unsafe_load(i) / z)
        i += 1


def swap_inplace(x: Ptr, y: Ptr, n: Int):
    var i = 0
    while i + SW <= n:
        var vx = x.unsafe_load[width=SW](i)
        var vy = y.unsafe_load[width=SW](i)
        x.unsafe_store(i, vy)
        y.unsafe_store(i, vx)
        i += SW
    while i < n:
        var t = x.unsafe_load(i)
        x.unsafe_store(i, y.unsafe_load(i))
        y.unsafe_store(i, t)
        i += 1


# ------------------------------------------------------------ reductions
#
# A reduction reads its operands once and folds them into four SIMD
# accumulators, one `BLK`-element block at a time.  Each accumulator is a
# separate FMA dependency chain, so the loads that follow the one just issued
# issue immediately instead of waiting out the FMA latency.  Operator precedence
# is unchanged -- the same terms are multiplied and subtracted, only the order
# of summation differs -- so a caller still gets a plain sum over the array.


def dot(x: Ptr, y: Ptr, n: Int) -> Float64:
    var zero = SIMD[DType.float64, SW](0.0)
    var a0 = zero
    var a1 = zero
    var a2 = zero
    var a3 = zero
    var i = 0
    while i + BLK <= n:
        a0 += x.unsafe_load[width=SW](i) * y.unsafe_load[width=SW](i)
        a1 += x.unsafe_load[width=SW](i + SW) * y.unsafe_load[width=SW](i + SW)
        a2 += x.unsafe_load[width=SW](i + 2 * SW) * y.unsafe_load[width=SW](i + 2 * SW)
        a3 += x.unsafe_load[width=SW](i + 3 * SW) * y.unsafe_load[width=SW](i + 3 * SW)
        i += BLK
    var acc = a0 + a1 + a2 + a3
    while i + SW <= n:
        acc += x.unsafe_load[width=SW](i) * y.unsafe_load[width=SW](i)
        i += SW
    var s = 0.0
    while i < n:
        s += x.unsafe_load(i) * y.unsafe_load(i)
        i += 1
    return s + acc.reduce_add()


def dot_norm2(x: Ptr, y: Ptr, n: Int) -> Tuple[Float64, Float64]:
    """(x.y, x.x) in a single pass, so x is read once instead of twice."""
    var zero = SIMD[DType.float64, SW](0.0)
    var d0 = zero
    var d1 = zero
    var d2 = zero
    var d3 = zero
    var q0 = zero
    var q1 = zero
    var q2 = zero
    var q3 = zero
    var i = 0
    while i + BLK <= n:
        var v0 = x.unsafe_load[width=SW](i)
        var v1 = x.unsafe_load[width=SW](i + SW)
        var v2 = x.unsafe_load[width=SW](i + 2 * SW)
        var v3 = x.unsafe_load[width=SW](i + 3 * SW)
        d0 += v0 * y.unsafe_load[width=SW](i)
        d1 += v1 * y.unsafe_load[width=SW](i + SW)
        d2 += v2 * y.unsafe_load[width=SW](i + 2 * SW)
        d3 += v3 * y.unsafe_load[width=SW](i + 3 * SW)
        q0 += v0 * v0
        q1 += v1 * v1
        q2 += v2 * v2
        q3 += v3 * v3
        i += BLK
    var dacc = d0 + d1 + d2 + d3
    var nacc = q0 + q1 + q2 + q3
    while i + SW <= n:
        var vx = x.unsafe_load[width=SW](i)
        dacc += vx * y.unsafe_load[width=SW](i)
        nacc += vx * vx
        i += SW
    var d = 0.0
    var q = 0.0
    while i < n:
        var xi = x.unsafe_load(i)
        d += xi * y.unsafe_load(i)
        q += xi * xi
        i += 1
    return (d + dacc.reduce_add(), q + nacc.reduce_add())


def asum(x: Ptr, n: Int) -> Float64:
    var zero = SIMD[DType.float64, SW](0.0)
    var a0 = zero
    var a1 = zero
    var a2 = zero
    var a3 = zero
    var i = 0
    while i + BLK <= n:
        a0 += abs(x.unsafe_load[width=SW](i))
        a1 += abs(x.unsafe_load[width=SW](i + SW))
        a2 += abs(x.unsafe_load[width=SW](i + 2 * SW))
        a3 += abs(x.unsafe_load[width=SW](i + 3 * SW))
        i += BLK
    var acc = a0 + a1 + a2 + a3
    while i + SW <= n:
        acc += abs(x.unsafe_load[width=SW](i))
        i += SW
    var s = 0.0
    while i < n:
        s += abs(x.unsafe_load(i))
        i += 1
    return s + acc.reduce_add()


def norm2(x: Ptr, n: Int) -> Float64:
    var zero = SIMD[DType.float64, SW](0.0)
    var a0 = zero
    var a1 = zero
    var a2 = zero
    var a3 = zero
    var i = 0
    while i + BLK <= n:
        var v0 = x.unsafe_load[width=SW](i)
        var v1 = x.unsafe_load[width=SW](i + SW)
        var v2 = x.unsafe_load[width=SW](i + 2 * SW)
        var v3 = x.unsafe_load[width=SW](i + 3 * SW)
        a0 += v0 * v0
        a1 += v1 * v1
        a2 += v2 * v2
        a3 += v3 * v3
        i += BLK
    var acc = a0 + a1 + a2 + a3
    while i + SW <= n:
        var vx = x.unsafe_load[width=SW](i)
        acc += vx * vx
        i += SW
    var s = 0.0
    while i < n:
        s += x.unsafe_load(i) * x.unsafe_load(i)
        i += 1
    return s + acc.reduce_add()


def total(x: Ptr, n: Int) -> Float64:
    var zero = SIMD[DType.float64, SW](0.0)
    var a0 = zero
    var a1 = zero
    var a2 = zero
    var a3 = zero
    var i = 0
    while i + BLK <= n:
        a0 += x.unsafe_load[width=SW](i)
        a1 += x.unsafe_load[width=SW](i + SW)
        a2 += x.unsafe_load[width=SW](i + 2 * SW)
        a3 += x.unsafe_load[width=SW](i + 3 * SW)
        i += BLK
    var acc = a0 + a1 + a2 + a3
    while i + SW <= n:
        acc += x.unsafe_load[width=SW](i)
        i += SW
    var s = 0.0
    while i < n:
        s += x.unsafe_load(i)
        i += 1
    return s + acc.reduce_add()


def norm_infinity(x: Ptr, n: Int) -> Float64:
    """max |x[i]|, keeping upstream's rule that the first NaN is adopted and
    ends the scan."""
    var m = 0.0
    var i = 0
    while i < n:
        var tmp = abs(x.unsafe_load(i))
        if tmp > m or tmp != tmp:
            m = tmp
            if tmp != tmp:
                break
        i += 1
    return m


def max_pointwise_divide(x: Ptr, y: Ptr, n: Int) -> Float64:
    """max |x_i / y_i|, with upstream's substitution of x_i where y_i is 0."""
    var m = 0.0
    var i = 0
    while i < n:
        var yi = y.unsafe_load(i)
        var v = abs(x.unsafe_load(i) if yi == 0.0 else x.unsafe_load(i) / yi)
        m = v if v > m else m
        i += 1
    return m


def maxpy(y: Ptr, alpha: Ptr, n: Int, nv: Int, base: Int):
    """y += sum_k alpha[k] * VV(k) over a flat basis block of length-n vectors."""
    var k = 0
    while k < nv:
        var xk = Ptr(unsafe_from_address=base + k * n * 8)
        var a = alpha.unsafe_load(k)
        var i = 0
        while i + SW <= n:
            y.unsafe_store(
                i, y.unsafe_load[width=SW](i) + a * xk.unsafe_load[width=SW](i)
            )
            i += SW
        while i < n:
            y.unsafe_store(i, y.unsafe_load(i) + a * xk.unsafe_load(i))
            i += 1
        k += 1


def maxpy_from_zero(y: Ptr, alpha: Ptr, n: Int, nv: Int, base: Int):
    """`maxpy` with the old `y` dropped, i.e. VecMAXPBY with beta = 0.

    The first term is assigned rather than accumulated into a zeroed buffer, so
    the destination is written once and never read back.
    """
    var k = 0
    while k < nv:
        var xk = Ptr(unsafe_from_address=base + k * n * 8)
        var a = alpha.unsafe_load(k)
        if k == 0:
            var i = 0
            while i + SW <= n:
                y.unsafe_store(i, a * xk.unsafe_load[width=SW](i))
                i += SW
            while i < n:
                y.unsafe_store(i, a * xk.unsafe_load(i))
                i += 1
        else:
            var i = 0
            while i + SW <= n:
                y.unsafe_store(
                    i, y.unsafe_load[width=SW](i) + a * xk.unsafe_load[width=SW](i)
                )
                i += SW
            while i < n:
                y.unsafe_store(i, y.unsafe_load(i) + a * xk.unsafe_load(i))
                i += 1
        k += 1