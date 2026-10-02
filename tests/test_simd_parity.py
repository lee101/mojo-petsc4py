"""The SIMD sweep paths in `src/simdvec.mojo`, exercised against upstream.

Every elementwise `Vec` kernel in the port runs a vector body `SW` elements wide
and finishes with a scalar tail.  A test at a length that is a multiple of the
SIMD width only ever reaches the vector body; these cases walk every length from
0 to a few vectors past the width, so the tail is entered at every phase, and
they compare against real petsc4py running the same operation.

Pointwise results are required to match upstream entry for entry with no
tolerance: those kernels are deterministic, so a difference is a bug in the
sweep and not a rounding difference.  The reductions are compared with the same
tolerance `test_vec_parity.py` uses, because upstream reaches them through BLAS
with its own unrolling, and `exp`/`log` with the tolerance that file uses for
libm.

`dotNorm2` is not compared against upstream directly: petsc4py 3.25.5 raises
`TypeError` on that method on every call, which the README documents as
divergence 1, so it is checked against its own two parts instead.
"""

import numpy as np
import pytest

petsc4py = pytest.importorskip("petsc4py")
from petsc4py import PETSc as Ref  # noqa: E402

from mojopetsc4py import PETSc as Mojo  # noqa: E402

SEED = 424242
# The port reads the SIMD width of float64 at compile time; four is what AVX2
# gives.  The lengths below walk past it either way, so nothing here depends on
# the exact value -- what matters is that a whole number of vectors plus every
# possible remainder is covered.
SW = 4

# the same tolerances test_vec_parity.py uses
TOL = 1e-13
TRANSCENDENTAL_TOL = 1e-9

# petsc.h PETSC_MIN_REAL / PETSC_MAX_REAL, the seeds VecMax_Seq and VecMin_Seq
# start from
MIN_REAL = -1.7976931348623157e308
MAX_REAL = 1.7976931348623157e308

LENGTHS = list(range(SW * 3 + 4))


def same(x, y, tol=TOL):
    np.testing.assert_allclose(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        rtol=tol,
        atol=tol,
        equal_nan=True,
    )


def pair(a):
    """Matched (port, upstream) vectors holding the same values."""
    ref = Ref.Vec().createSeq(len(a))
    ref.setArray(a)
    mojo = Mojo.Vec().createSeq(len(a))
    mojo.setArray(a)
    return mojo, ref


def data(n, seed=SEED):
    rng = np.random.default_rng(seed + n)
    return rng.normal(size=n) * 4.0


def two(n, seed=SEED):
    """Two independent arrays of length `n`."""
    return data(n, seed), data(n, seed + 1)


@pytest.mark.parametrize("n", LENGTHS)
def test_pointwise_maps_match_upstream_exactly(n):
    """Every 2-in/1-out sweep, at every length that leaves a remainder."""
    a, b = two(n)
    for method in ("pointwiseMult", "pointwiseMin", "pointwiseMax",
                   "pointwiseMaxAbs"):
        pm, rm = pair(a)
        pb, rb = pair(b)
        getattr(rm, method)(rb, rm)
        getattr(pm, method)(pb, pm)
        assert np.array_equal(rm.getArray(), pm.getArray()), method
    # the divisor is kept off zero, so dividing by zero -- which both sides
    # agree on anyway -- is not what is under test here
    pm, rm = pair(a)
    pb, rb = pair(np.abs(b) + 1.0)
    rm.pointwiseDivide(rb, rm)
    pm.pointwiseDivide(pb, pm)
    assert np.array_equal(rm.getArray(), pm.getArray())


@pytest.mark.parametrize("n", LENGTHS)
def test_reductions_match_upstream(n):
    """The reduction sweeps, including the empty-vector case."""
    a, b = two(n)
    pm, rm = pair(a)
    pb, rb = pair(b)
    same(pm.dot(pb), rm.dot(rb))
    same(pm.tDot(pb), rm.tDot(rb))
    same(pm.norm(), rm.norm())
    same(pm.sum(), rm.sum())
    if not n:
        return
    # dotNorm2 is the fused dot plus squared norm
    got = pm.dotNorm2(pb)
    same(got[0], rm.dot(rb))
    same(got[1], rm.norm() ** 2)
    same(pm.mean(), rm.mean())
    same(pm.norm(Mojo.NormType.NORM_1), rm.norm(Ref.NormType.NORM_1))
    same(
        pm.norm(Mojo.NormType.NORM_INFINITY),
        rm.norm(Ref.NormType.NORM_INFINITY),
        tol=0.0,
    )
    same(pm.norm(Mojo.NormType.NORM_1_AND_2),
         rm.norm(Ref.NormType.NORM_1_AND_2))
    assert pm.max() == rm.max()
    assert pm.min() == rm.min()

    # a zero vector is the branch VecNormalize guards on
    pm, rm = pair(a * 0.0)
    assert pm.norm() == 0.0
    pm.normalize()
    rm.normalize()
    same(pm.getArray(), rm.getArray(), tol=0.0)


@pytest.mark.parametrize("n", LENGTHS)
def test_inplace_kernels_match_upstream(n):
    """The sweeps that rewrite the operand they read."""
    a, b = two(n)
    for method, arg in (("scale", 0.25), ("shift", 3.5)):
        pm, rm = pair(a)
        getattr(pm, method)(arg)
        getattr(rm, method)(arg)
        same(pm.getArray(), rm.getArray(), tol=0.0)

    for method in ("abs", "sqrtabs", "conjugate"):
        pm, rm = pair(a)
        getattr(pm, method)()
        getattr(rm, method)()
        same(pm.getArray(), rm.getArray(), tol=0.0)

    # log is undefined on a non-positive input, so it gets a positive one
    pm, rm = pair(np.abs(a) + 1.0)
    pm.log()
    rm.log()
    same(pm.getArray(), rm.getArray(), tol=TRANSCENDENTAL_TOL)

    pm, rm = pair(a)
    pm.exp()
    rm.exp()
    same(pm.getArray(), rm.getArray(), tol=TRANSCENDENTAL_TOL)

    pm, rm = pair(a)
    pb, rb = pair(b)
    pm.axpy(2.0, pb)
    rm.axpy(2.0, rb)
    same(pm.getArray(), rm.getArray(), tol=0.0)

    pm, rm = pair(a)
    pb, rb = pair(b)
    pm.aypx(-0.5, pb)
    rm.aypx(-0.5, rb)
    same(pm.getArray(), rm.getArray(), tol=0.0)

    # the three axpby branches: both coefficients live, one of each zeroed
    for alpha, beta in ((2.0, 3.0), (0.0, 2.0), (2.0, 0.0)):
        pm, rm = pair(a)
        pb, rb = pair(b)
        pm.axpby(alpha, beta, pb)
        rm.axpby(alpha, beta, rb)
        same(pm.getArray(), rm.getArray(), tol=0.0)

    # axpbypcz reads three vectors and writes the third: the fused form.  It is
    # a port-only entry point -- upstream reaches the same operation through
    # MAXPY -- so it is checked against the arithmetic it stands for.
    # axpbypcz(alpha, beta, gamma, x, y) computes z = alpha*x + beta*y + gamma*z
    # on self, reading z before storing it, so the right-hand side uses the
    # pre-update values throughout
    pm, _rm = pair(a)
    pb, _rb = pair(b)
    pm.axpbypcz(1.5, -2.0, 0.5, pb, pm)
    # the port sums the three terms in the same order the C routine does, which
    # numpy's expression reassociates, so this one is a few-ulp comparison
    same(pm.getArray(), 1.5 * b - 2.0 * a + 0.5 * a, tol=TOL)

    # waxpy: self becomes w = alpha*x + y.  Upstream rejects w aliasing y, so
    # the destination here is self and y is a third, separate vector
    c = data(n, SEED + 2)
    pw, rw = pair(a)
    pb, rb = pair(b)
    pc, rc = pair(c)
    pw.waxpy(1.25, pb, pc)
    rw.waxpy(1.25, rb, rc)
    # upstream runs waxpy through its own vector-kernel path, which sums
    # alpha*x and y in a different order, so this one is a few-ulp comparison
    same(pw.getArray(), rw.getArray(), tol=TOL)
    same(pw.getArray(), 1.25 * b + c, tol=TOL)


@pytest.mark.parametrize("n", LENGTHS)
def test_axpy_with_self_alias(n):
    """`axpy` is the one elementwise kernel upstream allows to alias, and it is
    the one where the vector body reads and writes the same buffer."""
    a, _b = two(n)
    pm, rm = pair(a)
    pm.axpy(2.0, pm)
    rm.axpy(2.0, rm)
    same(pm.getArray(), rm.getArray(), tol=0.0)
    # x aliased into y means y = (1 + alpha) * y
    same(pm.getArray(), a * 3.0, tol=0.0)


@pytest.mark.parametrize("n", LENGTHS)
def test_swap_matches_upstream(n):
    a, b = two(n)
    # (rm, rb) is the upstream pair holding (a, b), (pm, pb) the port pair
    pm, rm = pair(a)
    pb, rb = pair(b)
    pm.swap(pb)
    rm.swap(rb)
    # after the swap each side holds a's values in its y and b's in its x
    same(pm.getArray(), rm.getArray(), tol=0.0)
    same(pb.getArray(), rb.getArray(), tol=0.0)
    same(pm.getArray(), b, tol=0.0)
    same(pb.getArray(), a, tol=0.0)


@pytest.mark.parametrize("n", LENGTHS)
def test_set_zero_and_copy(n):
    pm, rm = pair(np.arange(n, dtype=np.float64))
    pm.set(2.5)
    rm.set(2.5)
    same(pm.getArray(), rm.getArray(), tol=0.0)

    pm.zeroEntries()
    rm.zeroEntries()
    same(pm.getArray(), rm.getArray(), tol=0.0)

    a, b = two(n)
    cm, cr = pair(a)
    dm, dr = pair(b)
    same(cm.copy().getArray(), cr.copy().getArray(), tol=0.0)
    same(dm.copy().getArray(), dr.copy().getArray(), tol=0.0)


@pytest.mark.parametrize("n", LENGTHS)
def test_guarded_kernels_match_upstream(n):
    """The sweeps whose body is a compare and a branch: reciprocal, chop and
    maxPointwiseDivide."""
    if not n:
        return
    a, b = two(n)
    # an exact zero to exercise the reciprocal guard, and a very small entry to
    # show the guard is an equality test rather than a tolerance; at n == 1
    # they would be the same entry, so only the tiny one is placed
    a[-1] = 1e-20
    b[0] = 0.0
    b[-1] = 1e-20
    if n > 1:
        a[0] = 0.0

    pm, rm = pair(a)
    pm.reciprocal()
    rm.reciprocal()
    same(pm.getArray(), rm.getArray())
    assert pm.getArray()[-1] == pytest.approx(1e20, rel=1e-12)
    if n > 1:
        # the guard is exact: an exact zero becomes 0, not inf
        assert pm.getArray()[0] == 0.0

    pm, rm = pair(a)
    pm.chop(1e-10)
    rm.chop(1e-10)
    same(pm.getArray(), rm.getArray(), tol=0.0)

    pm, rm = pair(a)
    pb, rb = pair(b)
    same(pm.maxPointwiseDivide(pb), rm.maxPointwiseDivide(rb))


@pytest.mark.parametrize("n", LENGTHS)
def test_maxpy_over_several_vectors(n):
    """`maxpy` walks a list of vectors, one sweep per term."""
    rng = np.random.default_rng(SEED)
    vs = [rng.normal(size=n) for _ in range(4)]
    alphas = np.array([1.5, -2.0, 0.25, 3.0])
    port_v = [Mojo.Vec().createSeq(n) for _ in vs]
    ref_v = [Ref.Vec().createSeq(n) for _ in vs]
    for v, p, r in zip(vs, port_v, ref_v):
        p.setArray(v)
        r.setArray(v)
    ones = np.ones(n)
    pm, rm = pair(ones)
    pm.maxpy(alphas, port_v)
    rm.maxpy(alphas, ref_v)
    same(pm.getArray(), rm.getArray())
    same(pm.getArray(), 1.0 + sum(alphas[k] * vs[k] for k in range(4)))


@pytest.mark.parametrize("n", LENGTHS)
def test_min_max_tie_breaking(n):
    """The extreme index is order-dependent, so `VecMinMax_Seq` is the one
    reduction here that is still a scalar scan.  These are the cases that pin
    its tie-breaking down, which is what a vector reduction would break."""
    if not n:
        # upstream reports the seed it started from, and index -1
        pm, rm = pair(np.zeros(0))
        assert pm.max() == rm.max() == (-1, MIN_REAL)
        assert pm.min() == rm.min() == (-1, MAX_REAL)
        return

    if n == 1:
        # the first and last entry are the same one
        pm, rm = pair(np.array([5.0]))
        assert pm.max() == rm.max() == (0, 5.0)
        assert pm.min() == rm.min() == (0, 5.0)
        return

    # two entries tie at 5.0; the strict comparison reports the earlier one
    a = np.full(n, 2.0)
    a[n // 2] = 5.0
    a[-1] = 5.0
    pm, rm = pair(a)
    assert pm.max() == rm.max() == (n // 2, 5.0)
    assert pm.min() == rm.min() == (0, 2.0)

    # a tied pair at the two ends, with a constant in between: the minimum is
    # the tied pair and the strict comparison reports 0, while the maximum is
    # the constant, first seen at index 1 because the accumulator is seeded
    # from x[0]
    b = np.full(n, -3.0)
    b[0] = -9.0
    b[-1] = -9.0
    pm, rm = pair(b)
    assert pm.min() == rm.min() == (0, -9.0)
    # at n == 2 the two end positions are the same entry, so there is no -3
    # left and the maximum is the -9 itself
    expected_max = (1, -3.0) if n > 2 else (0, -9.0)
    assert pm.max() == rm.max() == expected_max

    # all-equal: no entry is ever strictly greater, so both report index 0
    pm, rm = pair(np.full(n, 4.0))
    assert pm.max() == rm.max() == (0, 4.0)
    assert pm.min() == rm.min() == (0, 4.0)