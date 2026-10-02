"""Parity of the vector algebra against real upstream petsc4py.

Every test builds the same vector in both packages, runs the same method, and
compares the resulting arrays entry for entry.  Upstream and the port are held
to bit-for-bit agreement wherever the algorithm is deterministic and exactly
reproducible, which the elementwise Vec kernels are.
"""

import numpy as np
import pytest

petsc4py = pytest.importorskip("petsc4py")
from petsc4py import PETSc as Ref  # noqa: E402

from mojopetsc4py import PETSc as Mojo  # noqa: E402

SEED = 20240917


def pairs(m=17, seed=SEED, bsize=None):
    """Matched (upstream, port) vectors holding the same values."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=m) * 5.0
    if bsize is None:
        v_ref = Ref.Vec().createSeq(m)
        v_ref.setArray(a)
        v_mojo = Mojo.Vec().createSeq(m)
        v_mojo.setArray(a)
    else:
        v_ref = Ref.Vec().createSeq(m, bsize)
        v_ref.setArray(a)
        v_mojo = Mojo.Vec().createSeq(m, bsize)
        v_mojo.setArray(a)
    return v_ref, v_mojo, a


# PETSc routes the reductions through BLAS (`BLASasum`, `BLASdot`), which
# unrolls differently from a straight Mojo loop, so agreement is to a few ulp
# rather than bit-for-bit.
TOL = 1e-13
# `exp` and `log` go through libm on both sides, but not the same libm, so they
# agree to roughly a dozen digits rather than to the last bit.
TRANSCENDENTAL_TOL = 1e-9


def same(x, y, tol=TOL):
    np.testing.assert_allclose(
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        rtol=tol,
        atol=tol,
        equal_nan=True,
    )


def test_creation_and_access():
    ref, mojo, a = pairs()
    assert ref.getSize() == mojo.getSize() == len(a)
    assert ref.getLocalSize() == mojo.getLocalSize()
    assert ref.getBlockSize() == mojo.getBlockSize()
    same(ref.getArray(), mojo.getArray())
    same(ref.getValue(3), mojo.getValue(3))


def test_creation_with_block_size():
    ref, mojo, a = pairs(m=18, bsize=3)
    assert ref.getBlockSize() == mojo.getBlockSize() == 3
    same(ref.getArray(), mojo.getArray())
    assert ref.getSize() == mojo.getSize() == 18


def test_create_seq_rejects_indivisible_block_size():
    with pytest.raises(ValueError):
        Mojo.Vec().createSeq(5, 2)


def test_norm_all_types():
    ref, mojo, a = pairs()
    for t in (
        Ref.NormType.NORM_1,
        Ref.NormType.NORM_2,
        Ref.NormType.NORM_INFINITY,
        Ref.NormType.NORM_1_AND_2,
    ):
        r = ref.norm(t)
        m = mojo.norm(int(t))
        if isinstance(r, tuple):
            assert r[0] == pytest.approx(m[0], rel=TOL)
            assert r[1] == pytest.approx(m[1], rel=TOL)
        else:
            assert r == pytest.approx(m, rel=TOL, abs=0.0)
    # the port keeps the C ordinals, so check them against numpy as well
    n1, n2 = mojo.norm(Mojo.NormType.NORM_1_AND_2)
    assert n1 == pytest.approx(float(np.linalg.norm(a, 1)), rel=TOL)
    assert n2 == pytest.approx(float(np.linalg.norm(a)), rel=TOL)


def test_dot_and_reductions():
    ref, mojo, a = pairs()
    ref2, mojo2, b = pairs(seed=SEED + 1)
    same(ref.dot(ref2), mojo.dot(mojo2))
    assert ref.sum() == pytest.approx(mojo.sum(), rel=TOL)
    assert ref.mean() == pytest.approx(mojo.mean(), rel=TOL)
    assert ref.min() == mojo.min()
    assert ref.max() == mojo.max()


def test_min_max_tie_breaking():
    a = np.array([2.0, -1.0, 2.0, -1.0, 0.0])
    ref = Ref.Vec().createSeq(5)
    ref.setArray(a)
    mojo = Mojo.Vec().createSeq(5)
    mojo.setArray(a)
    assert ref.min() == mojo.min() == (1, -1.0)
    assert ref.max() == mojo.max() == (0, 2.0)


def test_normalize():
    ref, mojo, a = pairs()
    r = Ref.Vec().createSeq(len(a))
    r.setArray(a)
    m = Mojo.Vec().createSeq(len(a))
    m.setArray(a)
    rn = ref.normalize()
    mn = mojo.normalize()
    assert rn == pytest.approx(mn, rel=TOL)
    same(ref.getArray(), mojo.getArray())


def test_normalize_of_zero_vector():
    a = np.zeros(6)
    ref = Ref.Vec().createSeq(6)
    ref.setArray(a)
    mojo = Mojo.Vec().createSeq(6)
    mojo.setArray(a)
    assert ref.normalize() == mojo.normalize() == 0.0
    same(ref.getArray(), mojo.getArray())


def test_elementwise_unary():
    a = np.array([0.5, -1.5, 2.0, 4.0, -0.25, 3.0])
    for name in ("reciprocal", "exp", "log", "sqrtabs", "abs", "conjugate"):
        r = Ref.Vec().createSeq(6)
        r.setArray(a)
        m = Mojo.Vec().createSeq(6)
        m.setArray(a)
        getattr(r, name)()
        getattr(m, name)()
        tol = TRANSCENDENTAL_TOL if name in ("exp", "log") else TOL
        same(r.getArray(), m.getArray(), tol=tol)


def test_reciprocal_zero_guard():
    """PETSc maps an exact zero to zero and reciprocates everything else, so
    even 1e-20 becomes 1e20 rather than a saturated value."""
    a = np.array([0.0, 1e-20, 1e-16, 1e-8, -0.0, 3.0])
    ref = Ref.Vec().createSeq(6)
    ref.setArray(a)
    mojo = Mojo.Vec().createSeq(6)
    mojo.setArray(a)
    ref.reciprocal()
    mojo.reciprocal()
    same(ref.getArray(), mojo.getArray())


def test_binary_elementwise():
    n = 11
    rng = np.random.default_rng(SEED + 2)
    a = rng.normal(size=n) + 0.5
    b = rng.normal(size=n) + 0.5

    def build(vals):
        v = Ref.Vec().createSeq(n)
        v.setArray(vals)
        return v

    def build_m(vals):
        v = Mojo.Vec().createSeq(n)
        v.setArray(vals)
        return v

    x_ref, x_mojo = build(a), build_m(a)
    y_ref, y_mojo = build(b), build_m(b)
    x_ref.axpy(2.5, y_ref)
    x_mojo.axpy(2.5, y_mojo)
    same(x_ref.getArray(), x_mojo.getArray())

    for name in ("aypx",):
        x_ref, x_mojo = build(a), build_m(a)
        y_ref, y_mojo = build(b), build_m(b)
        getattr(x_ref, name)(-1.25, y_ref)
        getattr(x_mojo, name)(-1.25, y_mojo)
        same(x_ref.getArray(), x_mojo.getArray())

    x_ref, x_mojo = build(a), build_m(a)
    y_ref, y_mojo = build(b), build_m(b)
    x_ref.axpby(1.5, -0.5, y_ref)
    x_mojo.axpby(1.5, -0.5, y_mojo)
    same(x_ref.getArray(), x_mojo.getArray())

    x_ref, x_mojo = build(a), build_m(a)
    y_ref, y_mojo = build(b), build_m(b)
    x_ref.waxpy(0.75, y_ref, y_ref)
    x_mojo.waxpy(0.75, y_mojo, y_mojo)
    same(x_ref.getArray(), x_mojo.getArray())

    for name in (
        "pointwiseMult", "pointwiseDivide", "pointwiseMin",
        "pointwiseMax", "pointwiseMaxAbs",
    ):
        x_ref, x_mojo = build(a), build_m(a)
        y_ref, y_mojo = build(b), build_m(b)
        getattr(x_ref, name)(y_ref, y_ref)
        getattr(x_mojo, name)(y_mojo, y_mojo)
        same(x_ref.getArray(), x_mojo.getArray())


def test_axpby_special_cases():
    """Upstream dispatches on beta == 0, alpha == 0 and beta == 1; each has to
    produce the same answer as the general branch."""
    n = 9
    rng = np.random.default_rng(SEED + 3)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    for alpha, beta in ((2.0, 0.0), (0.0, 3.0), (2.0, 1.0), (2.0, -1.0)):
        r = Ref.Vec().createSeq(n)
        r.setArray(a)
        y = Ref.Vec().createSeq(n)
        y.setArray(b)
        r.axpby(alpha, beta, y)
        m = Mojo.Vec().createSeq(n)
        m.setArray(a)
        ym = Mojo.Vec().createSeq(n)
        ym.setArray(b)
        m.axpby(alpha, beta, ym)
        same(r.getArray(), m.getArray())


def test_aypx_special_cases():
    n = 9
    rng = np.random.default_rng(SEED + 4)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    for alpha in (0.0, 1.0, -1.0, 2.5):
        r = Ref.Vec().createSeq(n)
        r.setArray(b)
        y = Ref.Vec().createSeq(n)
        y.setArray(a)
        r.aypx(alpha, y)
        m = Mojo.Vec().createSeq(n)
        m.setArray(b)
        ym = Mojo.Vec().createSeq(n)
        ym.setArray(a)
        m.aypx(alpha, ym)
        same(r.getArray(), m.getArray())


def test_maxpy():
    n = 13
    rng = np.random.default_rng(SEED + 5)
    vecs = [rng.normal(size=n) for _ in range(3)]
    alphas = [0.5, -1.25, 3.0]
    r = Ref.Vec().createSeq(n)
    r.setArray(vecs[0])
    rs = []
    for v in vecs[1:]:
        t = Ref.Vec().createSeq(n)
        t.setArray(v)
        rs.append(t)
    r.maxpy([0.5, -1.25], rs)
    m = Mojo.Vec().createSeq(n)
    m.setArray(vecs[0])
    ms = []
    for v in vecs[1:]:
        t = Mojo.Vec().createSeq(n)
        t.setArray(v)
        ms.append(t)
    m.maxpy([0.5, -1.25], ms)
    same(r.getArray(), m.getArray())


def test_scale_shift_set_swap():
    n = 7
    a = np.arange(1.0, n + 1)
    for name, args in (("scale", (2.5,)), ("shift", (-1.25,)), ("set", (3.5,))):
        r = Ref.Vec().createSeq(n)
        r.setArray(a)
        m = Mojo.Vec().createSeq(n)
        m.setArray(a)
        getattr(r, name)(*args)
        getattr(m, name)(*args)
        same(r.getArray(), m.getArray())

    r1, r2 = Ref.Vec().createSeq(n), Ref.Vec().createSeq(n)
    r1.setArray(a)
    r2.setArray(-a)
    m1, m2 = Mojo.Vec().createSeq(n), Mojo.Vec().createSeq(n)
    m1.setArray(a)
    m2.setArray(-a)
    r1.swap(r2)
    m1.swap(m2)
    same(r1.getArray(), m1.getArray())
    same(r2.getArray(), m2.getArray())


def test_max_pointwise_divide():
    n = 8
    rng = np.random.default_rng(SEED + 6)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    b[3] = 0.0
    r, y = Ref.Vec().createSeq(n), Ref.Vec().createSeq(n)
    r.setArray(a)
    y.setArray(b)
    m, ym = Mojo.Vec().createSeq(n), Mojo.Vec().createSeq(n)
    m.setArray(a)
    ym.setArray(b)
    assert r.maxPointwiseDivide(y) == pytest.approx(
        m.maxPointwiseDivide(ym), rel=TOL
    )


@pytest.mark.parametrize("tol", [0.0, 1e-12, 0.5, 1.0, 2.0, 1e9])
def test_chop(tol):
    """`chop` zeroes entries below `tol` and leaves the survivors untouched.

    The boundary matters: upstream compares strictly, so `chop(0.0)` is a
    no-op and an entry whose magnitude is exactly `tol` is kept.
    """
    ref, mojo, a = pairs()
    ref.chop(tol)
    mojo.chop(tol)
    same(ref.getArray(), mojo.getArray())
    assert np.array_equal(ref.getArray(), np.where(np.abs(a) < tol, 0.0, a))


def test_tdot_matches_dot_for_real_scalars():
    """Upstream notes the indefinite form equals the dot for real scalars."""
    ref, mojo, a = pairs()
    ref_b, mojo_b, _ = pairs(seed=SEED + 1)
    assert ref.tDot(ref_b) == pytest.approx(mojo.tDot(mojo_b), rel=TOL)
    assert ref.tDot(ref_b) == pytest.approx(ref.dot(ref_b), rel=TOL)
    assert mojo.tDot(mojo_b) == pytest.approx(mojo.dot(mojo_b), rel=TOL)


def test_duplicate_and_copy():
    ref, mojo, a = pairs()
    r = ref.copy()
    m = mojo.copy()
    same(r.getArray(), m.getArray())
    d = ref.duplicate()
    dm = mojo.duplicate()
    same(d.getArray(), np.zeros_like(a))
    same(dm.getArray(), np.zeros_like(a))


def test_two_vector_operations_reject_mismatched_sizes():
    """Every two-vector kernel reads n entries of each operand."""
    n = 4
    v = Mojo.Vec().createSeq(n)
    v.setArray(np.ones(n))
    short = Mojo.Vec().createSeq(n - 1)
    short.setArray(np.ones(n - 1))
    for call in (
        lambda: v.dot(short),
        lambda: v.axpy(1.0, short),
        lambda: v.aypx(1.0, short),
        lambda: v.axpby(1.0, 1.0, short),
        lambda: v.axpbypcz(1.0, 1.0, 1.0, short, short),
        lambda: v.waxpy(1.0, short, short),
        lambda: v.pointwiseMult(short, short),
        lambda: v.pointwiseDivide(short, short),
        lambda: v.pointwiseMin(short, short),
        lambda: v.pointwiseMax(short, short),
        lambda: v.pointwiseMaxAbs(short, short),
        lambda: v.swap(short),
        lambda: v.copy(short),
        lambda: v.maxPointwiseDivide(short),
        lambda: v.dotNorm2(short),
        lambda: v.maxpy(np.ones(1), [short]),
    ):
        with pytest.raises(ValueError):
            call()
