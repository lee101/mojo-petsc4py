"""The block-accumulator reductions and the batched inner products.

`test_simd_parity.py` walks lengths 0 to 15, which never enters a reduction's
four-accumulator block.  These cases are the ones that do: every length from the
block width to several blocks past it, and the lengths a Krylov solve actually
uses.  They also pin the two operations that must not move under the batched
inner products: a row's dot is the value a separate `dot` produces, and the
GMRES/FGMRES solves agree with upstream done the same way.
"""

import numpy as np
import pytest

petsc4py = pytest.importorskip("petsc4py")
from petsc4py import PETSc as Ref  # noqa: E402

from mojopetsc4py import PETSc as Mojo  # noqa: E402

# The block is four vectors wide (src/simd.mojo BLK); SW is the vector width.
SW = 4
BLK = 4 * SW
TOL = 1e-13

# lengths on both sides of every boundary the reduction has: the block body, the
# one-vector body, the scalar tail, and the Krylov sizes
LENGTHS = sorted({
    0, 1, 2, SW - 1, SW, SW + 1,
    BLK - 1, BLK, BLK + 1, 2 * BLK - 1, 2 * BLK, 2 * BLK + 1,
    3 * BLK + 5, 1024, 1025, 4096, 4097, 16384, 16385,
})


def data(n, seed=7):
    rng = np.random.default_rng(seed + n)
    return rng.normal(size=n) * 4.0


def pair(a):
    ref = Ref.Vec().createSeq(len(a))
    ref.setArray(a)
    mojo = Mojo.Vec().createSeq(len(a))
    mojo.setArray(a)
    return mojo, ref


def same(x, y, tol=TOL):
    np.testing.assert_allclose(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64),
        rtol=tol, atol=tol, equal_nan=True,
    )


@pytest.mark.parametrize("n", LENGTHS)
def test_reductions_agree_with_numpy_and_upstream(n):
    """A block reduction must count every element exactly once.

    The reference here is NumPy over the whole array as well as upstream, so a
    body that skipped a block (or double-counted the tail) fails on the value
    rather than on a tolerance.
    """
    a = data(n)
    pm, rm = pair(a)
    if not n:
        # an empty reduction is 0 and an empty mean is NaN on both sides
        same(pm.sum(), 0.0)
        assert np.isnan(pm.mean()) and np.isnan(rm.mean())
        same(pm.norm(), 0.0)
        return
    same(pm.sum(), np.sum(a))
    same(pm.sum(), rm.sum())
    same(pm.mean(), np.mean(a))
    same(pm.mean(), rm.mean())
    same(pm.norm(), np.sqrt(np.sum(a * a)))
    same(pm.norm(), rm.norm())
    same(pm.norm(Mojo.NormType.NORM_1), np.sum(np.abs(a)))
    same(pm.norm(Mojo.NormType.NORM_1), rm.norm(Ref.NormType.NORM_1))
    same(pm.dot(pm), np.sum(a * a))
    same(pm.dot(pm), rm.dot(rm))


@pytest.mark.parametrize("n", LENGTHS)
def test_dot_norm2_is_the_two_parts(n):
    """The fused (x.y, x.x) is each part computed over the same array."""
    a, b = data(n, 11), data(n, 12)
    pm, _ = pair(a)
    pb, _ = pair(b)
    d, q = pm.dotNorm2(pb)
    same(d, pm.dot(pb))
    same(q, pm.dot(pm))


@pytest.mark.parametrize("n", [0, 1, SW - 1, SW, SW + 1, 2 * SW + 3, 64])
def test_reciprocal_guards(n):
    """The reciprocal is exact-zero guarded and leaves every other entry alone.

    A zero maps to zero; a subnormal reciprocates to infinity rather than being
    folded into the guard; a NaN stays a NaN.
    """
    if not n:
        return
    a = data(n) + 3.0
    a[-1] = 0.0
    if n > 1:
        a[0] = 5e-324          # the smallest subnormal: 1/x overflows to inf
    if n > 2:
        a[1] = np.nan
    pm, rm = pair(a)
    pm.reciprocal()
    rm.reciprocal()
    same(pm.getArray(), rm.getArray())
    assert pm.getArray()[-1] == 0.0
    if n > 1:
        assert np.isinf(pm.getArray()[0])
    if n > 2:
        assert np.isnan(pm.getArray()[1])


@pytest.mark.parametrize("n", [0, 1, SW - 1, SW, SW + 1, 2 * SW + 3, 64])
def test_chop_threshold_is_exact(n):
    """`chop` zeroes strictly below `tol` and leaves `tol` itself untouched."""
    if not n:
        return
    a = np.abs(data(n)) + 0.25
    a[0] = 1.0                # exactly tol: must be rewritten to 1.0, not zero
    if n > 1:
        a[1] = 1.0 - 1e-12    # just below
    pm, rm = pair(a)
    pm.chop(1.0)
    rm.chop(1.0)
    same(pm.getArray(), rm.getArray(), tol=0.0)
    assert pm.getArray()[0] == 1.0
    if n > 1:
        assert pm.getArray()[1] == 0.0


def solve(ksp_type, n=1024):
    """Solve the 1-D Laplacian with the port and with upstream, same settings.

    GMRES/FGMRES orthogonalise with the classical Gram-Schmidt path, which is
    the one that batches its inner products through `VecMDot`; this drives that
    path through a full solve rather than through a unit-scale call.
    """
    rng = np.random.default_rng(3)
    b = rng.normal(size=n)
    # tridiagonal SPD: 4 on the diagonal, -1 off it, as CSR with int32 indices
    diag = 4.0
    off = -1.0
    rows = np.repeat(np.arange(n, dtype=np.int64), 3)
    cols = (rows + np.tile(np.array([-1, 0, 1]), n)).astype(np.int64)
    keep = (cols >= 0) & (cols < n)
    rows, cols = rows[keep], cols[keep]
    vals = np.where(cols == rows, diag, off)
    counts = np.bincount(rows, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)

    out = []
    for mod in (Mojo, Ref):
        if mod is Mojo:
            m = mod.Mat().createSeqAIJ((n, n), nnz=3)
        else:
            m = mod.Mat().create()
            m.setSizes([n, n])
            m.setType("seqaij")
            m.setUp()
        m.setValuesCSR(indptr.astype(np.int32), cols.astype(np.int32), vals)
        m.assemble()
        pc = mod.PC().create()
        pc.setType("none")
        pc.setOperators(m)
        ksp = mod.KSP().create()
        ksp.setPC(pc)
        ksp.setOperators(m)
        ksp.setType(ksp_type)
        ksp.setTolerances(rtol=1e-10, atol=1e-50, divtol=1e30, max_it=2000)
        bv = mod.Vec().createSeq(n)
        bv.setArray(b)
        xv = mod.Vec().createSeq(n)
        ksp.solve(bv, xv)
        out.append((ksp.getIterationNumber(), xv.getArray().copy()))
    return out


@pytest.mark.parametrize("ksp_type", ["gmres", "fgmres", "cg", "bcgs"])
def test_solve_through_batched_inner_products_matches_upstream(ksp_type):
    """GMRES/FGMRES run their batched inner products inside a real solve.

    The two Arnoldi methods are the drivers that call the batched form; CG and
    BCGS call the single-vector `dot` and are here as the control that the
    change did not move them.  The iteration count and the iterate are both
    asserted, so a reassociated inner product shows up as a different iterate
    rather than passing on a tolerance.
    """
    (its_m, x_m), (its_r, x_r) = solve(ksp_type)
    assert its_m == its_r
    np.testing.assert_allclose(x_m, x_r, rtol=1e-10, atol=1e-12)
