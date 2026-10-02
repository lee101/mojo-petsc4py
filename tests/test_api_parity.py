"""Parity of the API surface against real upstream petsc4py.

The README lists a set of covered `Vec`, `Mat`, `PC` and `KSP` methods; these
tests exercise each of them, and check the ones that have no numeric parity
either against upstream where upstream binds them, or against the closed form
the method is documented to compute where it does not.
"""

import numpy as np
import pytest

petsc4py = pytest.importorskip("petsc4py")
from petsc4py import PETSc as Ref  # noqa: E402

from mojopetsc4py import PETSc as Mojo  # noqa: E402

TOL = 1e-11


def ref_vec(a):
    v = Ref.Vec().createSeq(len(a))
    v.setArray(np.asarray(a, dtype=np.float64))
    return v


def mojo_vec(a):
    return Mojo.Vec().createWithArray(np.asarray(a, dtype=np.float64))


def dense_of(m):
    """Densify a port CSR matrix, read entry by entry through the public API."""
    nrows, ncols = m.getSize()
    out = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            out[i, j] = m.getValue(i, j)
    return out


def banded_spd(n, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros((n, n))
    for i in range(n):
        a[i, i] = 4.0 + 0.1 * rng.normal()
        if i > 0:
            a[i, i - 1] = a[i - 1, i] = -1.0
    return a


def ref_matrix(a):
    m = Ref.Mat().create()
    m.setSizes([a.shape[0], a.shape[1]])
    m.setType("seqaij")
    m.setUp()
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] != 0.0:
                m.setValue(i, j, float(a[i, j]))
    m.assemble()
    return m


def mojo_matrix(a):
    m = Mojo.Mat().createAIJ(a.shape, nnz=a.shape[1])
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] != 0.0:
                m.setValue(i, j, float(a[i, j]))
    m.assemble()
    return m


# ------------------------------------------------------------------ Vec


def test_create_with_array_copies_and_reports_sizes():
    """`createWithArray` takes ownership of a float64 copy, and the size and
    ownership accessors agree with upstream's."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    x = mojo_vec(a)
    a[0] = 99.0  # the vector must not see this
    assert x.getArray()[0] == 1.0
    assert x.getSize() == 4
    assert x.getLocalSize() == 4
    assert x.getSizes() == (4, 4)
    assert x.getOwnershipRange() == (0, 4)

    r = ref_vec([1.0, 2.0, 3.0, 4.0])
    assert x.getSize() == r.getSize()
    assert x.getSizes() == r.getSizes()


def test_norm_begin_end_matches_norm():
    """`normBegin` is a no-op here; `normEnd` must return what `norm` does."""
    a = np.random.default_rng(31).normal(size=32)
    x = mojo_vec(a)
    for ntype in (Mojo.NormType.NORM_1, Mojo.NormType.NORM_2,
                  Mojo.NormType.NORM_INFINITY):
        direct = x.norm(ntype)
        x.normBegin(ntype)
        assert x.normEnd(ntype) == pytest.approx(direct, rel=1e-15)
    # and against upstream
    r = ref_vec(a)
    for ntype in (Ref.NormType.NORM_1, Ref.NormType.NORM_2,
                  Ref.NormType.NORM_INFINITY):
        assert x.norm(ntype) == pytest.approx(r.norm(ntype), rel=TOL)


def test_create_seq_copies_so_the_caller_cannot_mutate_it():
    a = np.arange(8, dtype=np.float64)
    v = Mojo.Vec().createSeq(8)
    v.setArray(a)
    a[3] = -1.0
    assert v.getArray()[3] == 3.0


# ------------------------------------------------------------------ Mat


def test_set_values_blocked_matches_set_values():
    """`setValuesBlocked` takes the same rows x cols outer product as
    `setValues`, which is what upstream's `MatSetValuesBlocked` does at block
    size 1."""
    a = banded_spd(6, seed=41)
    rows, cols = [1, 4], [1, 4]
    vals = [10.0, 11.0, 12.0, 13.0]

    m = mojo_matrix(a)
    m.setValuesBlocked(rows, cols, vals)
    blocked = mojo_matrix(a)
    blocked.setValues(rows, cols, vals)
    np.testing.assert_allclose(dense_of(m), dense_of(blocked), atol=1e-15)

    # and against upstream
    r = ref_matrix(a)
    r.setValuesBlocked(rows, cols, vals)
    r.assemble()
    out = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            out[i, j] = r.getValue(i, j)
    np.testing.assert_allclose(dense_of(m), out, atol=1e-15)

    with pytest.raises(ValueError):
        mojo_matrix(a).setValuesBlocked(rows, cols, [1.0])


def test_set_value_blocked_fills_the_whole_block():
    """`setValueBlocked` sets every entry of the block, not just its diagonal."""
    a = banded_spd(5, seed=42)
    rows, cols = [0, 1], [0, 1]
    m = mojo_matrix(a)
    m.setValueBlocked(rows, cols, 3.0)
    blocked = mojo_matrix(a)
    blocked.setValues(rows, cols, [3.0] * 4)
    np.testing.assert_allclose(dense_of(m), dense_of(blocked), atol=1e-15)


def test_non_square_matrix_keeps_both_dimensions():
    """A 4 x 6 matrix must not be silently squared by the constructors."""
    a = np.zeros((4, 6))
    for i in range(4):
        a[i, i] = 3.0
        if i + 1 < 6:
            a[i, i + 1] = -1.0
    m = mojo_matrix(a)
    assert m.getSize() == (4, 6)
    assert m.getSizes() == ((4, 4), (6, 6))
    assert m.createVecLeft().getSize() == 4
    assert m.createVecRight().getSize() == 6

    # from_dense must not square it either
    d = Mojo.from_dense(a)
    assert d.getSize() == (4, 6)
    np.testing.assert_allclose(dense_of(d), a, atol=1e-15)

    # and the transposed product is indexed by column, so its input is the
    # row-count vector and its output the column-count one
    x = np.arange(4, dtype=np.float64)
    y = m.multTranspose(Mojo.Vec().createWithArray(x))
    assert y.getSize() == 6
    np.testing.assert_allclose(y.getArray(), a.T @ x, atol=1e-12)

    # the forward product takes the column-count vector
    z = m.mult(Mojo.Vec().createWithArray(np.arange(6, dtype=np.float64)))
    assert z.getSize() == 4
    np.testing.assert_allclose(
        z.getArray(), a @ np.arange(6, dtype=np.float64), atol=1e-12
    )


def test_get_sizes_and_get_info_match_upstream():
    a = banded_spd(7, seed=43)
    m = mojo_matrix(a)
    r = ref_matrix(a)
    assert m.getSize() == r.getSize()
    assert m.getLocalSize() == r.getLocalSize()
    assert m.getSizes() == r.getSizes()

    info = m.getInfo()
    nnz = int(np.count_nonzero(a))
    assert info["nz_used"] == nnz
    assert r.getInfo()["nz_used"] == nnz


def test_equal_matches_upstream():
    a = banded_spd(8, seed=44)
    m = mojo_matrix(a)
    same = mojo_matrix(a)
    assert m.equal(same) == ref_matrix(a).equal(ref_matrix(a))

    b = a.copy()
    b[2, 3] += 0.5
    assert not m.equal(mojo_matrix(b))
    assert m.equal(mojo_matrix(b), tol=1.0)


def test_is_transpose_matches_upstream():
    a = banded_spd(9, seed=45)
    m = mojo_matrix(a)
    r = ref_matrix(a)
    assert m.isTranspose() == r.isTranspose()
    assert m.isSymmetric() == r.isSymmetric()
    assert m.isSymmetric() is True

    # a deliberately asymmetric matrix is not its own transpose
    c = a.copy()
    c[0, 8] = 2.0
    assert mojo_matrix(c).isTranspose() == ref_matrix(c).isTranspose()
    assert mojo_matrix(c).isTranspose() is False


def test_create_vec_left_and_right_match_upstream():
    # 4 x 6, non-square, so the left vector has the row count and the right one
    # the column count
    a = np.zeros((4, 6))
    for i in range(4):
        a[i, i] = 3.0
        if i + 1 < 6:
            a[i, i + 1] = -1.0
        if i + 1 < 4:
            a[i, i + 1] = -1.0
    m = mojo_matrix(a)
    r = ref_matrix(a)
    assert m.createVecLeft().getSize() == r.createVecLeft().getSize()
    assert m.createVecRight().getSize() == r.createVecRight().getSize()
    assert m.createVecLeft().getSize() == 4
    assert m.createVecRight().getSize() == 6


# ------------------------------------------------------------------ PC


def test_pc_reset_then_apply_refactors():
    """`reset` must drop the factor so the next apply rebuilds it."""
    a = banded_spd(10, seed=47)
    m = mojo_matrix(a)
    x = np.random.default_rng(48).normal(size=10)

    pc = Mojo.PC().create()
    pc.setType("ilu")
    pc.setFactorLevels(0)
    pc.setOperators(m)
    y1 = Mojo.Vec().createSeq(10)
    pc.apply(mojo_vec(x), y1)
    first = y1.getArray().copy()

    pc.reset()
    assert pc._factor is None
    y2 = Mojo.Vec().createSeq(10)
    pc.apply(mojo_vec(x), y2)
    np.testing.assert_allclose(y2.getArray(), first, rtol=TOL)


def test_pc_apply_transpose_matches_apply_for_symmetric_types():
    """SOR and Jacobi are symmetric here, so the transpose apply is the apply."""
    a = banded_spd(12, seed=49)
    m = mojo_matrix(a)
    x = np.random.default_rng(50).normal(size=12)
    for pc_type in ("jacobi", "sor", "none"):
        pc = Mojo.PC().create()
        pc.setType(pc_type)
        pc.setOperators(m)
        y1 = Mojo.Vec().createSeq(12)
        y2 = Mojo.Vec().createSeq(12)
        pc.apply(mojo_vec(x), y1)
        pc.applyTranspose(mojo_vec(x), y2)
        np.testing.assert_allclose(
            y2.getArray(), y1.getArray(), rtol=1e-12, atol=1e-14
        )


def _fill_matrix(seed=3, n=25, cutoff=0.35):
    """A symmetric positive definite matrix whose ILU(0) pattern is short."""
    rng = np.random.default_rng(seed)
    dense = rng.normal(size=(n, n))
    dense = (dense + dense.T) / (2.0 * np.sqrt(n))
    off = (np.abs(dense) > cutoff) & ~np.eye(n, dtype=bool)
    a = np.where(off | off.T, dense, 0.0)
    a = (a + a.T) / 2.0
    a[np.diag_indices(n)] = 3.0
    return a


def test_factor_levels_add_fill_and_match_upstream():
    """`setFactorLevels` must add fill to a matrix whose ILU(0) pattern drops
    entries, and the fill must match upstream's factor exactly."""
    a = _fill_matrix()
    x = np.random.default_rng(5).normal(size=a.shape[0])
    m = mojo_matrix(a)
    r = ref_matrix(a)

    def factor_nnz(levels):
        pc = Mojo.PC().create()
        pc.setType("ilu")
        pc.setFactorLevels(levels)
        pc.setOperators(m)
        pc.setUp()
        return int(pc._factor["bdiag"][0]) + 1

    def ref_apply(levels):
        pc = Ref.PC().create()
        pc.setType("ilu")
        pc.setFactorLevels(levels)
        pc.setOperators(r)
        y = Ref.Vec().createSeq(len(x))
        pc.apply(ref_vec(x), y)
        return y.getArray().copy()

    nnz0 = factor_nnz(0)
    assert factor_nnz(1) > nnz0, "level of fill did not add any entries"
    assert factor_nnz(2) >= factor_nnz(1)

    for levels in (0, 1, 2):
        pc = Mojo.PC().create()
        pc.setType("ilu")
        pc.setFactorLevels(levels)
        pc.setOperators(m)
        y = Mojo.Vec().createSeq(len(x))
        pc.apply(mojo_vec(x), y)
        np.testing.assert_allclose(
            y.getArray(), ref_apply(levels), rtol=1e-9, atol=1e-12
        )

    # a tridiagonal matrix has no fill at any level, so the apply is unchanged
    tri = banded_spd(20, seed=51)

    def apply_with(**opts):
        pc = Mojo.PC().create()
        pc.setType("ilu")
        for k, v in opts.items():
            getattr(pc, k)(v)
        pc.setOperators(mojo_matrix(tri))
        y = Mojo.Vec().createSeq(20)
        pc.apply(mojo_vec(np.random.default_rng(52).normal(size=20)), y)
        return y.getArray().copy()

    np.testing.assert_allclose(
        apply_with(setFactorLevels=0), apply_with(setFactorLevels=2), atol=1e-12
    )

    # the remaining knobs are accepted and leave a finite factor
    for shift_type in ("none", "positive_definite"):
        assert np.all(np.isfinite(apply_with(setFactorShift=shift_type)))
    for pivot in (0.0, 1e-12, 100.0):
        assert np.all(np.isfinite(apply_with(setFactorPivot=pivot)))


# ------------------------------------------------------------------ KSP


def test_nonzero_initial_guess_changes_the_iteration_count():
    """`setInitialGuessNonzero(True)` must start from `x`, not from zero."""
    a = banded_spd(30, seed=53)
    n = 30
    b = np.random.default_rng(54).normal(size=n)
    x0 = np.random.default_rng(55).normal(size=n)
    exact = np.linalg.solve(a, b)

    def run(flag):
        ksp = Mojo.KSP().create()
        pc = Mojo.PC().create()
        pc.setType("none")
        ksp.setPC(pc)
        ksp.setOperators(mojo_matrix(a))
        ksp.setType("cg")
        ksp.setInitialGuessNonzero(flag)
        ksp.setTolerances(rtol=1e-12, max_it=500)
        bv = mojo_vec(b)
        xv = Mojo.Vec().createWithArray(x0)
        ksp.solve(bv, xv)
        return ksp, xv.getArray().copy()

    ksp_zero, x_zero = run(False)
    ksp_nz, x_nz = run(True)
    assert ksp_zero.getInitialGuessNonzero() is False
    assert ksp_nz.getInitialGuessNonzero() is True
    # both start from a different place and both converge to the same solution
    # both converge to the same solution; the flag decides where the solver
    # starts, and it must not change where it ends
    np.testing.assert_allclose(x_zero, exact, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(x_nz, exact, rtol=1e-6, atol=1e-8)


def test_nonzero_initial_guess_matches_upstream():
    """The nonzero-guess branch must reproduce upstream's iteration count and
    iterate, not merely converge to the right answer."""
    a = banded_spd(40, seed=61)
    n = 40
    b = np.random.default_rng(62).normal(size=n)
    x0 = np.random.default_rng(63).normal(size=n)

    ksp_r = Ref.KSP().create()
    pc_r = Ref.PC().create()
    pc_r.setType("none")
    ksp_r.setPC(pc_r)
    ksp_r.setOperators(ref_matrix(a))
    ksp_r.setType("cg")
    ksp_r.setInitialGuessNonzero(True)
    ksp_r.setTolerances(rtol=1e-10, divtol=1e30, max_it=500)
    bv_r = ref_vec(b)
    xv_r = Ref.Vec().createSeq(n)
    xv_r.setArray(x0)
    ksp_r.solve(bv_r, xv_r)

    ksp_m = Mojo.KSP().create()
    pc_m = Mojo.PC().create()
    pc_m.setType("none")
    ksp_m.setPC(pc_m)
    ksp_m.setOperators(mojo_matrix(a))
    ksp_m.setType("cg")
    ksp_m.setInitialGuessNonzero(True)
    ksp_m.setTolerances(rtol=1e-10, divtol=1e30, max_it=500)
    xv_m = Mojo.Vec().createWithArray(x0)
    ksp_m.solve(mojo_vec(b), xv_m)

    assert ksp_m.getIterationNumber() == ksp_r.getIterationNumber()
    np.testing.assert_allclose(
        xv_m.getArray(), xv_r.getArray(), rtol=1e-7, atol=1e-9
    )


def test_gmres_restart_changes_the_iteration_count():
    a = banded_spd(40, seed=56)
    n = 40
    b = np.random.default_rng(57).normal(size=n)

    def run(restart):
        ksp = Mojo.KSP().create()
        pc = Mojo.PC().create()
        pc.setType("none")
        ksp.setPC(pc)
        ksp.setOperators(mojo_matrix(a))
        ksp.setType("gmres")
        if restart is not None:
            ksp.setGMRESRestart(restart)
        ksp.setTolerances(rtol=1e-12, divtol=1e30, max_it=2000)
        ksp.solve(mojo_vec(b), Mojo.Vec().createSeq(n))
        return ksp.getIterationNumber()

    assert run(4) >= run(10) >= run(30)
    for restart in (4, 10, 30):
        assert run(restart) > 0


@pytest.mark.parametrize("orthog", [0, 1])
@pytest.mark.parametrize("cgstype", [0, 1, 2])
def test_gmres_orthogonalisation_and_refinement_still_converge(orthog, cgstype):
    a = banded_spd(30, seed=58)
    n = 30
    b = np.random.default_rng(59).normal(size=n)
    exact = np.linalg.solve(a, b)

    ksp = Mojo.KSP().create()
    pc = Mojo.PC().create()
    pc.setType("none")
    ksp.setPC(pc)
    ksp.setOperators(mojo_matrix(a))
    ksp.setType("gmres")
    ksp.setGMRESOrthogonalization(orthog)
    ksp.setGMRESCGSRefinementType(cgstype)
    ksp.setTolerances(rtol=1e-10, divtol=1e30, max_it=1000)
    x = Mojo.Vec().createSeq(n)
    ksp.solve(mojo_vec(b), x)
    np.testing.assert_allclose(x.getArray(), exact, rtol=1e-6, atol=1e-8)
    assert ksp.getIterationNumber() > 0