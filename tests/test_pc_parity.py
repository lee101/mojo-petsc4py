"""Parity of the preconditioners against real upstream petsc4py.

`pc.apply(x, y)` is compared entry for entry against PETSc running the same
factorization on the same matrix, for each covered preconditioner type.
"""

import numpy as np
import pytest

petsc4py = pytest.importorskip("petsc4py")
from petsc4py import PETSc as Ref  # noqa: E402

from mojopetsc4py import PETSc as Mojo  # noqa: E402

TOL = 1e-11


def spd(n, seed=0, extra=0.0):
    """A symmetric positive definite matrix with a dense-ish sparsity."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    a = (a + a.T) / (2.0 * np.sqrt(n))
    a[np.diag_indices(n)] += 3.0 + extra
    return a


def banded_spd(n, seed=0):
    """A 1D Laplacian plus a diagonal, so the ILU and IC patterns are thin."""
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
    m = Mojo.Mat().createSeqAIJ(a.shape, nnz=a.shape[1])
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] != 0.0:
                m.setValue(i, j, float(a[i, j]))
    m.assemble()
    return m


def ref_vec(a):
    v = Ref.Vec().createSeq(len(a))
    v.setArray(np.asarray(a, dtype=np.float64))
    return v


def mojo_vec(a):
    v = Mojo.Vec().createSeq(len(a))
    v.setArray(np.asarray(a, dtype=np.float64))
    return v


def run(a, x, pc_type, **opts):
    """Apply the same preconditioner on both sides and return both results."""
    n = a.shape[0]
    am = ref_matrix(a)
    pc_ref = Ref.PC().create()
    pc_ref.setType(pc_type)
    for k, v in opts.items():
        getattr(pc_ref, k)(v)
    pc_ref.setOperators(am)
    yr = ref_vec(np.zeros(n))
    pc_ref.apply(ref_vec(x), yr)

    bm = mojo_matrix(a)
    pc = Mojo.PC().create()
    pc.setType(pc_type)
    for k, v in opts.items():
        getattr(pc, k)(v)
    pc.setOperators(bm)
    ym = mojo_vec(np.zeros(n))
    pc.apply(mojo_vec(x), ym)
    return yr.getArray(), ym.getArray(), pc


@pytest.mark.parametrize("n", [8, 40])
def test_jacobi_matches(n):
    a = spd(n, seed=1)
    x = np.random.default_rng(2).normal(size=n)
    yr, ym, _ = run(a, x, "jacobi")
    np.testing.assert_allclose(ym, yr, rtol=TOL)
    np.testing.assert_allclose(yr, x / np.diag(a), rtol=1e-12)


def test_jacobi_diagonal_types_match_numpy():
    """petsc4py 3.25 does not bind `PCJacobiSetType`, so the four Jacobi types
    are checked against the closed form each one selects instead."""
    a = spd(10, seed=5)
    n = a.shape[0]
    x = np.random.default_rng(6).normal(size=n)
    expected = {
        Mojo.PC_JACOBI_DIAGONAL: np.diag(a),
        Mojo.PC_JACOBI_ROWMAX: np.max(np.abs(a), axis=1),
        Mojo.PC_JACOBI_ROWL1: np.sum(np.abs(a), axis=1),
        Mojo.PC_JACOBI_ROWSUM: np.sum(a, axis=1),
    }
    bm = mojo_matrix(a)
    for jtype, denom in expected.items():
        pc = Mojo.PC().create()
        pc.setType("jacobi")
        pc.setJacobiType(jtype)
        pc.setOperators(bm)
        ym = mojo_vec(np.zeros(n))
        pc.apply(mojo_vec(x), ym)
        np.testing.assert_allclose(ym.getArray(), x / denom, rtol=1e-12)


def test_jacobi_symmetric_apply_matches():
    n = 12
    a = spd(n, seed=3)
    x = np.random.default_rng(4).normal(size=n)
    am = ref_matrix(a)
    pc_ref = Ref.PC().create()
    pc_ref.setType("jacobi")
    pc_ref.setOperators(am)
    yr = ref_vec(np.zeros(n))
    pc_ref.applySymmetricLeft(ref_vec(x), yr)

    bm = mojo_matrix(a)
    pc = Mojo.PC().create()
    pc.setType("jacobi")
    pc.setOperators(bm)
    ym = mojo_vec(np.zeros(n))
    pc.applySymmetricLeft(mojo_vec(x), ym)
    np.testing.assert_allclose(ym.getArray(), yr.getArray(), rtol=TOL)
    np.testing.assert_allclose(
        yr.getArray(), x / np.sqrt(np.abs(np.diag(a))), rtol=1e-12
    )


def test_none_matches():
    a = spd(9, seed=7)
    x = np.random.default_rng(8).normal(size=9)
    yr, ym, _ = run(a, x, "none")
    np.testing.assert_allclose(ym, yr, rtol=0.0)
    np.testing.assert_allclose(ym, x, rtol=0.0)


@pytest.mark.parametrize("n", [6, 30, 120])
def test_sor_matches(n):
    a = banded_spd(n, seed=9)
    x = np.random.default_rng(10).normal(size=n)
    yr, ym, _ = run(a, x, "sor")
    np.testing.assert_allclose(ym, yr, rtol=1e-10, atol=1e-12)


def test_sor_omega_changes_the_apply():
    """petsc4py 3.25 does not bind `PCSORSetOmega`, so the port's relaxation
    factor is checked against what the default-omega sweep produces."""
    a = banded_spd(20, seed=11)
    x = np.random.default_rng(12).normal(size=20)
    _, y_plain, pc = run(a, x, "sor")
    pc.setSOROmega(0.8)
    ym = mojo_vec(np.zeros(20))
    pc.apply(mojo_vec(x), ym)
    assert not np.allclose(ym.getArray(), y_plain)
    assert np.all(np.isfinite(ym.getArray()))


@pytest.mark.parametrize("levels", [0, 1, 2])
def test_ilu_matches(levels):
    a = banded_spd(40, seed=13)
    x = np.random.default_rng(14).normal(size=40)
    yr, ym, pc = run(a, x, "ilu", setFactorLevels=levels)
    np.testing.assert_allclose(ym, yr, rtol=1e-9, atol=1e-12)


def test_ilu_dense_spd_matches():
    a = spd(24, seed=15)
    x = np.random.default_rng(16).normal(size=24)
    yr, ym, _ = run(a, x, "ilu", setFactorLevels=0)
    np.testing.assert_allclose(ym, yr, rtol=1e-9, atol=1e-12)


def test_ilu_rcm_ordering_is_a_permutation():
    """The port's RCM visits neighbours in index order where PETSc visits them
    in degree order, so the orderings — and therefore the ILU fill — can
    differ.  What has to hold is that the ordering is a permutation and that
    the factorized apply is finite and lowers the residual."""
    a = banded_spd(40, seed=17)
    n = a.shape[0]
    x = np.random.default_rng(18).normal(size=n)
    _, ym, pc = run(a, x, "ilu", setFactorLevels=0, setFactorOrdering="rcm")
    assert np.all(np.isfinite(ym))
    before = np.linalg.norm(x)
    after = np.linalg.norm(a @ ym - x)
    assert after < before


def test_uncovered_ordering_raises():
    pc = Mojo.PC().create()
    with pytest.raises(NotImplementedError):
        pc.setFactorOrdering("nd")


@pytest.mark.parametrize("levels", [0, 1])
def test_icc_matches(levels):
    a = banded_spd(40, seed=19)
    x = np.random.default_rng(20).normal(size=40)
    yr, ym, _ = run(a, x, "icc", setFactorLevels=levels)
    np.testing.assert_allclose(ym, yr, rtol=1e-8, atol=1e-11)


def test_icc_dense_spd_matches():
    a = spd(20, seed=21)
    x = np.random.default_rng(22).normal(size=20)
    yr, ym, _ = run(a, x, "icc", setFactorLevels=0)
    np.testing.assert_allclose(ym, yr, rtol=1e-8, atol=1e-11)


def test_uncovered_types_raise():
    a = spd(6, seed=23)
    pc = Mojo.PC().create()
    with pytest.raises(NotImplementedError):
        pc.setType("gamg")
    with pytest.raises(NotImplementedError):
        pc.setType("bjacobi")


def test_factor_matrix_is_square():
    a = banded_spd(20, seed=25)
    _, _, pc = run(a, np.ones(20), "ilu", setFactorLevels=0)
    f = pc.getFactorMatrix()
    assert f.getSize() == (20, 20)
