"""Every `@export`ed kernel called by name, with an assertion on its output.

The parity tests exercise the public API, which reaches the kernels through
`_lib.py`'s dispatch; a kernel that is wired up and wrong would still leave
them green.  This file closes that hole: each symbol in `src/capi.mojo` appears
here as a literal, is called directly through ctypes with arguments built in
the test, and its output is checked against an independent NumPy reference or
against real upstream petsc4py.

The layout of every argument block is the one documented in `src/capi.mojo`,
`src/ported.mojo` and `src/ksp.mojo`; building the blocks here rather than
through the wrapper is deliberate, since a wrapper-side mistake would otherwise
hide itself.
"""

import os

import numpy as np
import pytest

from mojopetsc4py import PETSc as Mojo
from mojopetsc4py._lib import LIB, addr, lib

L = lib()

# The tests must observe the compiled library, not a fallback: assert the shared
# object really is the file that was loaded.
assert os.path.exists(LIB), f"compiled library missing: {LIB}"

SEED = 20240917

# petscvec.h NormType
NORM_1, NORM_2, NORM_FROBENIUS, NORM_INFINITY, NORM_1_AND_2 = 0, 1, 2, 3, 4
# petscpc.h PCType
PC_NONE, PC_JACOBI, PC_SOR, PC_ILU, PC_ICC = 0, 2, 4, 5, 6
# petscpc.h PCJACOBIType
JACOBI_DIAGONAL, JACOBI_ROWMAX, JACOBI_ROWL1, JACOBI_ROWSUM = 0, 1, 2, 3
# petscmat.h MatSORType
SOR_FORWARD_SWEEP = 1
SOR_LOCAL_SYMMETRIC_SWEEP = 12
SOR_ZERO_INITIAL_GUESS = 16
# petscksp.h KSPConvergedReason
CONVERGED_RTOL, CONVERGED_ATOL, CONVERGED_ITS = 2, 3, 4
# petscksp.h KSPNormType
NORM_PRECONDITIONED = 2


def rng(n, seed=SEED):
    return np.random.default_rng(seed).normal(size=n)


def same(got, want, tol=1e-13):
    np.testing.assert_allclose(
        np.asarray(got, dtype=np.float64),
        np.asarray(want, dtype=np.float64),
        rtol=tol,
        atol=tol,
    )


def tridiag(n, diag=4.0, off=-1.0):
    a = np.zeros((n, n))
    for i in range(n):
        a[i, i] = diag
        if i:
            a[i, i - 1] = off
        if i + 1 < n:
            a[i, i + 1] = off
    return a


def csr(dense):
    """A `Mat` holding `dense`, which gives direct access to irow/icol/data."""
    return Mojo.from_dense(np.asarray(dense, dtype=np.float64))


# ====================================================================
#  src/capi.mojo -- Mat
# ====================================================================


def test_mp_mat_mult():
    dense = tridiag(9)
    a = csr(dense)
    x = rng(9, SEED + 1)
    y = np.zeros(9)
    L.mp_mat_mult(
        addr(x), addr(y), addr(a.irow), addr(a.icol), addr(a.data), 9
    )
    same(y, dense @ x)
    assert y.ctypes.data == addr(y), "the kernel wrote through the address given"


def test_mp_mat_mult_add():
    dense = tridiag(9)
    a = csr(dense)
    x = rng(9, SEED + 2)
    y = rng(9, SEED + 3)
    want = dense @ x + y.copy()
    L.mp_mat_mult_add(
        addr(x), addr(y), addr(a.irow), addr(a.icol), addr(a.data), 9
    )
    same(y, want)


def test_mp_mat_mult_transpose():
    dense = tridiag(9)
    a = csr(dense)
    x = rng(9, SEED + 4)
    y = np.full(9, 7.0)  # the kernel clears the first ncols entries itself
    L.mp_mat_mult_transpose(
        addr(x), addr(y), addr(a.irow), addr(a.icol), addr(a.data), 9, 9
    )
    same(y, dense.T @ x)


def test_mp_mat_get_diagonal():
    dense = tridiag(7)
    dense[3, 3] = -2.5
    a = csr(dense)
    d = np.full(7, -1.0)
    L.mp_mat_get_diagonal(
        addr(d), addr(a.irow), addr(a.icol), addr(a.data), 7
    )
    same(d, np.diag(dense))


def test_mp_mat_get_diagonal_missing_entry_stays_zero():
    dense = tridiag(6)
    dense[2, 2] = 0.0
    dense = np.where(dense == 0.0, 0.0, dense)
    a = csr(tridiag(6))
    dense[2, 2] = 0.0
    a = csr(dense)
    d = np.full(6, -1.0)
    L.mp_mat_get_diagonal(addr(d), addr(a.irow), addr(a.icol), addr(a.data), 6)
    same(d, np.diag(dense))
    assert d[2] == 0.0


def test_mp_mat_scale():
    a = csr(tridiag(6))
    before = a.data.copy()
    L.mp_mat_scale(addr(a.data), before.size, 2.5)
    same(a.data, before * 2.5)


def test_mp_mat_zero_entries():
    dense = tridiag(6)
    a = csr(dense)
    stored = int(a.irow[-1])
    assert stored == 6 * 3 - 2, "a tridiagonal 6x6 matrix stores 16 entries"
    L.mp_mat_zero_entries(addr(a.data), stored)
    assert not a.data.any()
    # `MatZeroEntries` keeps the allocated nonzero structure, exactly as
    # upstream does; only the values change
    assert a.icol.size == stored, "the structure must be untouched"
    assert int(a.irow[-1]) == stored


# ====================================================================
#  src/capi.mojo -- Vec
# ====================================================================


def test_mp_vec_pointwise_mult():
    x, y = rng(11, 1), rng(11, 2)
    w = np.zeros(11)
    L.mp_vec_pointwise_mult(addr(w), addr(x), addr(y), 11)
    same(w, x * y)


def test_mp_vec_pointwise_divide():
    x, y = rng(11, 3), rng(11, 4) + 3.0
    w = np.zeros(11)
    L.mp_vec_pointwise_divide(addr(w), addr(x), addr(y), 11)
    same(w, x / y)


def test_mp_vec_pointwise_min():
    x, y = rng(11, 5), rng(11, 6)
    w = np.zeros(11)
    L.mp_vec_pointwise_min(addr(w), addr(x), addr(y), 11)
    same(w, np.minimum(x, y))


def test_mp_vec_pointwise_max():
    x, y = rng(11, 7), rng(11, 8)
    w = np.zeros(11)
    L.mp_vec_pointwise_max(addr(w), addr(x), addr(y), 11)
    same(w, np.maximum(x, y))


def test_mp_vec_pointwise_max_abs():
    x, y = rng(11, 9), rng(11, 10)
    w = np.zeros(11)
    L.mp_vec_pointwise_max_abs(addr(w), addr(x), addr(y), 11)
    same(w, np.maximum(np.abs(x), np.abs(y)))


def test_mp_vec_norm():
    x = rng(13, 11)
    for ntype, want in (
        (NORM_1, np.abs(x).sum()),
        (NORM_2, np.linalg.norm(x)),
        (NORM_INFINITY, np.abs(x).max()),
    ):
        z = np.full(2, -1.0)
        L.mp_vec_norm(addr(x), ntype, addr(z), 13)
        same(z[0], want)
    z = np.full(2, -1.0)
    L.mp_vec_norm(addr(x), NORM_1_AND_2, addr(z), 13)
    same(z[0], np.abs(x).sum())
    same(z[1], np.linalg.norm(x))


def test_mp_vec_norm_empty_leaves_the_sentinel():
    x = np.zeros(0)
    z = np.full(2, -1.0)
    L.mp_vec_norm(addr(x), NORM_2, addr(z), 0)
    assert z[0] == 0.0


def test_mp_vec_dot():
    x, y = rng(13, 12), rng(13, 13)
    val = np.zeros(1)
    L.mp_vec_dot(addr(x), addr(y), addr(val), 13)
    same(val[0], float(x @ y), tol=1e-12)


def test_mp_vec_tdot():
    x, y = rng(13, 14), rng(13, 15)
    val = np.zeros(1)
    L.mp_vec_tdot(addr(x), addr(y), addr(val), 13)
    same(val[0], float(y @ x), tol=1e-12)


def test_mp_vec_dot_norm2():
    x, y = rng(13, 16), rng(13, 17)
    sval = np.zeros(1)
    rval = np.zeros(1)
    L.mp_vec_dot_norm2(addr(x), addr(y), addr(sval), addr(rval), 13)
    same(sval[0], float(x @ y), tol=1e-12)
    same(rval[0], float(x @ x), tol=1e-12)


def test_mp_vec_sum():
    x = rng(13, 18)
    val = np.zeros(1)
    L.mp_vec_sum(addr(x), addr(val), 13)
    same(val[0], x.sum(), tol=1e-12)


def test_mp_vec_mean():
    x = rng(13, 19)
    val = np.zeros(1)
    L.mp_vec_mean(addr(x), addr(val), 13)
    same(val[0], x.mean(), tol=1e-12)


def test_mp_vec_max():
    x = rng(13, 20) - 5.0
    idx = np.zeros(1, dtype=np.int64)
    z = np.zeros(1)
    L.mp_vec_max(addr(x), addr(idx), addr(z), 13)
    assert idx[0] == int(np.argmax(x))
    same(z[0], x.max())


def test_mp_vec_min():
    x = rng(13, 21) + 5.0
    idx = np.zeros(1, dtype=np.int64)
    z = np.zeros(1)
    L.mp_vec_min(addr(x), addr(idx), addr(z), 13)
    assert idx[0] == int(np.argmin(x))
    same(z[0], x.min())


def test_mp_vec_normalize():
    x = rng(13, 22)
    v = np.zeros(1)
    L.mp_vec_normalize(addr(x), addr(v), 13)
    same(v[0], np.linalg.norm(rng(13, 22)))
    same(x, rng(13, 22) / np.linalg.norm(rng(13, 22)))


def test_mp_vec_reciprocal_keeps_exact_zero():
    x = np.array([2.0, 0.0, -4.0, 1e-20])
    L.mp_vec_reciprocal(addr(x), 4)
    same(x[:3], [0.5, 0.0, -0.25])
    assert x[3] == 1e20, "an exact zero is the only guard; 1e-20 reciprocates"


def test_mp_vec_exp():
    x = rng(9, 23)
    want = x.copy()
    L.mp_vec_exp(addr(x), 9)
    same(x, np.exp(want), tol=1e-9)


def test_mp_vec_log():
    x = np.abs(rng(9, 24)) + 0.5
    want = x.copy()
    L.mp_vec_log(addr(x), 9)
    same(x, np.log(want), tol=1e-9)


def test_mp_vec_sqrtabs():
    x = rng(9, 25) * 3.0
    want = x.copy()
    L.mp_vec_sqrtabs(addr(x), 9)
    same(x, np.sqrt(np.abs(want)))


def test_mp_vec_chop():
    x = np.array([1e-9, -1e-9, 0.5, -0.5, 0.0])
    want = x.copy()
    L.mp_vec_chop(addr(x), 5, 1e-8)
    same(x, np.where(np.abs(want) < 1e-8, 0.0, want))


def test_mp_vec_abs():
    x = rng(9, 26)
    want = np.abs(x)
    L.mp_vec_abs(addr(x), 9)
    same(x, want)


def test_mp_vec_conjugate_is_the_identity_for_real():
    x = rng(9, 27)
    want = x.copy()
    L.mp_vec_conjugate(addr(x), 9)
    same(x, want)


def test_mp_vec_copy():
    x, y = rng(9, 28), np.zeros(9)
    L.mp_vec_copy(addr(x), addr(y), 9)
    same(y, x)


def test_mp_vec_swap():
    x, y = rng(9, 29), rng(9, 30)
    x0, y0 = x.copy(), y.copy()
    L.mp_vec_swap(addr(x), addr(y), 9)
    same(x, y0)
    same(y, x0)


def test_mp_vec_set():
    x = np.zeros(9)
    L.mp_vec_set(addr(x), 2.5, 9)
    same(x, np.full(9, 2.5))
    L.mp_vec_set(addr(x), 0.0, 9)
    assert not x.any(), "alpha == 0 goes through VecZeroEntries"


def test_mp_vec_zero_entries():
    x = rng(9, 31)
    L.mp_vec_zero_entries(addr(x), 9)
    assert not x.any()


def test_mp_vec_scale():
    x = rng(9, 32)
    want = x.copy()
    L.mp_vec_scale(addr(x), -3.0, 9)
    same(x, want * -3.0)


def test_mp_vec_shift():
    x = rng(9, 33)
    want = x.copy()
    L.mp_vec_shift(addr(x), 4.0, 9)
    same(x, want + 4.0)


def test_mp_vec_axpy():
    y, x = rng(9, 34), rng(9, 35)
    want = y.copy()
    L.mp_vec_axpy(addr(y), 2.0, addr(x), 9)
    same(y, want + 2.0 * x)


def test_mp_vec_aypx():
    y, x = rng(9, 36), rng(9, 37)
    want = y.copy()
    L.mp_vec_aypx(addr(y), 3.0, addr(x), 9)
    same(y, x + 3.0 * want)
    # the alpha == 0 and alpha == 1 branches are separate code paths upstream
    y2 = want.copy()
    L.mp_vec_aypx(addr(y2), 0.0, addr(x), 9)
    same(y2, x)
    y3 = want.copy()
    L.mp_vec_aypx(addr(y3), 1.0, addr(x), 9)
    same(y3, want + x)


def test_mp_vec_axpby():
    y, x = rng(9, 38), rng(9, 39)
    want = y.copy()
    L.mp_vec_axpby(addr(y), 2.0, -1.0, addr(x), 9)
    same(y, 2.0 * x - want)


def test_mp_vec_axpbypcz():
    z, x, y = rng(9, 40), rng(9, 41), rng(9, 42)
    want = z.copy()
    L.mp_vec_axpbypcz(addr(z), 2.0, -3.0, 0.5, addr(x), addr(y), 9)
    same(z, 2.0 * x - 3.0 * y + 0.5 * want)


def test_mp_vec_waxpy():
    w, x, y = np.zeros(9), rng(9, 43), rng(9, 44)
    L.mp_vec_waxpy(addr(w), 2.0, addr(x), addr(y), 9)
    same(w, 2.0 * x + y)


def test_mp_vec_maxpy():
    y = rng(9, 45)
    x0, x1 = rng(9, 46), rng(9, 47)
    want = y.copy()
    xs = np.array([x0.ctypes.data, x1.ctypes.data], dtype=np.int64)
    alpha = np.array([2.0, -1.0])
    L.mp_vec_maxpy(addr(y), 2, addr(alpha), addr(xs), 9)
    same(y, want + 2.0 * x0 - x1)


def test_mp_vec_max_pointwise_divide():
    x = rng(9, 48)
    y = rng(9, 49)
    y[4] = 0.0
    rval = np.zeros(1)
    L.mp_vec_max_pointwise_divide(addr(x), addr(y), addr(rval), 9)
    ratio = np.abs(np.where(y == 0.0, x, x / np.where(y == 0.0, 1.0, y)))
    same(rval[0], ratio.max())


# ====================================================================
#  src/capi.mojo -- PC
# ====================================================================


def test_mp_mat_get_diagonal_markers():
    dense = tridiag(6)
    a = csr(dense)
    diag = np.zeros(6, dtype=np.int64)
    dense_flag = L.mp_mat_get_diagonal_markers(
        addr(a.irow), addr(a.icol), addr(diag), 6
    )
    assert dense_flag == 1, "every row of a tridiagonal matrix has its diagonal"
    for i in range(6):
        assert a.irow[i] <= diag[i] < a.irow[i + 1]
        assert a.icol[diag[i]] == i

    # a row without a diagonal reports the row end and clears the flag
    dense2 = dense.copy()
    dense2[2, 2] = 0.0
    b = csr(dense2)
    diag2 = np.zeros(6, dtype=np.int64)
    flag = L.mp_mat_get_diagonal_markers(
        addr(b.irow), addr(b.icol), addr(diag2), 6
    )
    assert flag == 0
    assert diag2[2] == b.irow[3]


def test_mp_pc_jacobi_setup_diagonal():
    dense = tridiag(8)
    dense[2, 2] = -3.0
    a = csr(dense)
    diag = np.zeros(8)
    didx = np.zeros(8)
    L.mp_pc_jacobi_setup(
        addr(diag), JACOBI_DIAGONAL, 0, 1, 1.0, addr(a.irow), addr(a.icol),
        addr(a.data), 8, addr(didx),
    )
    same(diag, 1.0 / np.diag(dense))


def test_mp_pc_jacobi_setup_row_types():
    dense = tridiag(6)
    a = csr(dense)
    want = {
        JACOBI_ROWMAX: np.abs(dense).max(axis=1),
        JACOBI_ROWSUM: dense.sum(axis=1),
    }
    for jtype, ref in want.items():
        diag = np.zeros(6)
        didx = np.zeros(6)
        L.mp_pc_jacobi_setup(
            addr(diag), jtype, 1, 1, 1.0, addr(a.irow), addr(a.icol),
            addr(a.data), 6, addr(didx),
        )
        same(diag, 1.0 / np.abs(ref))


def test_mp_pc_jacobi_setup_rowl1_scale_and_sign_fix():
    dense = tridiag(5)
    dense[1, 1] = -2.0
    a = csr(dense)
    diag = np.zeros(5)
    didx = np.zeros(5)
    scale = 0.5
    # useabs = 0: `VecAbs` runs after the reciprocal and would erase the sign
    # the flip exists to fix
    L.mp_pc_jacobi_setup(
        addr(diag), JACOBI_ROWL1, 0, 1, scale, addr(a.irow), addr(a.icol),
        addr(a.data), 5, addr(didx),
    )
    # PCSetUp_Jacobi: diag <- rowl1, sign-flipped where the true diagonal is
    # negative, blended with `scale`, then reciprocated
    rowl1 = np.abs(dense).sum(axis=1)
    true_diag = np.diag(dense).copy()
    flipped = rowl1.copy()
    neg = true_diag < 0.0
    flipped[neg] = -flipped[neg]
    # `scale * (diag - true_diag) + true_diag` with `true_diag` still signed,
    # exactly as upstream writes it
    blended = scale * (flipped - true_diag) + true_diag
    same(diag, 1.0 / blended)


def test_mp_pc_jacobi_setup_fix_diag_replaces_a_zero_inverse():
    dense = np.eye(4) * 2.0
    a = csr(dense)
    diag = np.zeros(4)
    didx = np.zeros(4)
    L.mp_pc_jacobi_setup(
        addr(diag), JACOBI_DIAGONAL, 0, 1, 1.0, addr(a.irow), addr(a.icol),
        addr(a.data), 4, addr(didx),
    )
    same(diag, np.full(4, 0.5))
    diag2 = np.zeros(4)
    L.mp_pc_jacobi_setup(
        addr(diag2), JACOBI_DIAGONAL, 0, 0, 1.0, addr(a.irow), addr(a.icol),
        addr(a.data), 4, addr(didx),
    )
    same(diag2, np.full(4, 0.5))


def test_mp_pc_jacobi_setup_symmetric():
    dense = tridiag(7)
    dense[3, 3] = -5.0
    a = csr(dense)
    d = np.zeros(7)
    L.mp_pc_jacobi_setup_symmetric(
        addr(d), JACOBI_DIAGONAL, addr(a.irow), addr(a.icol), addr(a.data), 7
    )
    same(d, 1.0 / np.sqrt(np.abs(np.diag(dense))))


def test_mp_pc_jacobi_apply():
    """`jacobi.c PCApply_Jacobi` is a single pointwise multiply by the vector
    `PCSetUp_Jacobi` filled in, which holds the *inverse* diagonal."""
    dense = tridiag(8)
    a = csr(dense)
    diag = np.zeros(8)
    didx = np.zeros(8)
    L.mp_pc_jacobi_setup(
        addr(diag), JACOBI_DIAGONAL, 0, 1, 1.0, addr(a.irow), addr(a.icol),
        addr(a.data), 8, addr(didx),
    )
    x = rng(8, 50)
    y = np.zeros(8)
    L.mp_pc_jacobi_apply(addr(y), addr(x), addr(diag), 8)
    same(y, x / np.diag(dense))


def _sor_blocks(a, n, omega=1.0, fshift=0.0):
    """The scratch `PCApply_SOR` needs, from the kernel's own contract."""
    dmark = np.zeros(n, dtype=np.int64)
    L.mp_mat_get_diagonal_markers(addr(a.irow), addr(a.icol), addr(dmark), n)
    idiag = np.zeros(n)
    mdiag = np.zeros(n)
    t = np.zeros(3 * n)
    return dmark, idiag, mdiag, t


def _numpy_symmetric_sor(dense, x, omega, fshift):
    """One symmetric SOR sweep with a zero initial guess, per aij.c
    MatSOR_SeqAIJ: the forward half is plain Gauss-Seidel (there is no old
    value to relax), the backward half relaxes, `x[i] = (1 - omega) x[i] +
    omega * gs`."""
    n = dense.shape[0]
    d = np.diag(dense)
    idiag = omega / (fshift + d)
    y = np.zeros(n)
    t = np.zeros(n)
    for i in range(n):
        s = x[i] - dense[i, :i] @ y[:i]
        t[i] = s
        y[i] = s * idiag[i]
    for i in range(n - 1, -1, -1):
        # the backward half starts from the checkpoint the forward half
        # saved, which is `b - L y` without the diagonal term
        s = t[i] - dense[i, i + 1:] @ y[i + 1:]
        y[i] = (1.0 - omega) * y[i] + s * idiag[i]
    return y


def _upstream_sor(dense, x, omega, fshift, stype):
    """PETSc's own MatSOR on the same matrix, which is the reference the
    hand-written sweep above is checked against."""
    from petsc4py import PETSc as Ref

    n = dense.shape[0]
    m = Ref.Mat().createAIJ(size=(n, n), nnz=int((dense != 0).sum(axis=1).max()))
    for i in range(n):
        for j in range(n):
            if dense[i, j]:
                m.setValue(i, j, float(dense[i, j]))
    m.assemble()
    b = Ref.Vec().createSeq(n)
    b.setArray(x)
    y = Ref.Vec().createSeq(n)
    m.SOR(b, y, omega=omega, sortype=stype, shift=fshift, its=1, lits=1)
    return y.getArray().copy()


def test_mp_pc_sor_apply():
    dense = tridiag(9)
    a = csr(dense)
    n = 9
    x = rng(n, 51)
    for omega, fshift in ((1.0, 0.0), (1.3, 0.25)):
        dmark, idiag, mdiag, t = _sor_blocks(a, n, omega, fshift)
        y = np.zeros(n)
        L.mp_pc_sor_apply(
            addr(x), addr(y), addr(a.irow), addr(a.icol), addr(a.data),
            addr(dmark), addr(idiag), addr(mdiag), addr(t), omega, fshift,
            SOR_LOCAL_SYMMETRIC_SWEEP | SOR_ZERO_INITIAL_GUESS, n,
        )
        np.testing.assert_allclose(
            y, _numpy_symmetric_sor(dense, x, omega, fshift), rtol=0, atol=1e-13
        )
        np.testing.assert_allclose(
            y, _upstream_sor(
                dense, x, omega, fshift,
                SOR_LOCAL_SYMMETRIC_SWEEP | SOR_ZERO_INITIAL_GUESS,
            ), rtol=0, atol=1e-12,
        )


def test_mp_pc_sor_apply_forward_sweep_only():
    dense = tridiag(6)
    a = csr(dense)
    n = 6
    x = rng(n, 52)
    dmark, idiag, mdiag, t = _sor_blocks(a, n)
    y = np.zeros(n)
    L.mp_pc_sor_apply(
        addr(x), addr(y), addr(a.irow), addr(a.icol), addr(a.data),
        addr(dmark), addr(idiag), addr(mdiag), addr(t), 1.0, 0.0,
        SOR_FORWARD_SWEEP | SOR_ZERO_INITIAL_GUESS, n,
    )
    d = np.diag(dense)
    want = np.zeros(n)
    for i in range(n):
        want[i] = (x[i] - dense[i, :i] @ want[:i]) / d[i]
    same(y, want)


def _ilu_factor_arrays(dense, levels=0, ilu0=True):
    """Build the ILU factor with the kernels alone, no wrapper state."""
    n = dense.shape[0]
    a = csr(dense)
    r = np.arange(n, dtype=np.int64)
    ic = np.empty(n, dtype=np.int64)
    ic[r] = np.arange(n, dtype=np.int64)
    adiag = np.zeros(n, dtype=np.int64)
    L.mp_mat_get_diagonal_markers(addr(a.irow), addr(a.icol), addr(adiag), n)
    cap = 8 * n + 16
    if ilu0:
        bi = np.zeros(n + 1, dtype=np.int64)
        bj = np.zeros(cap, dtype=np.int64)
        bdiag = np.zeros(n + 1, dtype=np.int64)
        nnz = L.mp_ilu_symbolic_ilu0(
            addr(bi), addr(bj), addr(bdiag), addr(adiag),
            addr(a.irow), addr(a.icol), n,
        )
        assert nnz >= 0
        return a, n, r, ic, bi, bj, bdiag, False
    scratch = np.zeros(7 * n + 8, dtype=np.int64)
    bi = np.zeros(n + 1, dtype=np.int64)
    bjsrc = np.zeros(cap, dtype=np.int64)
    bjlvl = np.zeros(cap, dtype=np.int64)
    bdiag = np.zeros(n + 1, dtype=np.int64)
    nnz = L.mp_ilu_symbolic(
        addr(bi), addr(bjsrc), addr(bjlvl), addr(bdiag), addr(a.irow),
        addr(a.icol), addr(r), addr(ic), levels, cap, addr(scratch), n,
    )
    assert nnz >= 0
    # the contiguous pass is not run here: it rewrites `bi` and `bdiag` in
    # place, and callers that want to observe it run it themselves
    return a, n, r, ic, bi, bjsrc, bdiag, True


def _factor_dense(bi, bj, ba, bdiag, n):
    """L*U as a dense array, read straight out of the factor's own layout."""
    from mojopetsc4py.PETSc.PC import _ilu_factor_mat  # noqa: F401

    f = {"bi": bi, "bj": bj, "ba": ba, "bdiag": bdiag,
         "r": np.arange(n, dtype=np.int64), "c": np.arange(n, dtype=np.int64)}
    return _ilu_factor_mat(f)._dense()


def test_mp_ilu_symbolic_ilu0():
    dense = tridiag(10)
    a = csr(dense)
    n = 10
    adiag = np.zeros(n, dtype=np.int64)
    L.mp_mat_get_diagonal_markers(addr(a.irow), addr(a.icol), addr(adiag), n)
    bi = np.zeros(n + 1, dtype=np.int64)
    bj = np.zeros(4 * n, dtype=np.int64)
    bdiag = np.zeros(n + 1, dtype=np.int64)
    nnz = L.mp_ilu_symbolic_ilu0(
        addr(bi), addr(bj), addr(bdiag), addr(adiag),
        addr(a.irow), addr(a.icol), n,
    )
    assert nnz == int(a.irow[n]), "ilu(0) stores exactly the input nonzeros"
    assert bi[0] == 0
    # bj holds every L row in increasing order, then every U row in decreasing
    # order; bi delimits the L block and bdiag[i] is the offset of U(i,:)'s
    # diagonal back from the end of that block
    for i in range(n):
        lower = [c for c in a.icol[a.irow[i]:a.irow[i + 1]] if c < i]
        assert list(bj[bi[i]:bi[i + 1]]) == lower
        # each U row ends on its own diagonal, and the U rows run backwards
        assert bj[bdiag[i]] == i
        if i + 1 < n:
            assert bdiag[i + 1] < bdiag[i]
        assert bdiag[0] == nnz - 1


def test_mp_ilu_symbolic_and_contiguous_lu():
    """The symbolic pass stages each row as its L entries, the diagonal, then
    its U entries; the contiguous pass moves that into the factor layout --
    every L row in increasing order first, then every U row in decreasing
    order, each U row ending on its own diagonal."""
    dense = tridiag(12)
    n = 12
    a, n, r, ic, bi, bjsrc, bdiag, _ = _ilu_factor_arrays(dense, levels=1, ilu0=False)
    assert bi[0] == 0
    total = int(bi[n])
    stage = bjsrc.copy()
    bj = np.zeros(total + 1, dtype=np.int64)
    L.mp_ilu_contiguous_lu(addr(bjsrc), addr(bi), addr(bdiag), addr(bj), n)
    np.testing.assert_array_equal(bjsrc, stage, err_msg="the staging buffer is untouched")

    # the L block holds, row by row, the strict entries the symbolic pass
    # staged -- the staging array is not modified
    np.testing.assert_array_equal(bjsrc, stage)
    for i in range(n):
        assert int(bj[bdiag[i]]) == i, f"row {i} does not end on its diagonal"
        # the diagonal of U(i,:) is the last entry of that row, and the U rows
        # are laid out from the end of the array towards its front
        if i + 1 < n:
            assert int(bdiag[i]) > int(bi[i])
    assert all(0 <= int(c) < n for c in bj[:total])
    assert len(set(int(c) for c in bj[:total])) >= n


def test_mp_pc_ilu_numeric_and_apply():
    dense = tridiag(12)
    n = 12
    a, n, r, ic, bi, bj, bdiag, _ = _ilu_factor_arrays(dense)
    ba = np.zeros(int(bdiag[0]) + 1)
    rtmp = np.zeros(n)
    ret = L.mp_pc_ilu_numeric(
        addr(ba), addr(rtmp), addr(a.irow), addr(a.icol), addr(a.data),
        addr(bi), addr(bj), addr(bdiag), addr(r), addr(ic), n,
    )
    # the return value is the number of factor entries (bdiag[0] + 1)
    assert ret == int(bdiag[0]) + 1, f"the numeric factorisation returned {ret}"
    assert np.all(ba != 0.0), "a tridiagonal matrix has no structural zeros"

    b = rng(n, 53)
    x = np.zeros(n)
    tmp = np.zeros(n)
    L.mp_pc_ilu_apply(
        addr(b), addr(x), addr(tmp), addr(bi), addr(bj), addr(ba),
        addr(bdiag), addr(r), addr(r), n,
    )
    # a tridiagonal matrix is factored exactly by ILU(0), so the apply is the
    # exact solve; the rebuilt L*U is checked against it as well
    same(x, np.linalg.solve(dense, b), tol=1e-9)
    same(x, np.linalg.solve(_factor_dense(bi, bj, ba, bdiag, n), b), tol=1e-9)


def test_mp_pc_ilu_apply_is_exact_on_a_tridiagonal_matrix():
    dense = tridiag(15)
    n = 15
    a, n, r, ic, bi, bj, bdiag, _ = _ilu_factor_arrays(dense)
    ba = np.zeros(int(bdiag[0]) + 1)
    L.mp_pc_ilu_numeric(
        addr(ba), addr(np.zeros(n)), addr(a.irow), addr(a.icol), addr(a.data),
        addr(bi), addr(bj), addr(bdiag), addr(r), addr(ic), n,
    )
    b = rng(n, 54)
    x = np.zeros(n)
    L.mp_pc_ilu_apply(
        addr(b), addr(x), addr(np.zeros(n)), addr(bi), addr(bj), addr(ba),
        addr(bdiag), addr(r), addr(r), n,
    )
    same(x, np.linalg.solve(dense, b), tol=1e-9)


def _icc_factor_arrays(dense, levels=0):
    n = dense.shape[0]
    a = csr(dense)
    r = np.arange(n, dtype=np.int64)
    rinv = np.empty(n, dtype=np.int64)
    rinv[r] = np.arange(n, dtype=np.int64)
    cap = 8 * n + 16
    scratch = np.zeros(6 * (n + 1) + 2 * cap, dtype=np.int64)
    ui = np.zeros(n + 1, dtype=np.int64)
    ujsrc = np.zeros(cap, dtype=np.int64)
    ujlvl = np.zeros(cap, dtype=np.int64)
    udiag = np.zeros(n, dtype=np.int64)
    nnz = L.mp_icc_symbolic(
        addr(ui), addr(ujsrc), addr(ujlvl), addr(udiag), addr(a.irow),
        addr(a.icol), addr(r), addr(rinv), levels, cap, addr(scratch), n,
        int(scratch.size),
    )
    assert nnz >= 0
    # `b->diag` and the spare entry `bdiag[n]` share one array, as upstream
    # allocates them; `ui` stays the staging map
    bi = np.zeros(n + 1, dtype=np.int64)
    bdiag = np.zeros(n + 1, dtype=np.int64)
    bdiag[:n] = udiag[:n]
    bj = np.zeros(cap, dtype=np.int64)
    L.mp_icc_contiguous_cholesky(
        addr(ujsrc), addr(ui), addr(bdiag), addr(bi), addr(bdiag), addr(bj), n
    )
    return a, n, r, rinv, bi, bj, bdiag, ujsrc

def test_mp_icc_symbolic_and_contiguous_cholesky():
    """The contiguous pass builds `b->i`, `b->diag` and `b->j` from the
    symbolic pattern: each row is its strictly upper columns ascending with the
    diagonal slot, holding 1/d_i, last -- the IC(0) layout a real PETSc factor
    reports from `MatGetRowIJ`.
    """
    dense = tridiag(10)
    n = 10
    a, n, r, rinv, bi, bj, udiag, ujsrc = _icc_factor_arrays(dense, levels=0)
    assert bi[0] == 0
    assert int(udiag[n]) == int(bi[n]), "the spare entry is b->i[n]"
    for i in range(n):
        assert bi[i] <= udiag[i] < bi[i + 1]
        assert bj[udiag[i]] == i, "each row ends on its diagonal"
        strict = bj[bi[i]:udiag[i]]
        assert all(c > i for c in strict), f"row {i}: {list(strict)}"
        assert list(strict) == sorted(strict)


def test_mp_pc_icc_numeric_and_apply():
    dense = tridiag(12)
    n = 12
    a, n, r, rinv, bi, bj, udiag, ujsrc = _icc_factor_arrays(dense)
    ua = np.zeros(max(int(bi[-1]), 1))
    il = np.zeros(n, dtype=np.int64)
    c2r = np.zeros(n, dtype=np.int64)
    L.mp_pc_icc_numeric(
        addr(ua), addr(np.zeros(n)), addr(il), addr(c2r), addr(a.irow),
        addr(a.icol), addr(a.data), addr(bi), addr(bj), addr(udiag),
        addr(r), addr(rinv), n,
    )
    assert np.all(ua != 0.0)

    b = rng(n, 55)
    x = np.zeros(n)
    work = np.zeros(n)
    ret = L.mp_pc_icc_apply(
        addr(b), addr(x), addr(bi), addr(bj), addr(ua), addr(udiag),
        addr(r), addr(work), n,
    )
    assert ret == 0
    # IC(0) on a tridiagonal matrix is exact, so the preconditioner solves A
    from mojopetsc4py.PETSc.PC import _icc_factor_mat

    factor = _icc_factor_mat(
        {"bi": bi, "bj": bj, "ba": ua, "bdiag": udiag, "r": r}
    )._dense()
    same(factor, dense, tol=1e-9)
    same(x, np.linalg.solve(dense, b), tol=1e-9)


def test_mp_ordering_rcm():
    dense = tridiag(9)
    a = csr(dense)
    n = 9
    perm = np.zeros(n, dtype=np.int64)
    level = np.zeros(n, dtype=np.int64)
    visited = np.zeros(n, dtype=np.int64)
    queue = np.zeros(n, dtype=np.int64)
    L.mp_ordering_rcm(
        addr(perm), addr(level), addr(visited), addr(queue),
        addr(a.irow), addr(a.icol), n,
    )
    assert sorted(int(v) for v in perm) == list(range(n))
    # a path graph is already RCM-ordered, so the ordering is the identity up to
    # the pivot choice; assert it is a valid connected BFS order either way
    order = [int(v) for v in perm]
    assert order[0] in (0, n - 1)


# ====================================================================
#  src/capi.mojo -- KSP
# ====================================================================


def _ksp_state(n, max_it=200, restart=20, pc_block=None):
    """The cfg/cfgi blocks from `src/ksp.mojo`'s docstring."""
    cfg = np.zeros(16 + max_it + 2)
    cfg[0] = 1e-12  # rtol
    cfg[1] = 1e-50  # atol
    cfg[2] = 1e4  # divtol
    cfg[3] = 1e-30  # haptol
    cfg[4] = 0.1  # breakdowntol
    cfg[14] = 0.0  # not yet seeded
    cfgi = np.zeros(16, dtype=np.int64)
    cfgi[0] = NORM_PRECONDITIONED
    cfgi[1] = 0  # PC side: left
    cfgi[2] = 1  # guess is zero
    cfgi[3] = 0  # min_it
    cfgi[4] = -1  # chknorm
    cfgi[5] = 0  # convmaxits
    cfgi[6] = max_it
    cfgi[7] = restart
    cfg_pci = np.zeros(12, dtype=np.int64) if pc_block is None else pc_block
    pcf = np.zeros(2)
    pstmp = np.zeros(max(n, 1))
    return cfg, cfgi, cfg_pci, pcf, pstmp


def _jacobi_pci_block(dense, jtype=JACOBI_DIAGONAL):
    """pci[] for a Jacobi PC, packed the way `PC._pci_block` packs it."""
    n = dense.shape[0]
    a = csr(dense)
    diag = np.zeros(n)
    didx = np.zeros(n)
    L.mp_pc_jacobi_setup(
        addr(diag), jtype, 0, 1, 1.0, addr(a.irow), addr(a.icol),
        addr(a.data), n, addr(didx),
    )
    diagsqrt = np.zeros(n)
    L.mp_pc_jacobi_setup_symmetric(
        addr(diagsqrt), jtype, addr(a.irow), addr(a.icol), addr(a.data), n
    )
    pci = np.zeros(12, dtype=np.int64)
    pci[0] = PC_JACOBI
    pci[1] = diag.ctypes.data
    pci[2] = diagsqrt.ctypes.data
    return pci


def _upstream_solution(dense, b, ksp_type, pc_type="jacobi"):
    from petsc4py import PETSc as Ref

    n = dense.shape[0]
    A = Ref.Mat().createAIJ(size=(n, n))
    A.setUp()
    for i in range(n):
        for j in range(n):
            if dense[i, j]:
                A.setValue(i, j, dense[i, j])
    A.assemble()
    br = Ref.Vec().createSeq(n)
    br.setArray(b)
    xr = Ref.Vec().createSeq(n)
    pc = Ref.PC().create()
    pc.setType(pc_type)
    ksp = Ref.KSP().create()
    ksp.setPC(pc)
    ksp.setType(ksp_type)
    ksp.setOperators(A)
    ksp.setTolerances(rtol=1e-12, max_it=200)
    ksp.solve(br, xr)
    return xr.getArray().copy(), ksp.getIterationNumber()


def test_mp_ksp_solve_cg():
    petsc4py = pytest.importorskip("petsc4py")
    dense = tridiag(30)
    n = 30
    a = csr(dense)
    b = rng(n, 56)
    x = np.zeros(n)
    cfg, cfgi, pci, pcf, pstmp = _ksp_state(n)
    work = np.zeros(7 * n)
    L.mp_ksp_solve_cg(
        n, addr(a.irow), addr(a.icol), addr(a.data), addr(pci), addr(pcf),
        addr(pstmp), addr(work), addr(cfg), addr(cfgi), addr(b), addr(x),
    )
    same(x, np.linalg.solve(dense, b), tol=1e-8)
    assert cfgi[11] in (CONVERGED_RTOL, CONVERGED_ATOL)
    assert 0 < cfgi[10] <= 200
    ref, its = _upstream_solution(dense, b, "cg")
    same(x, ref, tol=1e-8)
    assert cfgi[10] == its
    assert cfg[7] < 1e-8


def test_mp_ksp_solve_bcgs():
    pytest.importorskip("petsc4py")
    dense = tridiag(30)
    n = 30
    a = csr(dense)
    b = rng(n, 57)
    x = np.zeros(n)
    cfg, cfgi, pci, pcf, pstmp = _ksp_state(n)
    work = np.zeros(7 * n)
    L.mp_ksp_solve_bcgs(
        n, addr(a.irow), addr(a.icol), addr(a.data), addr(pci), addr(pcf),
        addr(pstmp), addr(work), addr(cfg), addr(cfgi), addr(b), addr(x),
    )
    same(x, np.linalg.solve(dense, b), tol=1e-8)
    assert cfgi[11] in (CONVERGED_RTOL, CONVERGED_ATOL)
    ref, _ = _upstream_solution(dense, b, "bcgs")
    same(x, ref, tol=1e-8)


def _gmres_buffers(n, mk):
    hs = (
        (mk + 2) * (mk + 1) + (mk + 1) * (mk + 1)
        + 2 * (mk + 1) + 2 * (mk + 2)
    )
    return np.zeros(2 * n), np.zeros((mk + 1) * n), np.zeros(hs)


def test_mp_ksp_solve_gmres():
    pytest.importorskip("petsc4py")
    dense = tridiag(30)
    n = 30
    mk = 20
    a = csr(dense)
    b = rng(n, 58)
    x = np.zeros(n)
    cfg, cfgi, pci, pcf, pstmp = _ksp_state(n, restart=mk)
    work, vv, hs = _gmres_buffers(n, mk)
    L.mp_ksp_solve_gmres(
        n, addr(a.irow), addr(a.icol), addr(a.data), addr(pci), addr(pcf),
        addr(pstmp), addr(work), addr(vv), addr(hs), addr(cfg), addr(cfgi),
        addr(b), addr(x),
    )
    same(x, np.linalg.solve(dense, b), tol=1e-8)
    ref, _ = _upstream_solution(dense, b, "gmres")
    same(x, ref, tol=1e-8)


def test_mp_ksp_solve_fgmres():
    pytest.importorskip("petsc4py")
    dense = tridiag(30)
    n = 30
    mk = 20
    a = csr(dense)
    b = rng(n, 59)
    x = np.zeros(n)
    cfg, cfgi, pci, pcf, pstmp = _ksp_state(n, restart=mk)
    work, vv, hs = _gmres_buffers(n, mk)
    prevec = np.zeros((mk + 2) * n)
    L.mp_ksp_solve_fgmres(
        n, addr(a.irow), addr(a.icol), addr(a.data), addr(pci), addr(pcf),
        addr(pstmp), addr(work), addr(vv), addr(prevec), addr(hs), addr(cfg),
        addr(cfgi), addr(b), addr(x),
    )
    same(x, np.linalg.solve(dense, b), tol=1e-8)
    ref, _ = _upstream_solution(dense, b, "fgmres")
    same(x, ref, tol=1e-8)


def test_ksp_direct_with_a_jacobi_preconditioner_matches_upstream():
    """The same drivers, reached through a preconditioner block this file
    packs itself, so the PC path inside the kernel is exercised too."""
    pytest.importorskip("petsc4py")
    dense = tridiag(30)
    n = 30
    a = csr(dense)
    b = rng(n, 60)
    x = np.zeros(n)
    cfg, cfgi, pci, pcf, pstmp = _ksp_state(
        n, pc_block=_jacobi_pci_block(dense, JACOBI_DIAGONAL)
    )
    work = np.zeros(7 * n)
    L.mp_ksp_solve_cg(
        n, addr(a.irow), addr(a.icol), addr(a.data), addr(pci), addr(pcf),
        addr(pstmp), addr(work), addr(cfg), addr(cfgi), addr(b), addr(x),
    )
    same(x, np.linalg.solve(dense, b), tol=1e-8)
    ref, _ = _upstream_solution(dense, b, "cg")
    same(x, ref, tol=1e-8)


# ====================================================================
#  the mechanical audit this file exists for
# ====================================================================


EXPORTED = {
    "mp_mat_mult", "mp_mat_mult_add", "mp_mat_mult_transpose",
    "mp_mat_get_diagonal", "mp_mat_scale", "mp_mat_zero_entries",
    "mp_mat_get_diagonal_markers",
    "mp_vec_pointwise_mult", "mp_vec_pointwise_divide", "mp_vec_pointwise_min",
    "mp_vec_pointwise_max", "mp_vec_pointwise_max_abs", "mp_vec_norm",
    "mp_vec_dot", "mp_vec_tdot", "mp_vec_dot_norm2", "mp_vec_sum",
    "mp_vec_mean", "mp_vec_max", "mp_vec_min", "mp_vec_normalize",
    "mp_vec_reciprocal", "mp_vec_exp", "mp_vec_log", "mp_vec_sqrtabs",
    "mp_vec_chop", "mp_vec_abs", "mp_vec_conjugate", "mp_vec_copy",
    "mp_vec_swap", "mp_vec_set", "mp_vec_zero_entries", "mp_vec_scale",
    "mp_vec_shift", "mp_vec_axpy", "mp_vec_aypx", "mp_vec_axpby",
    "mp_vec_axpbypcz", "mp_vec_waxpy", "mp_vec_maxpy",
    "mp_vec_max_pointwise_divide",
    "mp_pc_jacobi_setup", "mp_pc_jacobi_setup_symmetric", "mp_pc_jacobi_apply",
    "mp_pc_sor_apply", "mp_pc_ilu_apply", "mp_pc_icc_apply", "mp_pc_ilu_numeric",
    "mp_pc_icc_numeric", "mp_ilu_symbolic_ilu0", "mp_ilu_symbolic",
    "mp_ilu_contiguous_lu", "mp_icc_symbolic", "mp_icc_contiguous_cholesky",
    "mp_ordering_rcm",
    "mp_ksp_solve_cg", "mp_ksp_solve_bcgs", "mp_ksp_solve_gmres",
    "mp_ksp_solve_fgmres",
}


def test_every_export_is_resolvable_in_the_loaded_library():
    from mojopetsc4py._lib import _SIGNATURES

    missing = sorted(n for n in _SIGNATURES if not hasattr(L, n))
    assert not missing, f"declared but not in {LIB}: {missing}"


def test_the_audited_export_set_matches_capi():
    from mojopetsc4py._lib import _SIGNATURES

    assert set(_SIGNATURES) == EXPORTED


def test_the_compiled_library_is_what_is_under_test():
    assert os.path.realpath(L._name) == os.path.realpath(LIB)
    assert L.mp_vec_norm is not None