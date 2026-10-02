"""Parity of the sequential matrix algebra against real upstream petsc4py.

The port stores a `MATSEQAIJ` matrix as compressed sparse row with 64-bit index
arrays; upstream uses 32-bit `PetscInt`, which changes no value.
"""

import numpy as np
import pytest

petsc4py = pytest.importorskip("petsc4py")
from petsc4py import PETSc as Ref  # noqa: E402

from mojopetsc4py import PETSc as Mojo  # noqa: E402

TOL = 1e-13


def laplacian(n):
    a = np.zeros((n, n))
    for i in range(n):
        a[i, i] = 2.0
        if i > 0:
            a[i, i - 1] = -1.0
        if i < n - 1:
            a[i, i + 1] = -1.0
    return a


def random_spd(n, seed=7):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(n, n))
    return m + m.T + 4.0 * np.eye(n)


def dense_pair(a):
    """The same dense matrix as an upstream MATSEQAIJ and a port Mat."""
    n, m = a.shape
    ref = Ref.Mat().create()
    ref.setSizes([n, m])
    ref.setType("seqaij")
    ref.setUp()
    for i in range(n):
        for j in range(m):
            if a[i, j] != 0.0:
                ref.setValue(i, j, float(a[i, j]))
    ref.assemble()

    mojo = Mojo.Mat().createSeqAIJ(n, nnz=m)
    for i in range(n):
        for j in range(m):
            if a[i, j] != 0.0:
                mojo.setValue(i, j, float(a[i, j]))
    mojo.assemble()
    return ref, mojo


def vec_pair(a, cls=None):
    cls = Ref if cls is None else cls
    v = cls.Vec().createSeq(len(a))
    v.setArray(np.asarray(a, dtype=np.float64))
    return v


def dense_of(ref, n, m):
    return np.array([[ref.getValue(i, j) for j in range(m)] for i in range(n)])


def test_mult_matches():
    a = laplacian(12)
    ref, mojo = dense_pair(a)
    x = np.arange(12.0) + 1.0
    yr = ref.createVecLeft()
    ref.mult(vec_pair(x), yr)
    ym = mojo.mult(vec_pair(x, Mojo), mojo.createVecLeft())
    np.testing.assert_allclose(yr.getArray(), a @ x, rtol=TOL)
    np.testing.assert_allclose(ym.getArray(), yr.getArray(), rtol=TOL)


def test_call_operator_matches():
    a = laplacian(6)
    _, mojo = dense_pair(a)
    x = np.ones(6)
    np.testing.assert_allclose(
        mojo(vec_pair(x, Mojo)).getArray(), a @ x, rtol=TOL
    )
    np.testing.assert_allclose((mojo @ vec_pair(x, Mojo)).getArray(), a @ x, rtol=TOL)


def test_mult_transpose_matches():
    a = random_spd(9, seed=3)
    ref, mojo = dense_pair(a)
    x = np.arange(9.0) + 0.5
    yr = ref.createVecLeft()
    ref.multTranspose(vec_pair(x), yr)
    ym = mojo.multTranspose(vec_pair(x, Mojo), mojo.createVecLeft())
    np.testing.assert_allclose(yr.getArray(), a.T @ x, rtol=TOL)
    np.testing.assert_allclose(ym.getArray(), yr.getArray(), rtol=TOL)


def test_mult_add_matches():
    a = laplacian(8)
    ref, mojo = dense_pair(a)
    x = np.linspace(0.1, 1.0, 8)
    v = np.linspace(2.0, 3.0, 8)
    yr = ref.createVecLeft()
    ref.multAdd(vec_pair(x), vec_pair(v), yr)
    ym = mojo.multAdd(
        vec_pair(x, Mojo), vec_pair(v, Mojo), mojo.createVecLeft()
    )
    np.testing.assert_allclose(yr.getArray(), a @ x + v, rtol=TOL)
    np.testing.assert_allclose(ym.getArray(), yr.getArray(), rtol=TOL)


def test_get_diagonal_matches():
    a = laplacian(10)
    ref, mojo = dense_pair(a)
    dr = ref.getDiagonal()
    dm = mojo.getDiagonal()
    np.testing.assert_allclose(dr.getArray(), np.diag(a), rtol=TOL)
    np.testing.assert_allclose(dm.getArray(), dr.getArray(), rtol=TOL)


def test_get_value_and_values_match():
    a = random_spd(7, seed=11)
    ref, mojo = dense_pair(a)
    rows = [0, 2, 5, 6]
    cols = [6, 0, 3, 1]
    for r, c in zip(rows, cols):
        assert mojo.getValue(r, c) == pytest.approx(ref.getValue(r, c), rel=TOL)

    np.testing.assert_allclose(
        mojo.getValues(rows, cols), ref.getValues(rows, cols), rtol=TOL
    )


def test_setvalues_fills_the_outer_product():
    """`setValues(rows, cols, values)` fills the rows (x) cols block, so
    `values` must hold at least len(rows) * len(cols) scalars."""
    ref = Ref.Mat().create()
    ref.setSizes([2, 3])
    ref.setType("seqaij")
    ref.setUp()
    mojo = Mojo.Mat().createSeqAIJ((2, 3), nnz=3)
    block = np.arange(6.0)
    ref.setValues([0, 1], [0, 1, 2], block)
    mojo.setValues([0, 1], [0, 1, 2], block)
    ref.assemble()
    dense = dense_of(ref, 2, 3)
    np.testing.assert_allclose(dense, block.reshape(2, 3), rtol=TOL)
    np.testing.assert_allclose(
        np.array([[mojo.getValue(i, j) for j in range(3)] for i in range(2)]),
        dense, rtol=TOL,
    )
    with pytest.raises(ValueError):
        mojo.setValues([0, 1], [0, 1, 2], [1.0, 2.0, 3.0])


def test_setvaluescsr_roundtrips_and_matches_upstream():
    """`setValuesCSR` installs a CSR triple directly and round-trips through
    `getValuesCSR`; the resulting matrix matches upstream built the same way."""
    a = laplacian(5)
    indptr, indices, data = [], [], []
    for r in range(a.shape[0]):
        indptr.append(len(indices))
        for c in range(a.shape[1]):
            if a[r, c] != 0.0:
                indices.append(c)
                data.append(a[r, c])
    indptr.append(len(indices))
    indptr = np.array(indptr, dtype=np.int32)
    indices = np.array(indices, dtype=np.int32)
    data = np.array(data, dtype=np.float64)

    ref = Ref.Mat().create()
    ref.setSizes([5, 5])
    ref.setType("seqaij")
    ref.setUp()
    ref.setValuesCSR(indptr, indices, data)
    ref.assemble()

    mojo = Mojo.Mat().createSeqAIJ((5, 5), nnz=2)
    mojo.setValuesCSR(indptr, indices, data)
    mojo.assemble()

    # the port round-trips its own CSR
    ri, rc, rv = mojo.getValuesCSR()
    np.testing.assert_array_equal(ri, indptr)
    np.testing.assert_array_equal(rc, indices)
    np.testing.assert_allclose(rv, data, rtol=TOL)

    np.testing.assert_allclose(dense_of(ref, 5, 5), a, rtol=TOL)
    np.testing.assert_allclose(
        np.array([[mojo.getValue(i, j) for j in range(5)] for i in range(5)]),
        a, rtol=TOL,
    )


def test_setvaluescsr_rejects_mismatched_arrays():
    mojo = Mojo.Mat().createSeqAIJ((3, 3), nnz=2)
    with pytest.raises(ValueError):
        mojo.setValuesCSR([0, 1], [0], [1.0])          # indices/values mismatch
    with pytest.raises(ValueError):
        mojo.setValuesCSR([0, 1], [0], [1.0])          # indptr too short for 3 rows


def test_is_symmetric():
    ref, mojo = dense_pair(laplacian(6))
    assert ref.isSymmetric(1e-14) is True
    assert mojo.isSymmetric(1e-14) is True


def test_scale_and_zero_entries_keep_the_structure():
    a = laplacian(5)
    ref, mojo = dense_pair(a)
    ref.scale(-2.0)
    mojo.scale(-2.0)
    ri, rc, rv = ref.getValuesCSR()
    mi, mc, mv = mojo.getValuesCSR()
    np.testing.assert_array_equal(ri, mi)
    np.testing.assert_array_equal(rc, mc)
    np.testing.assert_allclose(mv, rv, rtol=TOL)
    ref.zeroEntries()
    mojo.zeroEntries()
    ri, rc, rv = ref.getValuesCSR()
    mi, mc, mv = mojo.getValuesCSR()
    np.testing.assert_array_equal(ri, mi)
    np.testing.assert_array_equal(rc, mc)
    np.testing.assert_allclose(mv, rv, rtol=TOL)
    assert np.all(mv == 0.0)


def test_transpose_matrix_matches():
    a = random_spd(8, seed=5)
    ref, mojo = dense_pair(a)
    tr = ref.transpose()
    tr.setUp()
    tm = mojo.transpose()
    x = np.arange(8.0) + 1.0
    yr = tr.createVecLeft()
    tr.mult(vec_pair(x), yr)
    ym = tm.mult(vec_pair(x, Mojo), tm.createVecLeft())
    np.testing.assert_allclose(yr.getArray(), (a.T @ x)[:8], rtol=TOL)
    np.testing.assert_allclose(ym.getArray(), yr.getArray(), rtol=TOL)


def test_mult_transpose_nonsquare_sizes_the_output_by_column():
    """A 2x5 matrix times a length-2 vector gives a length-5 result."""
    a = np.zeros((2, 5))
    a[0, 0] = 1.0
    a[1, 4] = 2.0
    x = np.array([1.0, 1.0])
    m = Mojo.Vec().createSeq(2)
    m.setArray(x)
    # dense_pair only builds square matrices, so the reference product is
    # compared against numpy for the nonsquare case
    mojo = Mojo.from_dense(a)
    ym = mojo.multTranspose(m)
    assert ym.getSize() == 5
    np.testing.assert_allclose(ym.getArray(), a.T @ x, rtol=TOL)
    v = Mojo.Vec().createSeq(5)
    v.setArray(np.full(5, 3.0))
    ya = mojo.multTransposeAdd(m, v)
    assert ya.getSize() == 5
    np.testing.assert_allclose(ya.getArray(), a.T @ x + 3.0, rtol=TOL)


def test_set_value_after_assemble_keeps_the_earlier_entries():
    """`setValue` on a materialised matrix must not drop what is there."""
    _, mojo = dense_pair(np.diag([1.0, 2.0, 3.0]))
    mojo.setValue(2, 0, 5.0)
    _, c, v = mojo.getValuesCSR()
    assert c.tolist() == [0, 1, 0, 2]
    assert v.tolist() == [1.0, 2.0, 5.0, 3.0]
    assert mojo.getValue(1, 1) == 2.0


def test_operations_reject_mismatched_operand_sizes():
    a = np.zeros((3, 3)) + np.eye(3)
    _, mojo = dense_pair(a)
    short = Mojo.Vec().createSeq(2)
    short.setArray(np.ones(2))
    full = Mojo.Vec().createSeq(3)
    full.setArray(np.ones(3))
    with pytest.raises(ValueError):
        mojo.mult(short)
    with pytest.raises(ValueError):
        mojo.multTranspose(short)
    with pytest.raises(ValueError):
        mojo.getDiagonal(short)
    with pytest.raises(ValueError):
        mojo.mult(full, short)
