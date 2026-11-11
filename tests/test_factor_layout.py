"""Regression tests for the incomplete factorisation layouts.

The ILU and IC factors are built by a symbolic pass that lays rows out in the
order its linked list drains them, and then rearranged into the layout the
triangular solve reads.  Both passes index into flat caller-owned arrays with no
bounds checks, so a layout mistake is a silent out-of-bounds write rather than
an exception.  These tests pin the layouts down directly.
"""

import numpy as np
import pytest

from mojopetsc4py import PETSc as Mojo
from mojopetsc4py._lib import addr, lib


def laplacian_csr(k):
    """The k x k 5-point Laplacian as (indptr, indices, values)."""
    n = k * k
    indptr, indices, values = [0], [], []
    for i in range(n):
        r, c = divmod(i, k)
        for dr, dc in ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < k and 0 <= cc < k:
                indices.append(rr * k + cc)
                values.append(4.0 if (dr == 0 and dc == 0) else -1.0)
        indptr.append(len(indices))
    return (
        n,
        np.array(indptr, dtype=np.int32),
        np.array(indices, dtype=np.int32),
        np.array(values, dtype=np.float64),
    )


def laplacian(k):
    n, indptr, indices, values = laplacian_csr(k)
    m = Mojo.Mat().createSeqAIJ((n, n), nnz=5)
    m.setValuesCSR(indptr, indices, values)
    m.assemble()
    return m


def icc_stage(k, levels):
    """Run only the IC symbolic pass; return (ui, udiag, staging, cap)."""
    m = laplacian(k)
    n = m._nrows
    perm = np.arange(n, dtype=np.int64)
    cap = max(4 * n, 4 * int(m._irow[-1]) + 4 * n, 16)
    while True:
        ui = np.zeros(n + 1, dtype=np.int64)
        ujsrc = np.zeros(cap, dtype=np.int64)
        ujlvl = np.zeros(cap, dtype=np.int64)
        udiag = np.zeros(n, dtype=np.int64)
        scratch = np.zeros(6 * (n + 1) + 2 * cap, dtype=np.int64)
        nnz = lib().mp_icc_symbolic(
            addr(ui), addr(ujsrc), addr(ujlvl), addr(udiag), addr(m.irow),
            addr(m.icol), addr(perm), addr(perm), levels, cap, addr(scratch),
            n, int(scratch.size),
        )
        if nnz >= 0:
            return ui, udiag, ujsrc, cap
        cap = -nnz


def icc_symbolic(k, levels):
    """Run the IC symbolic + contiguous passes and return the factor arrays."""
    m = laplacian(k)
    n = m._nrows
    ui, udiag, ujsrc, cap = icc_stage(k, levels)
    bi = np.zeros(n + 1, dtype=np.int64)
    bdiag = np.zeros(n + 1, dtype=np.int64)
    bdiag[:n] = udiag[:n]
    bj = np.zeros(cap, dtype=np.int64)
    lib().mp_icc_contiguous_cholesky(
        addr(ujsrc), addr(ui), addr(bdiag), addr(bi), addr(bdiag), addr(bj), n
    )
    return bi, bj, bdiag, n


@pytest.mark.parametrize("k", [4, 6, 8, 12])
@pytest.mark.parametrize("levels", [0, 1, 2])
def test_icc_symbolic_pattern_is_well_formed(k, levels):
    """`udiag[i]` is the last entry of factor row i, as PETSc's SBAIJ stores it.

    The factor holds the strictly upper triangle of A, ascending, with the
    diagonal slot -- which holds 1/d_i -- ending the row; that is the layout
    MatGetRowIJ reports on a real IC factor.
    """
    bi, bj, udiag, n = icc_symbolic(k, levels)

    assert int(bi[0]) == 0
    assert int(bi[n]) <= len(bj)
    for i in range(n):
        lo, hi = int(bi[i]), int(bi[i + 1])
        assert lo <= hi, f"row {i}: bi[{i}]={lo} > bi[{i+1}]={hi}"
        assert hi <= int(bi[n]), f"row {i} runs past bi[n]"
        assert int(udiag[i]) == hi - 1, (
            f"row {i}: udiag[{i}]={udiag[i]} but the row ends at {hi - 1}"
        )
        cols = [int(bj[j]) for j in range(lo, hi)]
        assert all(0 <= c < n for c in cols), f"row {i}: column out of {cols}"
        assert cols[-1] == i, f"row {i}: last column is {cols[-1]}, not {i}"
        strict = cols[:-1]
        # IC(0) keeps A's own pattern, which is strictly upper here; the
        # level-of-fill merges add fill columns on either side of the
        # diagonal, so only the layout -- diagonal last, columns ascending --
        # is invariant
        if levels == 0:
            assert all(c > i for c in strict), (
                f"row {i}: IC(0) row is not strictly upper: {cols}"
            )
        assert strict == sorted(strict), (
            f"row {i}: strict columns not ascending: {strict}"
        )


@pytest.mark.parametrize("k", [4, 6, 8, 12])
@pytest.mark.parametrize("levels", [0, 1, 2])
def test_icc_contiguous_pass_leaves_ui_alone(k, levels):
    """`ui` is the staging row map; the factor's row starts are `b->i`.

    The pass used to write `udiag` through `ui`, which is a different array
    and must stay the map it is.
    """
    n = laplacian(k)._nrows
    ui, udiag, ujsrc, cap = icc_stage(k, levels)
    before_ui = ui.copy()
    bi = np.zeros(n + 1, dtype=np.int64)
    bdiag = np.zeros(n + 1, dtype=np.int64)
    bdiag[:n] = udiag[:n]
    bj = np.zeros(cap, dtype=np.int64)
    lib().mp_icc_contiguous_cholesky(
        addr(ujsrc), addr(ui), addr(bdiag), addr(bi), addr(bdiag), addr(bj), n
    )
    np.testing.assert_array_equal(ui, before_ui)
    np.testing.assert_array_equal(bi[1:], udiag + 1)


@pytest.mark.parametrize("k", [4, 8, 12, 16])
@pytest.mark.parametrize("levels", [0, 1, 2])
def test_icc_factor_pattern_covers_the_matrix_pattern(k, levels):
    """Every entry of A's upper triangle must be in the factor's pattern."""
    n, indptr, indices, _ = laplacian_csr(k)
    bi, bj, _, _ = icc_symbolic(k, levels)
    pattern = set()
    for i in range(n):
        for j in range(int(bi[i]), int(bi[i + 1])):
            pattern.add((i, int(bj[j])))
    missing = []
    for i in range(n):
        for j in range(int(indptr[i]), int(indptr[i + 1])):
            if int(indices[j]) >= i and (i, int(indices[j])) not in pattern:
                missing.append((i, int(indices[j])))
    assert not missing, f"entries of A missing from the IC factor: {missing[:8]}"


def test_ilu_factor_layout():
    """ILU(0) stores all L rows first, then the U rows in reverse order, each
    U row ending on its (inverted) diagonal."""
    m = laplacian(5)
    n = m._nrows
    adiag = np.zeros(n, dtype=np.int64)
    lib().mp_mat_get_diagonal_markers(
        addr(m.irow), addr(m.icol), addr(adiag), n
    )
    bi = np.zeros(n + 1, dtype=np.int64)
    bj = np.zeros(int(m.irow[-1]), dtype=np.int64)
    bdiag = np.zeros(n + 1, dtype=np.int64)
    nnz = lib().mp_ilu_symbolic_ilu0(
        addr(bi), addr(bj), addr(bdiag), addr(adiag), addr(m.irow),
        addr(m.icol), n,
    )
    assert nnz == int(m.irow[-1])
    assert int(bi[0]) == 0
    assert int(bdiag[n]) == int(bi[n]) - 1
    for i in range(n):
        # the L row holds the strictly lower entries, in ascending order
        lrow = [int(bj[j]) for j in range(int(bi[i]), int(bi[i + 1]))]
        assert all(c < i for c in lrow), f"L row {i}: {lrow}"
        assert lrow == sorted(lrow)
        # the U row is stored reversed and ends on its diagonal
        urow = [int(bj[j]) for j in range(int(bdiag[i + 1]) + 1, int(bdiag[i]) + 1)]
        assert urow[-1] == i, f"U row {i} does not end on its diagonal: {urow}"
        assert all(c > i for c in urow[:-1]), f"U row {i}: {urow}"
