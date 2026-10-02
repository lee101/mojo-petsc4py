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


def icc_symbolic(k, levels):
    """Run the IC symbolic + contiguous passes and return (ui, uj, udiag)."""
    m = laplacian(k)
    n = m._nrows
    perm = np.arange(n, dtype=np.int64)
    rinv = np.empty(n, dtype=np.int64)
    rinv[perm] = np.arange(n, dtype=np.int64)
    scratch = np.zeros(7 * n + 8, dtype=np.int64)
    cap = max(4 * n, 4 * int(m._irow[-1]) + 4 * n, 16)
    while True:
        ui = np.zeros(n + 1, dtype=np.int64)
        ujsrc = np.zeros(cap, dtype=np.int64)
        ujlvl = np.zeros(cap, dtype=np.int64)
        udiag = np.zeros(n, dtype=np.int64)
        nnz = lib().mp_icc_symbolic(
            addr(ui), addr(ujsrc), addr(ujlvl), addr(udiag), addr(m.irow),
            addr(m.icol), addr(perm), addr(rinv), levels, cap, addr(scratch), n,
        )
        if nnz >= 0:
            break
        cap = -nnz + n
    assert int(ui[n]) <= cap, "symbolic overran the buffer it was given"
    uj = np.zeros(max(int(ui[n]), 1), dtype=np.int64)
    lib().mp_icc_contiguous_cholesky(
        addr(ui), addr(udiag), addr(ujsrc), addr(uj), n
    )
    return ui, uj, udiag, n


@pytest.mark.parametrize("k", [4, 6, 8, 12])
@pytest.mark.parametrize("levels", [0, 1, 2])
def test_icc_symbolic_pattern_is_well_formed(k, levels):
    """U(i,:) runs ui[i] .. udiag[i] with the diagonal last, as PETSc stores it.

    The strictly upper triangle comes first in ascending column order and the
    diagonal slot, which holds 1/d_i, ends the row.
    """
    ui, uj, udiag, n = icc_symbolic(k, levels)

    assert int(ui[0]) == 0
    assert int(ui[n]) <= len(uj)
    for i in range(n):
        lo, hi = int(ui[i]), int(ui[i + 1])
        assert lo <= hi, f"row {i}: ui[{i}]={lo} > ui[{i+1}]={hi}"
        assert hi <= int(ui[n]), f"row {i} runs past ui[n]"
        assert int(udiag[i]) == hi - 1, (
            f"row {i}: udiag[{i}]={udiag[i]} but the row ends at {hi - 1}"
        )
        cols = [int(uj[j]) for j in range(lo, hi)]
        assert all(0 <= c < n for c in cols), f"row {i}: column out of {cols}"
        assert cols[-1] == i, f"row {i}: last column is {cols[-1]}, not {i}"
        upper = cols[:-1]
        assert all(c > i for c in upper), (
            f"row {i}: U(i,:) is not strictly upper: {cols}"
        )
        assert upper == sorted(upper), (
            f"row {i}: strict upper triangle not ascending: {upper}"
        )


@pytest.mark.parametrize("k", [4, 6, 8, 12])
@pytest.mark.parametrize("levels", [0, 1, 2])
def test_icc_contiguous_pass_does_not_touch_the_staging_buffer(k, levels):
    """The rearrange copies into a second buffer; the old in-place shift read
    the destination it had just overwritten, which corrupted the next row."""
    m = laplacian(k)
    n = m._nrows
    perm = np.arange(n, dtype=np.int64)
    rinv = np.empty(n, dtype=np.int64)
    rinv[perm] = np.arange(n, dtype=np.int64)
    scratch = np.zeros(7 * n + 8, dtype=np.int64)
    cap = max(4 * n, 4 * int(m._irow[-1]) + 4 * n, 16)
    while True:
        ui = np.zeros(n + 1, dtype=np.int64)
        ujsrc = np.zeros(cap, dtype=np.int64)
        ujlvl = np.zeros(cap, dtype=np.int64)
        udiag = np.zeros(n, dtype=np.int64)
        nnz = lib().mp_icc_symbolic(
            addr(ui), addr(ujsrc), addr(ujlvl), addr(udiag), addr(m.irow),
            addr(m.icol), addr(perm), addr(rinv), levels, cap, addr(scratch), n,
        )
        if nnz >= 0:
            break
        cap = -nnz + n
    before = ujsrc.copy()
    uj = np.zeros(max(int(ui[n]), 1), dtype=np.int64)
    lib().mp_icc_contiguous_cholesky(
        addr(ui), addr(udiag), addr(ujsrc), addr(uj), n
    )
    np.testing.assert_array_equal(ujsrc, before)


@pytest.mark.parametrize("k", [4, 8, 12, 16])
@pytest.mark.parametrize("levels", [0, 1, 2])
def test_icc_factor_pattern_covers_the_matrix_pattern(k, levels):
    """Every entry of A's upper triangle must be in the factor's pattern."""
    n, indptr, indices, _ = laplacian_csr(k)
    ui, uj, _, _ = icc_symbolic(k, levels)
    pattern = set()
    for i in range(n):
        for j in range(int(ui[i]), int(ui[i + 1])):
            pattern.add((i, int(uj[j])))
    missing = []
    for i in range(n):
        for j in range(int(indptr[i]), int(indptr[i + 1])):
            if int(indices[j]) >= i and (i, int(indices[j])) not in pattern:
                missing.append((i, int(indices[j])))
    assert not missing, f"entries of A missing from the IC factor: {missing[:8]}"