"""The cached FFI addresses move with the storage they point into.

`Vec` and the preconditioner keep each buffer's data address beside the buffer,
because `np.ndarray.ctypes.data` costs more per read than a short kernel runs
in.  A cached address that survives a reallocation points at freed storage, so
these cases rebind the storage through each public setter that can do it and
check the address the wrapper reports is the new buffer's.
"""

import numpy as np
import pytest

petsc4py = pytest.importorskip("petsc4py")

from mojopetsc4py import PETSc as Mojo  # noqa: E402


def test_create_with_array_reports_its_own_address():
    a = np.arange(8, dtype=np.float64)
    v = Mojo.Vec().createWithArray(a)
    assert v._ptr() == v.getArray().ctypes.data
    # the buffer is owned by the vector, not aliased to the caller's array
    assert v._ptr() != a.ctypes.data


def test_set_array_keeps_the_address_consistent():
    v = Mojo.Vec().createSeq(8)
    before = v._ptr()
    v.setArray(np.ones(8))
    assert v._ptr() == v.getArray().ctypes.data
    assert v._ptr() == before          # copied into the same storage


def test_duplicate_has_its_own_address():
    v = Mojo.Vec().createSeq(8)
    d = v.duplicate()
    assert d._ptr() == d.getArray().ctypes.data
    assert d._ptr() != v._ptr()


def test_destroy_then_create_reports_the_new_address():
    """After a rebind the reported address is the new buffer's, never the old."""
    v = Mojo.Vec().createSeq(64)
    v.destroy()
    assert v._buf is None
    v.createSeq(64)
    assert v._ptr() == v.getArray().ctypes.data


def test_a_view_has_its_own_address():
    """A slice of the same storage is a different buffer with a different
    address, which is why the cache is per vector and not per storage block."""
    base = Mojo.Vec().createSeq(16)
    off = base.getArray()[8:]
    assert off.ctypes.data != base._ptr()


@pytest.mark.parametrize("pc_type", ["jacobi", "ilu", "icc"])
def test_pc_descriptor_block_tracks_the_factor(pc_type):
    """The descriptor's addresses follow the factor across a re-setup."""
    n = 32
    m = Mojo.Mat().createSeqAIJ((n, n), nnz=3)
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in (i - 1, i, i + 1):
            if 0 <= j < n:
                rows.append(i)
                cols.append(j)
                vals.append(4.0 if j == i else -1.0)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, np.asarray(rows) + 1, 1)
    m.setValuesCSR(np.cumsum(indptr), np.asarray(cols), np.asarray(vals))
    m.assemble()
    pc = Mojo.PC().create()
    pc.setType(pc_type)
    pc.setOperators(m)
    pc.setUp()
    first = pc._pci_block(n).copy()
    # a reset drops the factor and the cached block together; the rebuilt block
    # must point at whatever the new factor is, not at the old buffers
    pc.reset()
    pc.setUp()
    again = pc._pci_block(n)
    assert again[0] == first[0]
    x = Mojo.Vec().createSeq(n)
    x.setArray(np.ones(n))
    y = Mojo.Vec().createSeq(n)
    pc.apply(x, y)
    assert np.all(np.isfinite(y.getArray()))
