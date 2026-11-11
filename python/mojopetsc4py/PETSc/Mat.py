"""Matrix types and operations, mirroring `petsc4py/PETSc/Mat.pyx`.

Sequential `MATSEQAIJ` only: compressed sparse row with `int64` index arrays.


Transcribed from PETSc 3.25.5 / petsc4py 3.25.5 (BSD-2-Clause); see NOTICE
at the repository root for their copyright notices, and LICENSE for this
file's own MIT licence.
"""

from __future__ import annotations

import numpy as np

from .._lib import addr, f64, lib
from .Vec import Vec


class MatType:
    """The matrix type."""

    AIJ = "aij"
    SEQAIJ = "seqaij"
    MPIAIJ = "mpiaij"
    SBAIJ = "sbaij"
    DENSE = "dense"
    SHELL = "shell"
    NEST = "nest"


class MatInfoType:
    """Matrix information flags."""

    NNZ = "nz_allocated"
    NZ_USED = "nz_used"
    NNZ_USED = "nz_used"
    NNZ_ALLOCATED = "nz_allocated"
    FACTOR_NONZERO = "factor_nonz"
    ZERO_TOLERANCE = "rcond"
    ROW_ZERO_TOLERANCE = "rcond"


class MatStructures:
    """`MatStructure` (`Mat.Structure` upstream): the copy modes `compose`
    and `copy` take."""

    SAME_NONZERO_PATTERN = 2
    DIFFERENT_NONZERO_PATTERN = 0

    SAME = SAME_NONZERO_PATTERN
    DIFFERENT = DIFFERENT_NONZERO_PATTERN


class MatOrderingType:
    """Matrix ordering types (`MatOrderingType` in Mat.pyx)."""
    NATURAL = "natural"
    ND = "nd"
    RCM = "rcm"
    MD = "md"
    ROWLENGTH = "rowlength"
    QMD = "qmd"
    AMD = "amd"
    METISND = "metisnd"
    EXTERNAL = "external"
    NATURAL_OR_ND = "natural_or_nd"


class MatSORType:
    """SOR sweep types (`MatSORType` in Mat.pyx)."""

    ZERO_INITIAL_GUESS = 16
    FORWARD_SWEEP = 1
    BACKWARD_SWEEP = 2
    LOCAL_FORWARD_SWEEP = 4
    LOCAL_BACKWARD_SWEEP = 8
    LOCAL_SYMMETRIC_SWEEP = 12
    EISENSTAT = 32
    APPLY_UPPER = 64
    APPLY_LOWER = 128
    FORWARD = 1
    BACKWARD = 2
    SYMMETRIC = 3
    LOCAL_FORWARD = 4
    LOCAL_BACKWARD = 8
    LOCAL_SYMMETRIC = 12


class Mat:
    """A sequential AIJ (CSR) matrix in the layout `MATSEQAIJ` uses.

    `irow[nrows + 1]` holds the row starts, `icol[nnz]` the column indices
    ascending within a row, `data[nnz]` the values in the same order.
    """

    Type = MatType
    InfoType = MatInfoType
    OrderingType = MatOrderingType
    Structure = MatStructures
    SORType = MatSORType

    def __init__(self):
        self._irow = None
        self._icol = None
        self._data = None
        self._nrows = 0
        self._ncols = 0
        self._assembled = False
        self._nnz_used = 0
        self._pending = None

    # ---------------------------------------------------------- creation
    def create(self, comm=None) -> "Mat":
        return self

    def createSeqAIJ(self, size, bsize=None, nnz=None, comm=None) -> "Mat":
        """Create a sequential `Type.SEQAIJ` matrix.

        `size` is the number of rows, or a `(rows, cols)` pair.
        """
        if isinstance(size, (tuple, list)):
            m, n = int(size[0]), int(size[1])
        else:
            m = n = int(size)
        per_row = 1 if nnz is None else int(nnz)
        self._nrows = m
        self._ncols = n
        self._irow = np.arange(m + 1, dtype=np.int64) * per_row
        self._icol = np.zeros(m * per_row, dtype=np.int64)
        self._data = np.zeros(m * per_row, dtype=np.float64)
        self._assembled = True
        self._pending = {}
        return self

    def createAIJ(self, size, bsize=None, nnz=None, comm=None) -> "Mat":
        return self.createSeqAIJ(size, bsize, nnz, comm)

    def setSizes(self, size, bsize=None) -> "Mat":
        if isinstance(size, (tuple, list)):
            self._nrows, self._ncols = int(size[0]), int(size[1])
        else:
            self._nrows = self._ncols = int(size)
        return self

    def setType(self, mat_type) -> "Mat":
        return self

    def assemble(self) -> "Mat":
        self._materialize()
        return self

    def duplicate(self, copy: bool = False) -> "Mat":
        self._materialize()
        out = Mat()
        out._nrows = self._nrows
        out._ncols = self._ncols
        out._assembled = self._assembled
        out._nnz_used = self._nnz_used
        out._irow = self._irow.copy()
        out._icol = self._icol.copy()
        out._data = self._data.copy() if copy else np.zeros_like(self._data)
        return out

    def copy(self, result=None, structure=None) -> "Mat":
        """Copy this matrix into `result` (a duplicate when `None`).

        The pending-entry dict is flushed first, so a copy taken before the
        source is assembled holds the same entries as one taken after.  With
        the default structure (or `SAME_NONZERO_PATTERN`) the target gets the
        source's CSR arrays.  With `DIFFERENT_NONZERO_PATTERN` the target keeps
        its own pattern and receives the values of the entries it already has,
        as `MatCopy` does: an entry the target does not hold is dropped, and one
        it holds and the source lacks becomes zero.
        """
        source = self
        source._materialize()
        target = source.duplicate() if result is None else result
        if structure == MatStructures.DIFFERENT_NONZERO_PATTERN:
            target._materialize()
            if (target._nrows, target._ncols) != (source._nrows, source._ncols):
                raise ValueError(
                    f"incompatible sizes: {source.getSize()} into"
                    f" {target.getSize()}"
                )
            src = {
                (int(source._icol[j]), i): float(source._data[j])
                for i in range(source._nrows)
                for j in range(int(source._irow[i]), int(source._irow[i + 1]))
            }
            trow, tcol, _ = target.getValuesCSR()
            newdata = np.zeros(len(tcol), dtype=np.float64)
            for i in range(target._nrows):
                for j in range(int(trow[i]), int(trow[i + 1])):
                    newdata[j] = src.get((int(tcol[j]), i), 0.0)
            target._data[: len(newdata)] = newdata
            return target
        target._nrows = source._nrows
        target._ncols = source._ncols
        target._irow = source._irow.copy()
        target._icol = source._icol.copy()
        target._data = source._data.copy()
        target._nnz_used = source._nnz_used
        target._assembled = source._assembled
        target._pending = None
        return target

    def destroy(self) -> "Mat":
        self._irow = None
        return self

    def _check(self):
        if self._irow is None:
            raise RuntimeError("Mat has not been created")
        return self

    # ---------------------------------------------------------- assembly
    def setUp(self) -> "Mat":
        self._check()._materialize()
        self._assembled = True
        return self

    def assemblyBegin(self) -> None:
        return None

    def assemblyEnd(self) -> None:
        self._assembled = True

    def assemble(self) -> None:
        self.assemblyEnd()

    def zeroEntries(self) -> None:
        """Set every stored entry to zero, keeping the nonzero structure, as
        upstream's `MatZeroEntries` does."""
        self._check()
        self._materialize()
        # every stored value is cleared, including one that was never staged as
        # a pending entry (a matrix built by `from_dense`, or one that was
        # scaled after assembly); `MatZeroEntries` keeps the structure
        self._data[:] = 0.0

    def setPreallocationNNZ(self, nnz) -> "Mat":
        return self

    def setOption(self, option, flag) -> None:
        return None

    def setType(self, mat_type) -> None:
        return None

    def getType(self) -> str:
        return "seqaij"

    # ---------------------------------------------------------- accessors
    def getSize(self):
        return (self._nrows, self._ncols)

    def getLocalSize(self):
        return (self._nrows, self._ncols)

    def getSizes(self):
        return ((self._nrows, self._nrows), (self._ncols, self._ncols))

    def getOwnershipRange(self):
        return (0, self._nrows)

    def getOwnershipRangeColumn(self):
        return (0, self._ncols)

    def getBlockSize(self) -> int:
        return 1

    def getNNZ(self) -> int:
        self._materialize()
        return int(self._nnz_used)

    def getInfo(self, info=None) -> dict:
        """Return the nonzero counts.  `nz_used` is the stored count, which
        includes entries that have been inserted but are zero."""
        self._materialize()
        return {
            "nz_allocated": int(len(self._data)),
            "nz_used": int(self._nnz_used),
            "rcond": 0.0,
        }

    @property
    def irow(self) -> np.ndarray:
        self._materialize()
        return self._irow

    @property
    def icol(self) -> np.ndarray:
        self._materialize()
        return self._icol

    @property
    def data(self) -> np.ndarray:
        self._materialize()
        return self._data

    def getValuesCSR(self):
        """Return the matrix in compressed sparse row form."""
        self._materialize()
        nnz = int(self._nnz_used)
        return (
            self._irow[: self._nrows + 1].copy(),
            self._icol[:nnz].copy(),
            self._data[:nnz].copy(),
        )

    def setValuesCSR(self, indptr, indices, data=None) -> "Mat":
        """Install a matrix directly from compressed sparse row arrays.

        Upstream's signature is `setValuesCSR(I, V)` for the AIJ matrix it was
        written for; here the sequential layout is CSR, so the three arrays are
        the row starts, the column indices and the values, which is the same
        data `getValuesCSR` returns.  This bypasses the pending-entry dict
        entirely, which is what makes building a large matrix cheap.
        """
        irow = np.asarray(indptr, dtype=np.int64)
        icol = np.asarray(indices, dtype=np.int64)
        vals = np.zeros(0, dtype=np.float64) if data is None else np.asarray(
            data, dtype=np.float64
        )
        if len(irow) != self._nrows + 1:
            raise ValueError(
                f"incompatible array sizes: indptr has {len(irow)} entries,"
                f" {self._nrows + 1} are needed"
            )
        if len(icol) != len(vals):
            raise ValueError(
                f"incompatible array sizes: {len(icol)} indices but"
                f" {len(vals)} values"
            )
        nnz = int(irow[-1])
        if self._data.size < nnz:
            self._data = np.zeros(nnz, dtype=np.float64)
            self._icol = np.zeros(nnz, dtype=np.int64)
        self._nnz_used = nnz
        self._irow[:] = irow
        self._icol[:nnz] = icol
        self._data[:nnz] = vals
        self._nnz_used = nnz
        self._pending = None
        return self

    def _locate(self, row: int, col: int):
        """Index of (row, col) in the CSR arrays, or (-1, -1) when absent."""
        if row < 0 or row >= self._nrows or col < 0 or col >= self._ncols:
            return (-1, -1)
        lo = int(self._irow[row])
        hi = int(self._irow[row + 1])
        pos = int(np.searchsorted(self._icol[lo:hi], col))
        if pos >= hi - lo or int(self._icol[lo + pos]) != col:
            return (-1, -1)
        return (row, lo + pos)

    def getValue(self, row, col) -> float:
        """Return the value in the (row, col) position."""
        return self._entry(int(row), int(col))

    def getValues(self, rows, cols, values=None) -> np.ndarray:
        """Return the values in the rows (x) cols block."""
        rows = list(rows)
        cols = list(cols)
        out = np.array(
            [[self.getValue(r, c) for c in cols] for r in rows],
            dtype=np.float64,
        )
        if values is not None:
            values[:] = out
            return values
        return out

    def _dense(self) -> np.ndarray:
        """Return the matrix as a dense 2-D array."""
        self._materialize()
        out = np.zeros((self._nrows, self._ncols), dtype=np.float64)
        for i in range(self._nrows):
            for j in range(int(self._irow[i]), int(self._irow[i + 1])):
                out[i, int(self._icol[j])] += float(self._data[j])
        return out

    def _materialize(self):
        """Rebuild the CSR arrays from the pending entries.

        PETSc's `MatSetValues` requires the caller to have preallocated a slot
        on a sequential AIJ matrix; a Python-level dict is friendlier and gives
        the same result, so entries are accumulated here and sorted and
        coalesced into CSR on demand.
        """
        if self._pending is None:
            return
        rows = self._pending
        nrow = self._nrows
        counts = np.zeros(nrow, dtype=np.int64)
        for r, cols in rows.items():
            counts[r] = len(cols)
        irow = np.zeros(nrow + 1, dtype=np.int64)
        irow[1:] = np.cumsum(counts)
        total = int(irow[-1])
        icol = np.zeros(max(total, 1), dtype=np.int64)
        data = np.zeros(max(total, 1), dtype=np.float64)
        for r, cols in rows.items():
            k = int(irow[r])
            for c in sorted(cols):
                icol[k] = c
                data[k] = cols[c]
                k += 1
        self._irow = irow
        self._icol = icol
        self._data = data
        self._nnz_used = total
        self._pending = None

    def _entry(self, row: int, col: int) -> float:
        if self._pending is not None:
            return self._pending.get(row, {}).get(col, 0.0)
        i, j = self._locate(row, col)
        return 0.0 if i < 0 else float(self._data[j])

    def _spill(self):
        """Move the current CSR entries into `_pending` so that a value set
        after materialisation keeps the entries already there."""
        if self._pending is not None:
            return
        self._pending = {}
        for i in range(self._nrows):
            cols = {}
            for j in range(int(self._irow[i]), int(self._irow[i + 1])):
                cols[int(self._icol[j])] = float(self._data[j])
            if cols:
                self._pending[i] = cols

    def _setValue(self, row: int, col: int, value: float, addv: bool):
        if row < 0 or row >= self._nrows or col < 0 or col >= self._ncols:
            raise ValueError(f"entry ({row}, {col}) is out of range")
        self._spill()
        cols = self._pending.setdefault(row, {})
        cols[col] = cols.get(col, 0.0) + value if addv else value

    def setValue(self, row, col, value, addv=None) -> None:
        self._setValue(int(row), int(col), float(value), addv is not None)

    def setValues(self, rows, cols, values, addv=None, insertmode=None) -> None:
        """Set the rows ⊗ cols entries of the matrix.

        `values` is a flat sequence of at least `len(rows) * len(cols)`
        scalars laid out over the outer product, as upstream documents; entries
        already present are added to when `addv` is given.
        """
        rows = list(rows)
        cols = list(cols)
        vals = list(values)
        need = len(rows) * len(cols)
        if len(vals) < need:
            raise ValueError(
                f"incompatible array sizes: ni={len(rows)}, nj={len(cols)},"
                f" nv={len(vals)}; at least {need} values are needed"
            )
        add = addv is not None
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                self._setValue(int(r), int(c), float(vals[i * len(cols) + j]), add)

    def setValueBlocked(self, rows, cols, value, addv=None) -> None:
        """Set every entry of the `rows` x `cols` block to a single `value`.

        Block size 1 here, so this is the diagonal-set shorthand: the block is
        the outer product of the two index lists, exactly as
        `MatSetValueBlocked` is `MatSetValues` with `bs * bs` copies per entry.
        """
        rows = list(rows)
        cols = list(cols)
        add = addv is not None
        for i, r in enumerate(rows):
            for c in cols:
                self._setValue(int(r), int(c), float(value), add)

    def setValuesBlocked(self, rows, cols, values, addv=None) -> None:
        """Set the `rows` x `cols` block, `values` laid out over the outer
        product, as upstream's `MatSetValuesBlocked` takes them."""
        rows = list(rows)
        cols = list(cols)
        vals = list(values)
        need = len(rows) * len(cols)
        if len(vals) < need:
            raise ValueError(
                f"incompatible array sizes: ni={len(rows)}, nj={len(cols)},"
                f" nv={len(vals)}; at least {need} values are needed"
            )
        add = addv is not None
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                self._setValue(
                    int(r), int(c), float(vals[i * len(cols) + j]), add
                )

    # ---------------------------------------------------------- algebra
    def _out(self, x: Vec) -> Vec:
        """`y = A @ x` takes a column-length vector and returns a row-length
        one."""
        if len(x.getArray()) != self._ncols:
            raise ValueError(
                f"incompatible sizes: matrix {self._ncols} columns, vector"
                f" {len(x.getArray())}"
            )
        return Vec().createSeq(self._nrows, x.getBlockSize())

    def _outT(self, x: Vec) -> Vec:
        """The output of the transposed product is indexed by column, and the
        input is indexed by row."""
        if len(x.getArray()) != self._nrows:
            raise ValueError(
                f"incompatible sizes: matrix {self._nrows} rows, vector"
                f" {len(x.getArray())}"
            )
        return Vec().createSeq(self._ncols, x.getBlockSize())

    def _check_y(self, y: Vec, n: int) -> Vec:
        if len(y.getArray()) != n:
            raise ValueError(
                f"incompatible sizes: output has {len(y.getArray())} entries,"
                f" {n} are needed"
            )
        return y

    # ---------------------------------------------------------- algebra

    def mult(self, x: Vec, y: Vec = None) -> Vec:
        """Perform the matrix vector product y = A @ x."""
        self._materialize()
        out = self._out(x) if y is None else self._check_y(y, self._nrows)
        lib().mp_mat_mult(
            addr(x.getArray()), addr(out.getArray()), addr(self._irow),
            addr(self._icol), addr(self._data), self._nrows,
        )
        return out

    def multAdd(self, x: Vec, v: Vec, y: Vec = None) -> Vec:
        """Perform the matrix vector product y = A @ x + v."""
        self._materialize()
        out = self._out(x) if y is None else self._check_y(y, self._nrows)
        self._check_y(v, self._nrows)
        if out is x:
            raise ValueError(
                "output y may not alias the input x in multAdd"
            )
        out.getArray()[:] = v.getArray()
        lib().mp_mat_mult_add(
            addr(x.getArray()), addr(out.getArray()), addr(self._irow),
            addr(self._icol), addr(self._data), self._nrows,
        )
        return out

    def multTranspose(self, x: Vec, y: Vec = None) -> Vec:
        """Perform the transposed matrix vector product y = Aᵀ @ x."""
        self._materialize()
        out = self._outT(x) if y is None else self._check_y(y, self._ncols)
        lib().mp_mat_mult_transpose(
            addr(x.getArray()), addr(out.getArray()), addr(self._irow),
            addr(self._icol), addr(self._data), self._nrows, self._ncols,
        )
        return out

    def multTransposeAdd(self, x: Vec, v: Vec, y: Vec = None) -> Vec:
        """Perform the transposed product y = Aᵀ @ x + v."""
        self._materialize()
        out = self._outT(x) if y is None else self._check_y(y, self._ncols)
        self._check_y(v, self._ncols)
        self._check_y(x, self._nrows)
        # PCArrayAliasCheck: the transpose kernel clears `y` before it reads
        # `x`, so upstream refuses `y is x`.
        if out is x:
            raise ValueError(
                "output y may not alias the input x in multTransposeAdd"
            )
        if out is v:
            # upstream allows this and adds the original v; the kernel clears
            # its output first, so keep v in a buffer the kernel does not write
            v = v.copy()
        out.getArray()[:] = v.getArray()
        lib().mp_mat_mult_transpose(
            addr(x.getArray()), addr(out.getArray()), addr(self._irow),
            addr(self._icol), addr(self._data), self._nrows, self._ncols,
        )
        lib().mp_vec_axpy(
            addr(out.getArray()), 1.0, addr(v.getArray()), self._ncols
        )
        return out

    def createVecLeft(self) -> Vec:
        """A vector a matrix-vector product can be applied to."""
        return Vec().createSeq(self._nrows)

    def createVecRight(self) -> Vec:
        """A vector for the transposed product, indexed by column."""
        return Vec().createSeq(self._ncols)

    def __call__(self, x, y=None):
        if y is None:
            y = self.createVecLeft()
        self.mult(x, y)
        return y

    def getDiagonal(self, result: Vec = None) -> Vec:
        self._materialize()
        out = (
            Vec().createSeq(self._nrows) if result is None
            else self._check_y(result, self._nrows)
        )
        lib().mp_mat_get_diagonal(
            addr(out.getArray()), addr(self._irow), addr(self._icol),
            addr(self._data), self._nrows,
        )
        return out

    def createDiagonal(self, d) -> "Mat":
        n = len(d.getArray())
        out = Mat().createSeqAIJ(n, nnz=1)
        rows = np.arange(n, dtype=np.int64)
        out.setValues(rows, rows, d.getArray())
        return out

    def createTranspose(self, mat=None) -> "Mat":
        src = self if mat is None else mat
        return _transpose(src)

    def transpose(self, out: "Mat" = None) -> "Mat":
        """Return the transpose of the matrix."""
        return _transpose(self)

    def scale(self, alpha) -> None:
        """Scale the matrix entries by a factor."""
        self._check()
        self._materialize()
        lib().mp_mat_scale(addr(self._data), int(len(self._data)), float(alpha))

    def equal(self, mat, tol: float = 0.0) -> bool:
        """Compare the stored entries, so the pending dict is flushed first."""
        other = mat
        self._materialize()
        other._materialize()
        if self._nrows != other._nrows or self._ncols != other._ncols:
            return False
        if not np.array_equal(self._irow, other._irow):
            return False
        if not np.array_equal(self._icol, other._icol):
            return False
        return bool(np.allclose(self._data, other._data, rtol=0, atol=tol))

    def isSymmetric(self, tol: float = 0.0) -> bool:
        """Return whether the matrix equals its own transpose within `tol`."""
        if self._nrows != self._ncols:
            return False
        return bool(np.allclose(self._dense(), _transpose(self)._dense(),
                                rtol=0, atol=tol))

    def isTranspose(self, mat=None, tol: float = 0.0) -> bool:
        """Return whether this matrix is the transpose of `mat` within `tol`."""
        other = self if mat is None else mat
        if self._nrows != other._ncols or self._ncols != other._nrows:
            return False
        return bool(np.allclose(self._dense(), _transpose(other)._dense(),
                                rtol=0, atol=tol))

    def __matmul__(self, x):
        return self.mult(x)

    def __mul__(self, other):
        if not isinstance(other, Mat):
            return NotImplemented
        return _matmul(self, other)

    def __neg__(self):
        out = self.duplicate(copy=True)
        out.scale(-1.0)
        return out




def _transpose(m: Mat) -> Mat:
    """Transpose a CSR matrix, summing duplicate entries as PETSc does."""
    rows, cols, vals = m.getValuesCSR()
    out = Mat().createSeqAIJ(m._ncols, nnz=max(1, int(m._irow[-1])))
    nnz = len(cols)
    counts = np.zeros(m._ncols + 1, dtype=np.int64)
    for j in cols:
        counts[int(j) + 1] += 1
    out._irow = np.cumsum(counts)
    total = int(out._irow[-1])
    if total > out._icol.size:
        out._icol = np.zeros(total, dtype=np.int64)
        out._data = np.zeros(total, dtype=np.float64)
    fill = out._irow.copy()
    for i in range(m._nrows):
        for j in range(int(rows[i]), int(rows[i + 1])):
            c = int(cols[j])
            k = int(fill[c])
            out._icol[k] = i
            out._data[k] = vals[j]
            fill[c] = k + 1
    out._nnz_used = total
    out._assembled = True
    out._pending = None
    return out


def _matmul(a: Mat, b: Mat) -> Mat:
    if a._ncols != b._nrows:
        raise ValueError(
            f"incompatible sizes: {a.getSize()} times {b.getSize()}"
        )
    ad = a._dense()
    bd = b._dense()
    return from_dense(ad @ bd)


def from_dense(array, mat=None) -> Mat:
    """Build a CSR matrix from a dense 2-D array."""
    arr = f64(array)
    if arr.ndim != 2:
        raise ValueError("expected a 2-D array")
    n, m = arr.shape
    out = mat if mat is not None else Mat()
    per_row = max(1, int(np.count_nonzero(arr[0])) if n else 1)
    out.createSeqAIJ((n, m), nnz=per_row)
    out._nrows = n
    out._ncols = m
    counts = np.count_nonzero(arr, axis=1).astype(np.int64)
    out._irow = np.zeros(n + 1, dtype=np.int64)
    out._irow[1:] = np.cumsum(counts)
    total = int(out._irow[-1])
    out._icol = np.zeros(max(total, 1), dtype=np.int64)
    out._data = np.zeros(max(total, 1), dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(m):
            if arr[i, j] != 0.0:
                out._icol[k] = j
                out._data[k] = arr[i, j]
                k += 1
    out._nnz_used = total
    out._assembled = True
    out._pending = None
    return out
