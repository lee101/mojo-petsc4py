"""Preconditioner types, mirroring `petsc4py/PETSc/PC.pyx`.

Covered types: `none`, `jacobi`, `sor`, `ilu`, `icc`.  Everything else in the
upstream `PCType` enum exists as a name but raises `NotImplementedError` when
selected; see the README's coverage section.
"""

from __future__ import annotations

import numpy as np

from .._lib import addr, lib
from .Mat import Mat, MatOrderingType, from_dense
from .Vec import Vec

# petscksp.h PCType ordinals
_NONE = 0
_JACOBI = 2
_SOR = 4
_ILU = 5
_ICC = 6

# petscmat.h MatSORType bit values
SOR_ZERO_INITIAL_GUESS = 16
SOR_FORWARD_SWEEP = 1
SOR_BACKWARD_SWEEP = 2
SOR_LOCAL_FORWARD_SWEEP = 4
SOR_LOCAL_BACKWARD_SWEEP = 8
SOR_LOCAL_SYMMETRIC_SWEEP = 12
SOR_EISENSTAT = 32
SOR_APPLY_UPPER = 64
SOR_APPLY_LOWER = 128


# petscpc.h PCJACOBIType, the values `PCJacobiSetType` accepts.
PC_JACOBI_DIAGONAL = 0
PC_JACOBI_ROWMAX = 1
PC_JACOBI_ROWL1 = 2
PC_JACOBI_ROWSUM = 3


class PCType:
    """The preconditioner method."""

    NONE = "none"
    JACOBI = "jacobi"
    SOR = "sor"
    LU = "lu"
    SHELL = "shell"
    BJACOBI = "bjacobi"
    ILU = "ilu"
    ICC = "icc"
    ASM = "asm"
    GASM = "gasm"
    KSP = "ksp"
    CHOLESKY = "cholesky"
    PYTHON = "python"
    MAT = "mat"
    FIELDSPLIT = "fieldsplit"
    LSC = "lsc"
    GAMG = "gamg"
    BDDC = "bddc"
    DEFLATION = "deflation"
    HPDDM = "hpddm"


SUPPORTED = {PCType.NONE, PCType.JACOBI, PCType.SOR, PCType.ILU, PCType.ICC}


class Side:
    """The preconditioning side."""

    LEFT = 0
    RIGHT = 1
    SYMMETRIC = 2
    DEFAULT = 0
    L = 0
    R = 1
    S = 2


class FailedReason:
    """The preconditioner failure reason."""

    NOERROR = 0
    FACTOR_NUMERIC_ZEROPIVOT = 1
    FACTOR_NUMERIC_SINGULAR = 2
    FACTOR_INF_OR_NAN = 3
    FACTOR_OTHER = 4
    OUT_OF_MEMORY = 5
    SETUP_ERROR = 6
    SETUP_FAILED = 7
    PROBLEM_TOO_SMALL = 8


class PC:
    """A preconditioner for a sequential AIJ matrix."""

    Type = PCType
    Side = Side
    FailedReason = FailedReason

    def __init__(self):
        self._type = PCType.NONE
        self._amat = None
        self._pmat = None
        self._setupcalled = False
        self._use_amat = False
        self._reuse = False
        self._failed_reason = FailedReason.NOERROR
        # factor options, mirroring PCFactorSet*
        self._levels = 0
        self._shift = 0.0
        self._zeropivot = 100.0
        self._shift_type = "positive_definite"
        self._ordering = MatOrderingType.NATURAL
        self._diagonal_fill = False
        # SOR options, mirroring PCSORSet*
        self._sor_omega = 1.0
        self._sor_fshift = 0.0
        self._sor_its = 1
        self._sor_lits = 1
        self._sor_sym = "symmetric"
        self._fix_diag = True
        self._use_abs = False
        self._jacobi_type = PC_JACOBI_DIAGONAL
        self._rowl1_scale = 1.0
        # scratch and factor storage
        self._scratch = None
        self._factor = None
        self._diag = None
        self._diagsqrt = None
        self._adiag = None
        self._dmark = None
        self._idiag = None
        self._mdiag = None
        self._sorwork = None
        self._pci = None
        self._pcf = None
        self._pstmp = None

    # ---------------------------------------------------------- lifecycle
    def create(self, comm=None) -> "PC":
        return self

    def destroy(self) -> "PC":
        self._factor = None
        self._setupcalled = False
        return self

    def setType(self, pc_type) -> None:
        if isinstance(pc_type, int):
            raise TypeError("expected a preconditioner type name")
        if pc_type not in SUPPORTED:
            raise NotImplementedError(
                f"PC type {pc_type!r} is not covered by mojo-petsc4py"
            )
        self._type = pc_type
        self._reset_factor()

    def getType(self) -> str:
        return self._type

    def setOperators(self, A: Mat = None, P: Mat = None) -> None:
        self._amat = A
        self._pmat = A if P is None else P
        self._reset_factor()

    def getOperators(self):
        return (self._amat, self._pmat)

    def setUseAmat(self, flag: bool) -> None:
        """Select which operator the apply uses; only `amat` is covered."""
        self._use_amat = bool(flag)

    def getUseAmat(self) -> bool:
        return self._use_amat

    def setReusePreconditioner(self, flag: bool) -> None:
        self._reuse = bool(flag)

    def setFailedReason(self, reason) -> None:
        self._failed_reason = reason

    def getFailedReason(self):
        return self._failed_reason

    def reset(self) -> None:
        self._setupcalled = False
        self._reset_factor()

    def _reset_factor(self):
        self._setupcalled = False
        self._factor = None
        self._diag = None
        self._diagsqrt = None

    # ---------------------------------------------------------- options
    def setFactorLevels(self, levels) -> None:
        self._levels = int(levels)
        self._setupcalled = False

    def setFactorShift(self, shift_type, amount=None) -> None:
        self._shift_type = str(shift_type)
        if amount is not None:
            self._shift = float(amount)
        self._setupcalled = False

    def setFactorPivot(self, flag) -> None:
        self._zeropivot = float(flag) if not isinstance(flag, bool) else (
            100.0 if flag else 0.0
        )
        self._diagonal_fill = bool(flag)
        self._setupcalled = False

    def setFactorOrdering(self, ordering) -> None:
        if ordering not in (MatOrderingType.NATURAL, MatOrderingType.RCM):
            raise NotImplementedError(
                f"matrix ordering {ordering!r} is not covered by mojo-petsc4py"
                " (natural and rcm are)"
            )
        self._ordering = ordering
        self._setupcalled = False

    def setFactorReorderForNonzeroDiagonal(self) -> None:
        self._diagonal_fill = True
        self._setupcalled = False

    def setSORSymmetric(self, flag) -> None:
        self._sor_sym = "symmetric" if flag else "forward"

    def setSOROmega(self, omega) -> None:
        self._sor_omega = float(omega)
        self._setupcalled = False

    def setSORIterations(self, its) -> None:
        self._sor_its = int(its)
        self._setupcalled = False

    def setSORShift(self, fshift) -> None:
        self._sor_fshift = float(fshift)

    def setJacobiType(self, jtype) -> None:
        self._jacobi_type = int(jtype)
        self._setupcalled = False

    def setJacobiUseAbs(self, flag) -> None:
        self._use_abs = bool(flag)
        self._setupcalled = False

    def setJacobiFixDiagonal(self, flag) -> None:
        self._fix_diag = bool(flag)
        self._setupcalled = False

    def setJacobiRowl1Scale(self, scale) -> None:
        self._rowl1_scale = float(scale)
        self._setupcalled = False

    def setFactorMatrixType(self, mtype) -> None:
        """Only `seqaij` is covered; accepted and ignored."""

    def setSPAIVerbose(self, level) -> None:
        raise NotImplementedError("spai is not covered by mojo-petsc4py")

    # ---------------------------------------------------------- setup
    def _scratch_vec(self, n: int) -> np.ndarray:
        if self._scratch is None or len(self._scratch) < 3 * n:
            self._scratch = np.zeros(3 * n, dtype=np.float64)
        return self._scratch

    def setUp(self) -> "PC":
        """Factor the matrix, if the chosen type needs it."""
        pmat = self._pmat
        if pmat is None:
            raise RuntimeError("PCSetOperators() has not been called")
        n = pmat._nrows
        if n != pmat._ncols:
            raise ValueError("the preconditioner matrix must be square")
        if self._setupcalled:
            return self
        if self._type == PCType.JACOBI:
            self._build_jacobi(pmat, n)
        elif self._type == PCType.SOR:
            self._build_sor(pmat, n)
        elif self._type in (PCType.ILU, PCType.ICC):
            self._build_factor(pmat, n)
        self._setupcalled = True
        return self

    def _build_jacobi(self, m: Mat, n: int):
        self._diag = np.zeros(max(n, 1), dtype=np.float64)
        didx = np.zeros(max(n, 1), dtype=np.float64)
        lib().mp_pc_jacobi_setup(
            addr(self._diag), self._jacobi_type, int(self._use_abs),
            int(self._fix_diag), self._rowl1_scale, addr(m.irow),
            addr(m.icol), addr(m.data), n, addr(didx),
        )
        self._diagsqrt = np.zeros(max(n, 1), dtype=np.float64)
        lib().mp_pc_jacobi_setup_symmetric(
            addr(self._diagsqrt), self._jacobi_type, addr(m.irow),
            addr(m.icol), addr(m.data), n,
        )

    def _build_sor(self, m: Mat, n: int):
        k = max(n, 1)
        self._dmark = np.zeros(k, dtype=np.int64)
        lib().mp_mat_get_diagonal_markers(
            addr(m.irow), addr(m.icol), addr(self._dmark), n
        )
        self._idiag = np.zeros(k, dtype=np.float64)
        self._mdiag = np.zeros(k, dtype=np.float64)
        self._sorwork = np.zeros(3 * k, dtype=np.float64)

    def _permutations(self, m: Mat, n: int):
        """(r, ic): the row ordering and the inverse column ordering."""
        if self._ordering == MatOrderingType.RCM and n > 0:
            perm = np.zeros(n, dtype=np.int64)
            level = np.zeros(n, dtype=np.int64)
            visited = np.zeros(n, dtype=np.int64)
            queue = np.zeros(n, dtype=np.int64)
            lib().mp_ordering_rcm(
                addr(perm), addr(level), addr(visited), addr(queue),
                addr(m.irow), addr(m.icol), n,
            )
            r = perm
        else:
            r = np.arange(n, dtype=np.int64)
        ic = np.empty(n, dtype=np.int64)
        ic[r] = np.arange(n, dtype=np.int64)
        return r, ic

    def _build_factor(self, m: Mat, n: int):
        if n == 0:
            # nothing to factor; the kernels must not be handed row 0 of an
            # empty irow
            return
        r, ic = self._permutations(m, n)
        adiag = np.zeros(n, dtype=np.int64)
        lib().mp_mat_get_diagonal_markers(
            addr(m.irow), addr(m.icol), addr(adiag), n
        )
        scratch = np.zeros(7 * n + 8, dtype=np.int64)
        if self._type == PCType.ILU:
            self._build_ilu(m, n, r, ic, adiag, scratch)
        else:
            self._build_icc(m, n, r, ic, adiag, scratch)

    def _build_ilu(self, m, n, r, ic, adiag, scratch):
        cap = max(4 * n, 4 * int(m._irow[-1]) + 4 * n, 16)
        while True:
            bi = np.zeros(n + 1, dtype=np.int64)
            bj = np.zeros(cap, dtype=np.int64)
            bjlvl = np.zeros(cap, dtype=np.int64)
            bdiag = np.zeros(n + 1, dtype=np.int64)
            if self._levels == 0 and np.array_equal(r, np.arange(n)):
                nnz = lib().mp_ilu_symbolic_ilu0(
                    addr(bi), addr(bj), addr(bdiag), addr(adiag),
                    addr(m.irow), addr(m.icol), n,
                )
                bjlvl = np.zeros(1, dtype=np.int64)
                contiguous = False
            else:
                bjsrc = np.zeros(cap, dtype=np.int64)
                nnz = lib().mp_ilu_symbolic(
                    addr(bi), addr(bjsrc), addr(bjlvl), addr(bdiag),
                    addr(m.irow), addr(m.icol), addr(r), addr(ic),
                    self._levels, cap, addr(scratch), n,
                )
                contiguous = True
            if nnz >= 0:
                if contiguous:
                    lib().mp_ilu_contiguous_lu(
                        addr(bjsrc), addr(bi), addr(bdiag), addr(bj), n
                    )
                break
            cap = -nnz + n
        ba = np.zeros(max(int(bdiag[0]) + 1, 1), dtype=np.float64)
        rtmp = np.zeros(n, dtype=np.float64)
        lib().mp_pc_ilu_numeric(
            addr(ba), addr(rtmp), addr(m.irow), addr(m.icol), addr(m.data),
            addr(bi), addr(bj), addr(bdiag), addr(r), addr(ic), n,
        )
        # the factor lives in the permuted coordinates, so the solution is
        # mapped back with the permutation itself, not with its inverse
        self._factor = {
            "bi": bi, "bj": bj, "ba": ba, "bdiag": bdiag, "r": r, "c": r,
        }

    def _build_icc(self, m, n, r, ic, adiag, scratch):
        cap = max(4 * n, 4 * int(m._irow[-1]) + 4 * n, 16)
        rinv = np.empty(n, dtype=np.int64)
        rinv[r] = np.arange(n, dtype=np.int64)
        while True:
            ui = np.zeros(n + 1, dtype=np.int64)
            uj = np.zeros(cap, dtype=np.int64)
            ujlvl = np.zeros(cap, dtype=np.int64)
            udiag = np.zeros(n, dtype=np.int64)
            nnz = lib().mp_icc_symbolic(
                addr(ui), addr(uj), addr(ujlvl), addr(udiag), addr(m.irow),
                addr(m.icol), addr(r), addr(rinv), self._levels, cap,
                addr(scratch), n,
            )
            if nnz >= 0:
                break
            cap = -nnz + n
        lib().mp_icc_contiguous_cholesky(addr(ui), addr(udiag), addr(uj), n)
        ua = np.zeros(max(int(ui[-1]), 1), dtype=np.float64)
        rtmp = np.zeros(n, dtype=np.float64)
        il = np.zeros(n, dtype=np.int64)
        c2r = np.zeros(n, dtype=np.int64)
        lib().mp_pc_icc_numeric(
            addr(ua), addr(rtmp), addr(il), addr(c2r), addr(m.irow),
            addr(m.icol), addr(m.data), addr(ui), addr(uj), addr(udiag),
            addr(r), addr(rinv), n,
        )
        # The IC factor stores L*U with only U present, so the forward solve
        # walks the transpose of U; build it once here.
        tirow, ticol, tdata = _csr_transpose(ui, uj, ua, n)
        self._factor = {
            "bi": ui, "bj": uj, "ba": ua, "bdiag": udiag, "r": r,
            "tirow": tirow, "ticol": ticol, "tdata": tdata,
        }

    # ---------------------------------------------------------- apply
    def _pci_block(self, n: int) -> np.ndarray:
        if self._pci is None or len(self._pci) < 12:
            self._pci = np.zeros(12, dtype=np.int64)
        pci = self._pci
        pci[:] = 0
        if self._type == PCType.NONE:
            pci[0] = _NONE
        elif self._type == PCType.JACOBI:
            pci[0] = _JACOBI
            pci[1] = self._diag.ctypes.data
            pci[2] = self._diagsqrt.ctypes.data
        elif self._type == PCType.SOR:
            m = self._pmat
            flag = (SOR_LOCAL_SYMMETRIC_SWEEP
                    if self._sor_sym == "symmetric" else SOR_FORWARD_SWEEP)
            pci[0] = _SOR
            pci[1] = m.irow.ctypes.data
            pci[2] = m.icol.ctypes.data
            pci[3] = m.data.ctypes.data
            pci[4] = self._dmark.ctypes.data
            pci[5] = self._idiag.ctypes.data
            pci[6] = self._mdiag.ctypes.data
            pci[7] = self._sorwork.ctypes.data
            pci[10] = flag | SOR_ZERO_INITIAL_GUESS
        else:
            f = self._factor
            if self._type == PCType.ILU:
                pci[0] = _ILU
                pci[1] = f["bi"].ctypes.data
                pci[2] = f["bj"].ctypes.data
                pci[3] = f["ba"].ctypes.data
                pci[4] = f["bdiag"].ctypes.data
                pci[5] = f["r"].ctypes.data
                pci[6] = f["c"].ctypes.data
            else:
                pci[0] = _ICC
                pci[1] = f["bi"].ctypes.data
                pci[2] = f["bj"].ctypes.data
                pci[3] = f["ba"].ctypes.data
                pci[4] = f["bdiag"].ctypes.data
                pci[5] = f["r"].ctypes.data
                pci[6] = f["tirow"].ctypes.data
                pci[7] = f["ticol"].ctypes.data
                pci[8] = f["tdata"].ctypes.data
        return pci

    def _pcf_block(self) -> np.ndarray:
        if self._pcf is None or len(self._pcf) < 2:
            self._pcf = np.zeros(2, dtype=np.float64)
        self._pcf[0] = self._sor_omega
        self._pcf[1] = self._sor_fshift
        return self._pcf

    def _workspace(self, n: int) -> np.ndarray:
        """The length-n scratch the triangular solves use."""
        if self._pstmp is None or len(self._pstmp) < n:
            self._pstmp = np.zeros(n, dtype=np.float64)
        return self._pstmp

    def _direct_apply(self, x: Vec, y: Vec):
        """The path that does not need a temporary: every covered type writes
        `y` from `x` only, so the factor is applied straight through."""
        m = self._pmat
        n = m._nrows
        if self._type == PCType.NONE:
            lib().mp_vec_copy(addr(x.getArray()), addr(y.getArray()), n)
        elif self._type == PCType.JACOBI:
            lib().mp_pc_jacobi_apply(
                addr(y.getArray()), addr(x.getArray()), addr(self._diag), n
            )
        elif self._type == PCType.SOR:
            lib().mp_pc_sor_apply(
                addr(x.getArray()), addr(y.getArray()), addr(m.irow),
                addr(m.icol), addr(m.data), addr(self._dmark),
                addr(self._idiag), addr(self._mdiag), addr(self._sorwork),
                self._sor_omega, self._sor_fshift,
                (SOR_LOCAL_SYMMETRIC_SWEEP if self._sor_sym == "symmetric"
                 else SOR_FORWARD_SWEEP) | SOR_ZERO_INITIAL_GUESS, n,
            )
        elif self._type == PCType.ILU:
            f = self._factor
            lib().mp_pc_ilu_apply(
                addr(x.getArray()), addr(y.getArray()), addr(self._workspace(n)),
                addr(f["bi"]), addr(f["bj"]), addr(f["ba"]), addr(f["bdiag"]),
                addr(f["r"]), addr(f["c"]), n,
            )
        else:
            f = self._factor
            lib().mp_pc_icc_apply(
                addr(x.getArray()), addr(y.getArray()), addr(self._workspace(n)),
                addr(f["bi"]), addr(f["bj"]), addr(f["ba"]), addr(f["bdiag"]),
                addr(f["r"]), addr(f["tirow"]), addr(f["ticol"]), addr(f["tdata"]),
                n,
            )

    def apply(self, x: Vec, y: Vec) -> None:
        """Apply the preconditioner to a vector."""
        self.setUp()
        n = self._pmat._nrows
        if len(x.getArray()) != n or len(y.getArray()) != n:
            raise ValueError(
                f"incompatible vector size: {n} is needed for both operands"
            )
        self._direct_apply(x, y)

    def applyTranspose(self, x: Vec, y: Vec) -> None:
        """Apply the transpose of the preconditioner.

        Only the SOR and Jacobi preconditioners are symmetric here, so this is
        `apply`; that matches upstream, whose `applytranspose` for those two is
        the same function object.
        """
        self.apply(x, y)

    def applySymmetricLeft(self, x: Vec, y: Vec) -> None:
        if self._type != PCType.JACOBI:
            self.apply(x, y)
            return
        self.setUp()
        n = self._pmat._nrows
        if len(x.getArray()) != n or len(y.getArray()) != n:
            raise ValueError(
                f"incompatible vector size: {n} is needed for both operands"
            )
        lib().mp_pc_jacobi_apply(
            addr(y.getArray()), addr(x.getArray()), addr(self._diagsqrt), n,
        )

    def applySymmetricRight(self, x: Vec, y: Vec) -> None:
        self.applySymmetricLeft(x, y)

    def getFactorMatrix(self) -> Mat:
        """Return the factor in CSR form."""
        self.setUp()
        f = self._factor
        if f is None:
            raise RuntimeError(f"PC type {self._type} has no factor matrix")
        if self._type == PCType.ILU:
            return _ilu_factor_mat(f)
        return _icc_factor_mat(f)


def _csr_transpose(row, col, val, n):
    """Transpose a CSR factor; the IC solve's forward sweep needs column access."""
    counts = np.zeros(n + 1, dtype=np.int64)
    for j in range(len(col)):
        counts[int(col[j]) + 1] += 1
    trow = np.cumsum(counts)
    total = int(trow[-1])
    tcol = np.zeros(max(total, 1), dtype=np.int64)
    tdata = np.zeros(max(total, 1), dtype=np.float64)
    fill = trow[:-1].copy()
    for i in range(n):
        for j in range(int(row[i]), int(row[i + 1])):
            c = int(col[j])
            k = int(fill[c])
            tcol[k] = i
            tdata[k] = val[j]
            fill[c] = k + 1
    return trow, tcol, tdata


def _ilu_factor_mat(f) -> Mat:
    """Rebuild the ILU factor as a square CSR matrix L*U, as PETSc's
    `PCFactorGetFactorMatrix` does."""
    bi, bj, ba, bdiag, r, c = (
        f["bi"], f["bj"], f["ba"], f["bdiag"], f["r"], f["c"]
    )
    n = len(bi) - 1
    dense = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(int(bi[i]), int(bi[i + 1])):
            row, col = int(r[i]), int(c[int(bj[j])])
            dense[row, col] = ba[j]
    for i in range(n - 1, -1, -1):
        for j in range(int(bdiag[i + 1]) + 1, int(bdiag[i])):
            row, col = int(r[i]), int(c[int(bj[j])])
            dense[row, col] = ba[j]
        piv = dense[int(r[i]), int(c[i])]
        if piv == 0.0:
            piv = 1.0
        dense[int(r[i]), int(c[i])] = 1.0 / piv
    return from_dense(dense)


def _icc_factor_mat(f) -> Mat:
    bi, bj, ba, bdiag, r = f["bi"], f["bj"], f["ba"], f["bdiag"], f["r"]
    n = len(bi) - 1
    dense = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        row = int(r[i])
        for j in range(int(bi[i]), int(bi[i + 1])):
            col = int(r[int(bj[j])])
            dense[col, row] = ba[j]  # L(i,j) = U(j,i)
    for i in range(n):
        d = int(bdiag[i])
        dense[int(r[i]), int(r[int(bj[d])])] = 1.0 / ba[d]
    return from_dense(dense)
