"""Krylov solvers, mirroring `petsc4py/PETSc/KSP.pyx`.

Covered types: `cg`, `gmres`, `fgmres`, `bcgs`.  The tolerances, the
convergence test, the residual history and the converged-reason reporting all
follow upstream; the iteration itself runs in Mojo.
"""

from __future__ import annotations

import math

import numpy as np

from .._lib import addr, lib
from .Mat import Mat
from .PC import PC
from .Vec import Vec, NormType

# PETSc default tolerances, KSPCreateDefault / petscimpl.h
DEFAULT_RTOL = 1e-5
DEFAULT_ATOL = 1e-50
DEFAULT_DTOL = 1.0e4
DEFAULT_MAX_IT = 1000
GMRES_DEFAULT_MAXK = 30


class KSPType:
    """KSP type."""

    RICHARDSON = "richardson"
    CG = "cg"
    GMRES = "gmres"
    FGMRES = "fgmres"
    BCGS = "bcgs"
    MINRES = "minres"
    PREONLY = "preonly"


SUPPORTED = {
    KSPType.CG, KSPType.GMRES, KSPType.FGMRES, KSPType.BCGS,
}


class NormType_:
    """KSP norm type."""

    NORM_DEFAULT = 0
    NORM_NONE = 1
    NORM_PRECONDITIONED = 2
    NORM_UNPRECONDITIONED = 3
    NORM_NATURAL = 4
    DEFAULT = 0
    NONE = 1
    NO = 1
    PRECONDITIONED = 2
    UNPRECONDITIONED = 3
    NATURAL = 4


class ConvergedReason:
    """KSP converged reason, with the `petscksp.h` ordinals."""

    CONVERGED_ITERATING = 0
    ITERATING = 0
    CONVERGED_RTOL_NORMAL_EQUATIONS = 1
    CONVERGED_RTOL = 2
    CONVERGED_ATOL = 3
    CONVERGED_ITS = 4
    CONVERGED_NEG_CURVE = 5
    CONVERGED_STEP_LENGTH = 6
    CONVERGED_HAPPY_BREAKDOWN = 7
    CONVERGED_ATOL_NORMAL_EQUATIONS = 9
    DIVERGED_NULL = -2
    DIVERGED_ITS = -3
    DIVERGED_DTOL = -4
    DIVERGED_BREAKDOWN = -5
    DIVERGED_BREAKDOWN_BICG = -6
    DIVERGED_NONSYMMETRIC = -7
    DIVERGED_INDEFINITE_PC = -8
    DIVERGED_NANORINF = -9
    DIVERGED_INDEFINITE_MAT = -10
    DIVERGED_PC_FAILED = -11
    # upstream's aliases
    DIVERGED_INDEFINITE_PRECONDITIONER = -8
    DIVERGED_PCSETUP_FAILED = -11
    DIVERGED_MAX_IT = -3


class GMRESCGSRefinementType:
    """GMRES CGS iterative refinement type."""

    REFINE_NEVER = 0
    REFINE_IFNEEDED = 1
    REFINE_ALWAYS = 2
    CGS_REFINE_NEVER = 0
    CGS_REFINE_IFNEEDED = 1
    CGS_REFINE_ALWAYS = 2


class GMRESOrthogonalizationType:
    """GMRES orthogonalization type."""

    CLASSICAL = 0
    MODIFIED = 1
    CGS = 0
    MGS = 1


class KSP:
    """A Krylov subspace solver for a sequential AIJ matrix."""

    Type = KSPType
    ConvergedReason = ConvergedReason
    NormType = NormType_
    GMRESCGSRefinementType = GMRESCGSRefinementType
    GMRESOrthogonalizationType = GMRESOrthogonalizationType

    def __init__(self):
        self._type = KSPType.GMRES
        self._amat = None
        self._pmat = None
        self._pc = PC()
        self._pc_side = 0
        self._normtype = 0
        self._rtol = DEFAULT_RTOL
        self._atol = DEFAULT_ATOL
        self._divtol = DEFAULT_DTOL
        self._max_it = DEFAULT_MAX_IT
        self._min_it = 0
        self._chknorm = -1
        self._convmaxits = 0
        self._guess_zero = 1
        self._restart = GMRES_DEFAULT_MAXK
        self._haptol = 1.0e-30
        self._breakdowntol = 0.1
        self._orthog = 0
        self._cgstype = 0
        self._its = 0
        self._rnorm = 0.0
        self._reason = 0
        self._history = None
        self._histlen = 0
        self._histreset = False
        self._rhs = None
        self._sol = None
        self._setupcalled = False
        self._cfg = None
        self._cfgi = None
        self._work = None
        self._vv = None
        self._prevec = None
        self._hs = None

    # ---------------------------------------------------------- lifecycle
    def create(self, comm=None) -> "KSP":
        return self

    def destroy(self) -> "KSP":
        self._history = None
        return self

    def setType(self, ksp_type) -> None:
        if isinstance(ksp_type, int):
            raise TypeError("expected a Krylov type name")
        if ksp_type not in SUPPORTED:
            raise NotImplementedError(
                f"KSP type {ksp_type!r} is not covered by mojo-petsc4py"
                f" ({sorted(SUPPORTED)} are)"
            )
        self._type = ksp_type
        self._setupcalled = False

    def getType(self) -> str:
        return self._type

    def setOperators(self, A: Mat = None, P: Mat = None) -> None:
        self._amat = A
        self._pmat = A if P is None else P
        self._pc.setOperators(self._amat, self._pmat)
        self._setupcalled = False

    def getOperators(self):
        return (self._amat, self._pmat)

    def setPC(self, pc: PC) -> None:
        self._pc = pc
        if self._pmat is not None:
            self._pc.setOperators(self._amat, self._pmat)
        self._setupcalled = False

    def getPC(self) -> PC:
        return self._pc

    def setPCSide(self, side) -> None:
        self._pc_side = int(side)

    def getPCSide(self):
        return self._pc_side

    def setNormType(self, normtype) -> None:
        self._normtype = int(normtype)

    def getNormType(self):
        return self._normtype

    def setTolerances(
        self, rtol=None, atol=None, divtol=None, max_it=None
    ) -> None:
        """Set the relative, absolute, divergence and maximum-iteration
        tolerances used by the default convergence test."""
        if rtol is not None:
            self._rtol = float(rtol)
        if atol is not None:
            self._atol = float(atol)
        if divtol is not None:
            self._divtol = float(divtol)
        if max_it is not None:
            self._max_it = int(max_it)
            self._convmaxits = 1

    def getTolerances(self):
        return (self._rtol, self._atol, self._divtol, self._max_it)

    def setMinimumIterations(self, min_it) -> None:
        self._min_it = int(min_it)

    def getMinimumIterations(self) -> int:
        return self._min_it

    def setCheckNorm(self, flag) -> None:
        """Set `ksp->chknorm`, the iteration from which norms are recomputed."""
        self._chknorm = int(flag)

    def setInitialGuessNonzero(self, flag) -> None:
        self._guess_zero = 0 if flag else 1

    def getInitialGuessNonzero(self) -> bool:
        return not self._guess_zero

    def setGMRESRestart(self, restart) -> None:
        self._restart = int(restart)
        self._setupcalled = False

    def setGMRESHapTol(self, haptol) -> None:
        self._haptol = float(haptol)

    def setGMRESBreakdownTolerance(self, tol) -> None:
        self._breakdowntol = float(tol)

    def setGMRESOrthogonalization(self, orthog) -> None:
        self._orthog = int(orthog)

    def setGMRESCGSRefinementType(self, cgstype) -> None:
        self._cgstype = int(cgstype)

    def setConvergenceHistory(
        self, length: int | None = None, reset: bool = False
    ) -> None:
        """Set the array used to hold the residual history.

        `length` is the size of the history array (upstream allocates 10000 when
        it is `None`); `reset` clears the counter before each solve.
        """
        size = 10000 if length is None or length is True else int(length)
        if size < 0:
            size = 10000
        self._histlen = size
        self._histreset = bool(reset)
        self._history = np.zeros(size, dtype=np.float64)

    def getConvergenceHistory(self) -> np.ndarray:
        """Return the array holding the residual history."""
        if self._history is None:
            return np.zeros(0, dtype=np.float64)
        return self._history

    def getIterationNumber(self) -> int:
        return self._its

    def getResidualNorm(self) -> float:
        return self._rnorm

    def getConvergedReason(self):
        return self._reason

    def getRhs(self) -> Vec:
        return self._rhs

    def getSolution(self) -> Vec:
        return self._sol

    def setUp(self) -> None:
        if self._amat is None:
            raise RuntimeError("KSP.setOperators() has not been called")
        self._pc.setUp()
        self._setupcalled = True

    # ---------------------------------------------------------- scratch
    # The driver keeps every buffer it needs in caller-owned arrays, so a solve
    # whose size, restart and tolerances are unchanged reuses the ones it
    # already has instead of allocating and zeroing several megabytes per
    # call.  PETSc allocates these in KSPSetUp for the same reason.
    def _ensure(self, nwork: int, nvec: int, hs_size: int):
        """Grow the driver's scratch if it no longer fits, and hand it back.

        A vector block that a previous solve did not write end to end is zeroed
        on reuse; an exact fit is left alone, since in that case the driver
        overwrote every entry it went on to read.  PETSc allocates the same
        blocks in `KSPSetUp` and reuses them for every solve, which is what
        this mirrors.
        """
        if self._work is None or len(self._work) < nwork:
            self._work = np.zeros(nwork, dtype=np.float64)
        elif len(self._work) > nwork:
            self._work[:nwork] = 0.0
        if nvec:
            if self._vv is None or len(self._vv) < nvec:
                self._vv = np.zeros(nvec, dtype=np.float64)
            elif len(self._vv) > nvec:
                self._vv[:nvec] = 0.0
        if hs_size:
            if self._hs is None or len(self._hs) < hs_size:
                self._hs = np.zeros(hs_size, dtype=np.float64)
            elif len(self._hs) > hs_size:
                self._hs[:hs_size] = 0.0
        return self._work, self._vv, self._hs

    # ---------------------------------------------------------- solve
    def solve(self, b: Vec, x: Vec) -> None:
        """Solve A x = b, taking the right-hand side from `b` and returning the
        solution in `x`.

        With the default `setInitialGuessNonzero(False)` the contents of `x` on
        entry are ignored and `x` is zeroed first, which is what PETSc's
        `KSPSolve` does before dispatching to the Krylov method.
        """
        if self._amat is None:
            raise RuntimeError("KSP.setOperators() has not been called")
        n = self._amat._nrows
        if self._amat._nrows != self._amat._ncols:
            raise ValueError("the solver matrix must be square")
        if len(b.getArray()) != n or len(x.getArray()) != n:
            raise ValueError(
                f"incompatible vector size: {n} is needed for both operands"
            )
        if not self._setupcalled:
            self.setUp()
        if self._guess_zero:
            x.zeroEntries()
        self._pc.setUp()
        self._rhs = b
        self._sol = x

        self._resolve_normtype()
        cfg, cfgi = self._state(n)
        pcf = self._pc._pcf_block()
        pstmp = self._pc._workspace(n)
        am = self._amat

        if self._type in (KSPType.GMRES, KSPType.FGMRES):
            mk = self._restart
            # HH (max_k + 2)*(max_k + 1), HES (max_k + 1)^2, CC and SS
            # max_k + 1 each, RS and ORTHOG max_k + 2 each; see ksp.mojo
            hs_size = (
                (mk + 2) * (mk + 1) + (mk + 1) * (mk + 1)
                + 2 * (mk + 1) + 2 * (mk + 2)
            )
            work, vv, hs = self._ensure(2 * n, (mk + 1) * n, hs_size)
            if self._type == KSPType.GMRES:
                lib().mp_ksp_solve_gmres(
                    n, addr(am.irow), addr(am.icol), addr(am.data),
                    addr(self._pc._pci_block(n)), addr(pcf), addr(pstmp),
                    addr(work), addr(vv), addr(hs), addr(cfg), addr(cfgi),
                    addr(b.getArray()), addr(x.getArray()),
                )
            else:
                prevec = self._ensure(0, (mk + 2) * n, 0)[1]
                lib().mp_ksp_solve_fgmres(
                    n, addr(am.irow), addr(am.icol), addr(am.data),
                    addr(self._pc._pci_block(n)), addr(pcf), addr(pstmp),
                    addr(work), addr(vv), addr(prevec), addr(hs), addr(cfg),
                    addr(cfgi), addr(b.getArray()), addr(x.getArray()),
                )
        else:
            work, _vv, _hs = self._ensure(7 * n, 0, 0)
            entry = (
                lib().mp_ksp_solve_cg if self._type == KSPType.CG
                else lib().mp_ksp_solve_bcgs
            )
            entry(
                n, addr(am.irow), addr(am.icol), addr(am.data),
                addr(self._pc._pci_block(n)), addr(pcf), addr(pstmp),
                addr(work), addr(cfg), addr(cfgi), addr(b.getArray()),
                addr(x.getArray()),
            )

        self._its = int(cfgi[10])
        self._reason = int(cfgi[11])
        self._rnorm = float(cfg[7])
        full = np.zeros(getattr(self, "_histlen", 0), dtype=np.float64)
        k = min(len(full), len(cfg) - 9)
        full[:k] = cfg[9:9 + k]
        self._history = full

    def _resolve_normtype(self):
        """KSP_NORM_DEFAULT resolves to KSP_NORM_PRECONDITIONED for every
        covered type, which is what `KSPSetDefaults` does at `KSPSetUp`."""
        if self._normtype == NormType_.NORM_DEFAULT:
            self._normtype = NormType_.NORM_PRECONDITIONED

    def _state(self, n: int):
        """Build the scalar blocks the Mojo driver reads; see the ksp.mojo
        docstring for the layout."""
        size = 16 + self._max_it + 2
        if self._cfg is None or len(self._cfg) < size:
            self._cfg = np.zeros(size, dtype=np.float64)
        cfg = self._cfg
        # the residual history is the one part of this block the driver only
        # appends to, so it is cleared rather than left from the last solve
        cfg[9:9 + self._max_it + 2] = 0.0
        cfg[0] = self._rtol
        cfg[1] = self._atol
        cfg[2] = self._divtol
        cfg[3] = self._haptol
        cfg[4] = self._breakdowntol
        cfg[6] = 0.0
        if self._cfgi is None:
            self._cfgi = np.zeros(16, dtype=np.int64)
        cfgi = self._cfgi
        cfgi[0] = self._normtype
        cfgi[1] = self._pc_side
        cfgi[2] = self._guess_zero
        cfgi[3] = self._min_it
        cfgi[4] = self._chknorm
        cfgi[5] = self._convmaxits
        cfgi[6] = self._max_it
        cfgi[7] = self._restart
        cfgi[8] = self._orthog
        cfgi[9] = self._cgstype

        # KSPConvergedDefault seeds rnorm0 and ttol on its first call; its
        # `!guess_zero` branch measures the right-hand side through the
        # preconditioner, which the kernel cannot do, so the value is computed
        # here into cfg[12] and flagged in cfgi[12].
        cfg[12] = self._rhs_norm()
        cfg[14] = 0.0
        cfgi[12] = 0 if self._guess_zero else 1
        return cfg, cfgi

    def _rhs_norm(self) -> float:
        """`KSPConvergedDefault`'s right-hand-side norm, for a nonzero guess."""
        b = self._rhs
        if b is None or self._normtype == NormType_.NORM_NONE:
            return 0.0
        if self._normtype == NormType_.UNPRECONDITIONED or self._pc_side == 1:
            return float(np.linalg.norm(b.getArray()))
        z = b.copy()
        self._pc.apply(b, z)
        if self._normtype == NormType_.PRECONDITIONED:
            return float(np.linalg.norm(z.getArray()))
        return math.sqrt(abs(z.dot(b)))
