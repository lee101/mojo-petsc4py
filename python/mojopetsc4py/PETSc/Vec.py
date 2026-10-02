"""Vector types and operations, mirroring `petsc4py/PETSc/Vec.pyx`.

Only the sequential real-double subset is covered; see the README.
"""

from __future__ import annotations

import numpy as np

from .. import _lib
from .._lib import addr, f64, lib


class VecType:
    """The vector type."""

    SEQ = "seq"
    STANDARD = "seq"
    MPI = "mpi"
    SHARED = "shared"
    NEST = "nest"


class NormType:
    """petscvec.h NormType."""

    NORM_1 = 0
    NORM_2 = 1
    NORM_FROBENIUS = 2
    NORM_INFINITY = 3
    NORM_1_AND_2 = 4
    # upstream aliases
    N_1 = NORM_1
    NORM_MAX = NORM_INFINITY


class Vec:
    """A sequential PETSc vector.

    The array is the whole local storage, exactly as `VecGetArray` hands it
    back: `bsize * n` float64 entries, block-major.
    """

    Type = VecType

    def __init__(self):
        self._array = None
        self._bsize = 1

    # ---------------------------------------------------------- creation
    def create(self) -> "Vec":
        self._array = None
        self._bsize = 1
        return self

    def createSeq(self, size, bsize=None, comm=None) -> "Vec":
        """Create a sequential `Type.SEQ` vector.

        `size` is the total number of scalar entries; `bsize` is the block size
        (`None` means 1) and must divide it, exactly as upstream's
        `Vec_Sizes`/`Sys_Sizes` require.
        """
        n = int(size[0] if isinstance(size, (tuple, list)) else size)
        bs = 1 if bsize is None else int(bsize)
        if bs < 1:
            raise ValueError(f"invalid block size {bs}")
        if n % bs:
            raise ValueError(
                f"global size {n} not divisible by block size {bs}"
            )
        self._array = np.zeros(n, dtype=np.float64)
        self._bsize = bs
        return self

    def createWithArray(self, array, bsize=None, comm=None) -> "Vec":
        arr = f64(array, copy=True)
        self._array = arr
        self._bsize = 1 if bsize is None else int(bsize)
        if len(arr) % self._bsize:
            raise ValueError(
                f"array length {len(arr)} is not a multiple of the block size"
                f" {self._bsize}"
            )
        return self

    def duplicate(self, array=None) -> "Vec":
        vec = Vec()
        vec._bsize = self._bsize
        if array is None:
            vec._array = np.zeros_like(self._array)
        else:
            arr = f64(array, copy=True)
            if len(arr) != self.getSize():
                raise ValueError(
                    f"incompatible array size: {len(arr)} != {self.getSize()}"
                )
            vec._array = arr
        return vec

    def copy(self, result=None) -> "Vec":
        target = self.duplicate() if result is None else result
        self._peer(target)
        lib().mp_vec_copy(addr(self._array), addr(target._array), self.getSize())
        return target

    def chop(self, tol) -> None:
        """Set entries with magnitude below `tol` to zero, the rest to +/-1."""
        lib().mp_vec_chop(addr(self._check()), self._n(), float(tol))

    def destroy(self) -> "Vec":
        self._array = None
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()

    # ---------------------------------------------------------- accessors
    def _check(self):
        if self._array is None:
            raise RuntimeError("Vec has not been created")
        return self._array

    def getArray(self, readonly: bool = False) -> np.ndarray:
        """Return the local portion of the vector as an `ndarray`."""
        return self._check()

    def setArray(self, array) -> None:
        """Set the values of the local portion of the vector."""
        self._check()
        arr = f64(array, copy=True)
        if len(arr) != len(self._array):
            raise ValueError(
                f"incompatible array size: {len(arr)} != {len(self._array)}"
            )
        self._array[:] = arr

    def placeArray(self, array) -> None:
        self.setArray(array)

    def getSize(self) -> int:
        return len(self._array)

    def getLocalSize(self) -> int:
        return len(self._array)

    def getSizes(self):
        n = len(self._array)
        return (n, n)

    def getBlockSize(self) -> int:
        return self._bsize

    def getOwnershipRange(self):
        return (0, len(self._array))

    def getType(self) -> str:
        return "seq"

    def getValue(self, index: int) -> float:
        return float(self._check()[index])

    def setValue(self, index: int, value: float) -> None:
        self._check()[index] = value

    def getValues(self, indices) -> np.ndarray:
        return self._check()[np.asarray(indices, dtype=np.int64)]

    def setValues(self, indices, values, addv=None) -> None:
        idx = np.asarray(indices, dtype=np.int64)
        val = f64(values)
        if addv is None:
            self._check()[idx] = val
        else:
            self._check()[idx] += val

    # ---------------------------------------------------------- reductions
    def _norm_slot(self):
        return np.zeros(2, dtype=np.float64)

    def norm(self, norm_type=None):
        """Compute the vector norm.

        A 2-tuple is returned if `NormType.NORM_1_AND_2` is specified.
        """
        ntype = NormType.NORM_2 if norm_type is None else int(norm_type)
        z = self._norm_slot()
        lib().mp_vec_norm(
            addr(self._check()), ntype, addr(z), len(self._array)
        )
        if ntype != NormType.NORM_1_AND_2:
            return float(z[0])
        return (float(z[0]), float(z[1]))

    def normBegin(self, norm_type=None) -> None:
        return None

    def normEnd(self, norm_type=None):
        return self.norm(norm_type)

    def dot(self, other: "Vec") -> float:
        val = np.zeros(1, dtype=np.float64)
        self._peer(other)
        lib().mp_vec_dot(
            addr(self._check()), addr(other._check()), addr(val), len(self._array)
        )
        return float(val[0])

    def tDot(self, other: "Vec") -> float:
        """The indefinite dot product y^T x, which skips the conjugate.

        For a real build this is `dot` with the arguments swapped, as upstream
        notes; it is kept separate so a complex build has somewhere to differ.
        """
        val = np.zeros(1, dtype=np.float64)
        self._peer(other)
        lib().mp_vec_tdot(
            addr(self._check()), addr(other._check()), addr(val), len(self._array)
        )
        return float(val[0])

    def dotNorm2(self, other: "Vec"):
        """Return the dot product with `vec` and the squared norm of self.

        Upstream petsc4py 3.25.5 raises `TypeError` from this method on every
        call (`toReal(float)` in Vec.pyx).  The documented divergence is that
        this port returns the pair instead.
        """
        sval = np.zeros(1, dtype=np.float64)
        rval = np.zeros(1, dtype=np.float64)
        self._peer(other)
        lib().mp_vec_dot_norm2(
            addr(self._check()), addr(other._check()), addr(sval), addr(rval),
            len(self._array),
        )
        return (float(sval[0]), float(rval[0]))

    def sum(self) -> float:
        val = np.zeros(1, dtype=np.float64)
        lib().mp_vec_sum(addr(self._check()), addr(val), len(self._array))
        return float(val[0])

    def mean(self) -> float:
        val = np.zeros(1, dtype=np.float64)
        lib().mp_vec_mean(addr(self._check()), addr(val), len(self._array))
        return float(val[0])

    def min(self):
        """Return the vector entry with minimum real part and its location."""
        idx = np.zeros(1, dtype=np.int64)
        val = np.zeros(1, dtype=np.float64)
        lib().mp_vec_min(
            addr(self._check()), addr(idx), addr(val), len(self._array)
        )
        return (int(idx[0]), float(val[0]))

    def max(self):
        """Return the vector entry with maximum real part and its location."""
        idx = np.zeros(1, dtype=np.int64)
        val = np.zeros(1, dtype=np.float64)
        lib().mp_vec_max(
            addr(self._check()), addr(idx), addr(val), len(self._array)
        )
        return (int(idx[0]), float(val[0]))

    def normalize(self) -> float:
        """Normalize by the 2-norm, returning the norm before normalizing."""
        val = np.zeros(1, dtype=np.float64)
        lib().mp_vec_normalize(
            addr(self._check()), addr(val), len(self._array)
        )
        return float(val[0])

    def maxPointwiseDivide(self, other: "Vec") -> float:
        val = np.zeros(1, dtype=np.float64)
        self._peer(other)
        lib().mp_vec_max_pointwise_divide(
            addr(self._check()), addr(other._check()), addr(val), len(self._array)
        )
        return float(val[0])

    # ---------------------------------------------------------- elementwise
    def _n(self):
        return len(self._check())

    def _peer(self, *others: "Vec") -> None:
        """Every two-vector kernel reads `n` entries of each operand, so a
        shorter operand would be read past its end."""
        n = self._n()
        for other in others:
            if len(other._check()) != n:
                raise ValueError(
                    f"incompatible sizes: {n} and {len(other._check())}"
                )

    def zeroEntries(self) -> None:
        lib().mp_vec_zero_entries(addr(self._check()), self._n())

    def set(self, alpha) -> None:
        lib().mp_vec_set(addr(self._check()), float(alpha), self._n())

    def scale(self, alpha) -> None:
        lib().mp_vec_scale(addr(self._check()), float(alpha), self._n())

    def shift(self, alpha) -> None:
        lib().mp_vec_shift(addr(self._check()), float(alpha), self._n())

    def swap(self, other: "Vec") -> None:
        self._peer(other)
        lib().mp_vec_swap(addr(self._check()), addr(other._check()), self._n())

    def axpy(self, alpha, other: "Vec") -> None:
        """Compute and store y = alpha*x + y."""
        self._peer(other)
        lib().mp_vec_axpy(
            addr(self._check()), float(alpha), addr(other._check()), self._n()
        )

    def aypx(self, alpha, other: "Vec") -> None:
        """Compute and store y = x + alpha*y."""
        self._peer(other)
        lib().mp_vec_aypx(
            addr(self._check()), float(alpha), addr(other._check()), self._n()
        )

    def axpby(self, alpha, beta, other: "Vec") -> None:
        """Compute and store y = alpha*x + beta*y."""
        self._peer(other)
        lib().mp_vec_axpby(
            addr(self._check()), float(alpha), float(beta), addr(other._check()),
            self._n(),
        )

    def axpbypcz(self, alpha, beta, gamma, other, another) -> None:
        self._peer(other, another)
        lib().mp_vec_axpbypcz(
            addr(self._check()), float(alpha), float(beta), float(gamma),
            addr(other._check()), addr(another._check()), self._n(),
        )

    def waxpy(self, alpha, other: "Vec", another: "Vec") -> None:
        """Compute and store w = alpha*x + y."""
        self._peer(other, another)
        lib().mp_vec_waxpy(
            addr(self._check()), float(alpha), addr(other._check()),
            addr(another._check()), self._n(),
        )

    def maxpy(self, alphas, vecs) -> None:
        """Compute and store y = sum(alpha[i]*vecs[i]) + y."""
        alph = f64(alphas)
        if len(alph) != len(vecs):
            raise ValueError("alphas and vecs must have the same length")
        for v in vecs:
            self._peer(v)
        addrs = np.array([v._check().ctypes.data for v in vecs], dtype=np.int64)
        lib().mp_vec_maxpy(
            addr(self._check()), len(alph), addr(alph), addr(addrs), self._n()
        )

    def pointwiseMult(self, other: "Vec", another: "Vec") -> None:
        self._peer(other, another)
        lib().mp_vec_pointwise_mult(
            addr(self._check()), addr(other._check()), addr(another._check()),
            self._n(),
        )

    def pointwiseDivide(self, other: "Vec", another: "Vec") -> None:
        self._peer(other, another)
        lib().mp_vec_pointwise_divide(
            addr(self._check()), addr(other._check()), addr(another._check()),
            self._n(),
        )

    def pointwiseMin(self, other: "Vec", another: "Vec") -> None:
        self._peer(other, another)
        lib().mp_vec_pointwise_min(
            addr(self._check()), addr(other._check()), addr(another._check()),
            self._n(),
        )

    def pointwiseMax(self, other: "Vec", another: "Vec") -> None:
        self._peer(other, another)
        lib().mp_vec_pointwise_max(
            addr(self._check()), addr(other._check()), addr(another._check()),
            self._n(),
        )

    def pointwiseMaxAbs(self, other: "Vec", another: "Vec") -> None:
        self._peer(other, another)
        lib().mp_vec_pointwise_max_abs(
            addr(self._check()), addr(other._check()), addr(another._check()),
            self._n(),
        )

    def reciprocal(self) -> None:
        lib().mp_vec_reciprocal(addr(self._check()), self._n())

    def exp(self) -> None:
        lib().mp_vec_exp(addr(self._check()), self._n())

    def log(self) -> None:
        lib().mp_vec_log(addr(self._check()), self._n())

    def sqrtabs(self) -> None:
        lib().mp_vec_sqrtabs(addr(self._check()), self._n())

    def abs(self) -> None:
        lib().mp_vec_abs(addr(self._check()), self._n())

    def conjugate(self) -> None:
        lib().mp_vec_conjugate(addr(self._check()), self._n())

    # ---------------------------------------------------------- operators
    def __neg__(self):
        out = self.duplicate()
        out.set(-1.0)
        out.pointwiseMult(self, out)
        return out

    def __pos__(self):
        return self.copy()

    def __abs__(self):
        out = self.duplicate()
        out.abs()
        return out

    def __add__(self, other):
        if not isinstance(other, Vec):
            return NotImplemented
        out = self.copy()
        out.axpy(1.0, other)
        return out

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, Vec):
            return NotImplemented
        out = self.duplicate()
        out.axpy(-1.0, other)
        return out

    def __rsub__(self, other):
        if not isinstance(other, Vec):
            return NotImplemented
        return other - self

    def __mul__(self, other):
        if not isinstance(other, Vec):
            return NotImplemented
        out = self.copy()
        out.pointwiseMult(self, other)
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, Vec):
            return NotImplemented
        out = self.duplicate()
        out.pointwiseDivide(self, other)
        return out

    def __matmul__(self, other):
        return self.dot(other)

    def __getitem__(self, i):
        return self.getArray()[i]

    def __setitem__(self, i, v):
        self._check()[i] = v
