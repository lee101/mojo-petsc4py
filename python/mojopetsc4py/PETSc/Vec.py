"""Vector types and operations, mirroring `petsc4py/PETSc/Vec.pyx`.

Only the sequential real-double subset is covered; see the README.


Transcribed from PETSc 3.25.5 / petsc4py 3.25.5 (BSD-2-Clause); see NOTICE
at the repository root for their copyright notices, and LICENSE for this
file's own MIT licence.
"""

from __future__ import annotations

import numpy as np

from .. import _lib
from .._lib import AddrBuf, addr, f64, lib


class _Array:
    """The vector's storage and its data address, resolved once.

    `array` is what the kernel entry points hand to the FFI and `add` is the same
    buffer's address; both are rebound together through `set`, so an address can
    never refer to a previous allocation.
    """

    __slots__ = ("array", "add")

    def __init__(self, array):
        self.array = array
        self.add = array.ctypes.data

    def __len__(self):
        return len(self.array)


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
        self._buf = None
        self._bsize = 1
        self._out = None
        self._out_addr = 0

    # ---------------------------------------------------------- creation
    def create(self) -> "Vec":
        self._buf = None
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
        self._buf = _Array(np.zeros(n, dtype=np.float64))
        self._bsize = bs
        return self

    def createWithArray(self, array, bsize=None, comm=None) -> "Vec":
        arr = f64(array, copy=True)
        self._buf = _Array(arr)
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
            vec._buf = _Array(np.zeros_like(self._check()))
        else:
            arr = f64(array, copy=True)
            if len(arr) != self.getSize():
                raise ValueError(
                    f"incompatible array size: {len(arr)} != {self.getSize()}"
                )
            vec._buf = _Array(arr)
        return vec

    def copy(self, result=None) -> "Vec":
        target = self.duplicate() if result is None else result
        self._peer(target)
        lib().mp_vec_copy(self._ptr(), target._ptr(), self.getSize())
        return target

    def chop(self, tol) -> None:
        """Zero the entries whose magnitude is below `tol`; the rest are left
        unchanged, which is `VecChop` (`VecChop_Seq`)."""
        lib().mp_vec_chop(self._ptr(), self._n(), float(tol))

    def destroy(self) -> "Vec":
        self._buf = None
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()

    # ---------------------------------------------------------- accessors
    def _check(self):
        if self._buf is None:
            raise RuntimeError("Vec has not been created")
        return self._buf.array

    def _ptr(self):
        """The cached data address of the local portion."""
        return self._check().ctypes.data

    def _size(self):
        """The number of scalar entries, without building a view to ask."""
        self._check()
        return len(self._buf.array)

    def _seen(self):
        """The ndarray the FFI takes for this vector, and its address."""
        return self._buf.array, self._buf.add

    def getArray(self, readonly: bool = False) -> np.ndarray:
        """Return the local portion of the vector as an `ndarray`."""
        return self._check()

    def setArray(self, array) -> None:
        """Set the values of the local portion of the vector."""
        cur = self._check()
        arr = f64(array, copy=True)
        if len(arr) != len(cur):
            raise ValueError(
                f"incompatible array size: {len(arr)} != {len(cur)}"
            )
        cur[:] = arr

    def placeArray(self, array) -> None:
        self.setArray(array)

    def getSize(self) -> int:
        return self._size()

    def getLocalSize(self) -> int:
        return self._size()

    def getSizes(self):
        n = self._size()
        return (n, n)

    def getBlockSize(self) -> int:
        return self._bsize

    def getOwnershipRange(self):
        return (0, self._size())

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

    def _result_slot(self):
        """A length-1 float64 slot for a reduction's result, and its address.

        The methods that return the slot itself keep upstream's fresh
        allocation; the ones that return a plain float have nothing to alias,
        and reusing the slot saves an allocation that measured at about the cost
        of the kernel it receives.  It is per-vector scratch, so two threads
        driving two different vectors do not share it, and it is only ever read
        immediately after the call that fills it.
        """
        out = self._out
        if out is None:
            out = np.zeros(1, dtype=np.float64)
            self._out = out
            self._out_addr = AddrBuf(out).addr
        return out, self._out_addr

    def norm(self, norm_type=None):
        """Compute the vector norm.

        A 2-tuple is returned if `NormType.NORM_1_AND_2` is specified.
        """
        ntype = NormType.NORM_2 if norm_type is None else int(norm_type)
        z = self._norm_slot()
        lib().mp_vec_norm(
            self._ptr(), ntype, addr(z), self._size()
        )
        if ntype != NormType.NORM_1_AND_2:
            return float(z[0])
        return (float(z[0]), float(z[1]))

    def normBegin(self, norm_type=None) -> None:
        return None

    def normEnd(self, norm_type=None):
        return self.norm(norm_type)

    def dot(self, other: "Vec") -> float:
        val, vaddr = self._result_slot()
        self._peer(other)
        lib().mp_vec_dot(self._ptr(), other._ptr(), vaddr, self._size())
        return float(val[0])

    def tDot(self, other: "Vec") -> float:
        """The indefinite dot product y^T x, which skips the conjugate.

        For a real build this is `dot` with the arguments swapped, as upstream
        notes; it is kept separate so a complex build has somewhere to differ.
        """
        val, vaddr = self._result_slot()
        self._peer(other)
        lib().mp_vec_tdot(self._ptr(), other._ptr(), vaddr, self._size())
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
            self._ptr(), other._ptr(), addr(sval), addr(rval),
            self._size(),
        )
        return (float(sval[0]), float(rval[0]))

    def sum(self) -> float:
        val, vaddr = self._result_slot()
        lib().mp_vec_sum(self._ptr(), vaddr, self._size())
        return float(val[0])

    def mean(self) -> float:
        val, vaddr = self._result_slot()
        lib().mp_vec_mean(self._ptr(), vaddr, self._size())
        return float(val[0])

    def min(self):
        """Return the vector entry with minimum real part and its location."""
        idx = np.zeros(1, dtype=np.int64)
        val = np.zeros(1, dtype=np.float64)
        lib().mp_vec_min(
            self._ptr(), addr(idx), addr(val), self._size()
        )
        return (int(idx[0]), float(val[0]))

    def max(self):
        """Return the vector entry with maximum real part and its location."""
        idx = np.zeros(1, dtype=np.int64)
        val = np.zeros(1, dtype=np.float64)
        lib().mp_vec_max(
            self._ptr(), addr(idx), addr(val), self._size()
        )
        return (int(idx[0]), float(val[0]))

    def normalize(self) -> float:
        """Normalize by the 2-norm, returning the norm before normalizing."""
        val, vaddr = self._result_slot()
        lib().mp_vec_normalize(self._ptr(), vaddr, self._size())
        return float(val[0])

    def maxPointwiseDivide(self, other: "Vec") -> float:
        val, vaddr = self._result_slot()
        self._peer(other)
        lib().mp_vec_max_pointwise_divide(
            self._ptr(), other._ptr(), vaddr, self._size()
        )
        return float(val[0])

    # ---------------------------------------------------------- elementwise
    def _n(self):
        return self._size()

    def _peer(self, *others: "Vec") -> None:
        """Every two-vector kernel reads `n` entries of each operand, so a
        shorter operand would be read past its end."""
        n = self._n()
        for other in others:
            if other._size() != n:
                raise ValueError(
                    f"incompatible sizes: {n} and {other._size()}"
                )

    def zeroEntries(self) -> None:
        lib().mp_vec_zero_entries(self._ptr(), self._n())

    def set(self, alpha) -> None:
        lib().mp_vec_set(self._ptr(), float(alpha), self._n())

    def scale(self, alpha) -> None:
        lib().mp_vec_scale(self._ptr(), float(alpha), self._n())

    def shift(self, alpha) -> None:
        lib().mp_vec_shift(self._ptr(), float(alpha), self._n())

    def swap(self, other: "Vec") -> None:
        self._peer(other)
        lib().mp_vec_swap(self._ptr(), other._ptr(), self._n())

    def axpy(self, alpha, other: "Vec") -> None:
        """Compute and store y = alpha*x + y."""
        self._peer(other)
        lib().mp_vec_axpy(
            self._ptr(), float(alpha), other._ptr(), self._n()
        )

    def aypx(self, alpha, other: "Vec") -> None:
        """Compute and store y = x + alpha*y."""
        self._peer(other)
        lib().mp_vec_aypx(
            self._ptr(), float(alpha), other._ptr(), self._n()
        )

    def axpby(self, alpha, beta, other: "Vec") -> None:
        """Compute and store y = alpha*x + beta*y."""
        self._peer(other)
        lib().mp_vec_axpby(
            self._ptr(), float(alpha), float(beta), other._ptr(),
            self._n(),
        )

    def axpbypcz(self, alpha, beta, gamma, other, another) -> None:
        self._peer(other, another)
        lib().mp_vec_axpbypcz(
            self._ptr(), float(alpha), float(beta), float(gamma),
            other._ptr(), another._ptr(), self._n(),
        )

    def waxpy(self, alpha, other: "Vec", another: "Vec") -> None:
        """Compute and store w = alpha*x + y."""
        self._peer(other, another)
        lib().mp_vec_waxpy(
            self._ptr(), float(alpha), other._ptr(),
            another._ptr(), self._n(),
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
            self._ptr(), len(alph), addr(alph), addr(addrs), self._n()
        )

    def pointwiseMult(self, other: "Vec", another: "Vec") -> None:
        self._peer(other, another)
        lib().mp_vec_pointwise_mult(
            self._ptr(), other._ptr(), another._ptr(),
            self._n(),
        )

    def pointwiseDivide(self, other: "Vec", another: "Vec") -> None:
        self._peer(other, another)
        lib().mp_vec_pointwise_divide(
            self._ptr(), other._ptr(), another._ptr(),
            self._n(),
        )

    def pointwiseMin(self, other: "Vec", another: "Vec") -> None:
        self._peer(other, another)
        lib().mp_vec_pointwise_min(
            self._ptr(), other._ptr(), another._ptr(),
            self._n(),
        )

    def pointwiseMax(self, other: "Vec", another: "Vec") -> None:
        self._peer(other, another)
        lib().mp_vec_pointwise_max(
            self._ptr(), other._ptr(), another._ptr(),
            self._n(),
        )

    def pointwiseMaxAbs(self, other: "Vec", another: "Vec") -> None:
        self._peer(other, another)
        lib().mp_vec_pointwise_max_abs(
            self._ptr(), other._ptr(), another._ptr(),
            self._n(),
        )

    def reciprocal(self) -> None:
        lib().mp_vec_reciprocal(self._ptr(), self._n())

    def exp(self) -> None:
        lib().mp_vec_exp(self._ptr(), self._n())

    def log(self) -> None:
        lib().mp_vec_log(self._ptr(), self._n())

    def sqrtabs(self) -> None:
        lib().mp_vec_sqrtabs(self._ptr(), self._n())

    def abs(self) -> None:
        lib().mp_vec_abs(self._ptr(), self._n())

    def conjugate(self) -> None:
        lib().mp_vec_conjugate(self._ptr(), self._n())

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
