"""The PETSc-compatible namespace, mirroring `petsc4py.PETSc`.

The re-export order follows the `include "..."` order of upstream's
`PETSc.pyx`: `Vec`, `Mat`, then `PC`, then `KSP`.
"""

from .Vec import Vec, VecType, NormType  # noqa: F401
from .Mat import (  # noqa: F401
    Mat, MatType, MatInfoType, MatOrderingType, MatSORType, from_dense,
)
from .PC import (  # noqa: F401
    PC, PCType, Side, FailedReason, PC_JACOBI_DIAGONAL, PC_JACOBI_ROWMAX,
    PC_JACOBI_ROWL1, PC_JACOBI_ROWSUM,
)
from .KSP import (  # noqa: F401
    KSP, KSPType, ConvergedReason, GMRESCGSRefinementType,
    GMRESOrthogonalizationType,
)
from .Sys import (  # noqa: F401
    getVersion, getVersionInfo, getMachine, Scalar, Real, Int, Bool,
)

# Upstream spells these under both their bare and their prefixed names.
PCSide = Side
PCFailedReason = FailedReason
KSPConvergedReason = ConvergedReason
KSPNormType = KSP.NormType

__all__ = [
    "Vec", "Mat", "PC", "KSP",
    "VecType", "NormType", "MatType", "MatInfoType", "MatOrderingType",
    "MatSORType", "PCType", "PCSide", "PCFailedReason", "KSPType",
    "PC_JACOBI_DIAGONAL", "PC_JACOBI_ROWMAX", "PC_JACOBI_ROWL1",
    "PC_JACOBI_ROWSUM",
    "KSPConvergedReason", "KSPNormType", "GMRESCGSRefinementType",
    "GMRESOrthogonalizationType", "Scalar", "Real", "Int", "Bool",
    "getVersion", "getVersionInfo", "getMachine", "from_dense",
]
