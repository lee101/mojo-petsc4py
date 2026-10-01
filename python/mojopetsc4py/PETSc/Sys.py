"""System-level information, mirroring `petsc4py/PETSc/Sys.pyx`."""

from __future__ import annotations

import platform
import sys

# The PETSc release whose Python-visible API and C numerics this port mirrors.
PETSC_VERSION = (3, 25, 5)
PETSC_VERSION_MAJOR = PETSC_VERSION[0]
PETSC_VERSION_MINOR = PETSC_VERSION[1]
PETSC_VERSION_SUBMINOR = PETSC_VERSION[2]
PETSC_VERSION_STRING = ".".join(str(v) for v in PETSC_VERSION)

# petscsys.h scalar kinds for the real-double, 64-bit-integer build.
Scalar = float
Real = float
Int = int
Bool = bool

__version__ = PETSC_VERSION_STRING


def getVersion() -> tuple[int, int, int]:
    """Return the PETSc version this port mirrors."""
    return PETSC_VERSION


def getVersionInfo() -> str:
    """Return a human-readable version string."""
    return (
        f"PETSc {PETSC_VERSION_STRING} interface (mojo-petsc4py) on"
        f" Python {sys.version.split()[0]} {platform.machine()}"
    )


def getMachine() -> str:
    return platform.machine()


def hasThreads() -> bool:
    return False
