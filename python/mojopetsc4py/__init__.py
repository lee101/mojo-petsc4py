"""Krylov solvers and preconditioners from petsc4py, running in Mojo.

The API mirrors upstream petsc4py for the covered subset, so
`from mojopetsc4py import PETSc` and `from petsc4py import PETSc` are
interchangeable for it.
"""

from . import PETSc  # noqa: F401

__version__ = "0.1.0"
__upstream__ = "petsc4py 3.25.5"

__all__ = ["PETSc"]
