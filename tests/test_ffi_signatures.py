"""The declared ctypes signatures must match the Mojo exports exactly.

Every `@export("C")` wrapper takes its buffers as `Int` addresses, so a
signature that is one argument short in `_lib.py` does not raise: ctypes reads
the missing argument from a register nobody set, and the kernel dereferences
whatever it finds.  This test parses both sides and compares them, so that class
of mistake cannot come back unnoticed.
"""

import ast
import re
from pathlib import Path

import pytest

from mojopetsc4py._lib import F, I, _SIGNATURES

SRC = Path(__file__).resolve().parent.parent / "src" / "capi.mojo"
MOJO = re.compile(
    r'@export\("([a-z_0-9]+)"\)\s*\ndef\s+\w+\((.*?)\)\s*abi\("C"\)'
    r"(\s*->\s*Int)?",
    re.S,
)


def mojo_signatures():
    """name -> (nargs, n_int, n_float, returns_int) for every @export."""
    out = {}
    for name, params, ret in MOJO.findall(SRC.read_text()):
        parts = [p.strip() for p in params.replace("\n", " ").split(",") if p.strip()]
        out[name] = (
            len(parts),
            sum(1 for p in parts if ": Int" in p),
            sum(1 for p in parts if ": Float64" in p),
            ret.strip() != "",
        )
    return out


def python_signatures():
    """name -> the same tuple, built from the table `_lib.py` actually uses."""
    out = {}
    for name, (argtypes, restype) in _SIGNATURES.items():
        out[name] = (
            len(argtypes),
            sum(1 for a in argtypes if a is I),
            sum(1 for a in argtypes if a is F),
            restype is not None,
        )
    return out


MOJO_SIGS = mojo_signatures()
PY_SIGS = python_signatures()


def test_capi_has_exports():
    assert MOJO_SIGS, "no @export found in src/capi.mojo"


@pytest.mark.parametrize("name", sorted(MOJO_SIGS))
def test_every_export_is_declared(name):
    assert name in PY_SIGS, f"{name} is exported but not declared in _lib.py"


@pytest.mark.parametrize("name", sorted(PY_SIGS))
def test_every_declaration_is_exported(name):
    assert name in MOJO_SIGS, f"{name} is declared but not exported"


@pytest.mark.parametrize("name", sorted(MOJO_SIGS))
def test_signature_agrees(name):
    assert PY_SIGS.get(name) == MOJO_SIGS[name]


def test_no_short_signature_declares_a_too_small_argtypes():
    """The specific failure this file exists to catch, spelled out."""
    # mp_pc_icc_numeric takes 13 addresses; a 12-entry argtypes list leaves the
    # 13th one to whatever is in the register.
    assert len(_SIGNATURES["mp_pc_icc_numeric"][0]) == 13
    assert MOJO_SIGS["mp_pc_icc_numeric"][0] == 13


def test_table_is_valid_python_with_the_expected_shape():
    """`_lib.py` is parsed with `ast`, so this also proves it still compiles."""
    tree = ast.parse((SRC.parent.parent / "python/mojopetsc4py/_lib.py").read_text())
    found = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_SIGNATURES":
                    found = node.value
    assert found is not None and isinstance(found, ast.Dict)
    assert len(found.keys) == len(_SIGNATURES)
    for key in found.keys:
        assert isinstance(key, ast.Constant) and isinstance(key.value, str)
        assert key.value.startswith("mp_")