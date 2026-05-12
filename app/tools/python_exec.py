"""Sandboxed Python REPL for calculations.

AST-allowlist sandbox. Rejects imports outside a whitelist and any reference
to dangerous names (os/sys/subprocess/open/__import__/exec/eval/...). Runs in
a stripped builtins namespace so even if a reject is missed, capability surface
is small.
"""
from __future__ import annotations

import ast
import io
import math
import statistics
from contextlib import redirect_stdout
from datetime import date, datetime, time, timedelta, timezone

from langchain_core.tools import tool

_ALLOWED_MODULES = {
    "math",
    "statistics",
    "datetime",
    "re",
    "json",
    "collections",
    "itertools",
    "functools",
}

_FORBIDDEN_NAMES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "shutil",
    "pathlib",
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "input",
    "breakpoint",
    "exit",
    "quit",
    "help",
    "memoryview",
}

_FORBIDDEN_ATTRS = {
    "__class__",
    "__bases__",
    "__subclasses__",
    "__mro__",
    "__globals__",
    "__builtins__",
    "__import__",
    "__getattribute__",
    "__dict__",
    "__code__",
}

_SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
    "float", "format", "frozenset", "hash", "hex", "id", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min", "next", "oct",
    "ord", "chr", "pow", "print", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "type", "zip", "True", "False",
    "None",
}


def _validate(node: ast.AST) -> str | None:
    """Walk AST. Return error message if disallowed; None if safe."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Import):
            for alias in sub.names:
                root = alias.name.split(".")[0]
                if root not in _ALLOWED_MODULES:
                    return f"import of '{alias.name}' is not allowed"
        elif isinstance(sub, ast.ImportFrom):
            root = (sub.module or "").split(".")[0]
            if root not in _ALLOWED_MODULES:
                return f"import from '{sub.module}' is not allowed"
        elif isinstance(sub, ast.Attribute):
            if sub.attr in _FORBIDDEN_ATTRS or sub.attr.startswith("__") and sub.attr.endswith("__"):
                return f"attribute access '{sub.attr}' is not allowed"
        elif isinstance(sub, ast.Name):
            if sub.id in _FORBIDDEN_NAMES:
                return f"name '{sub.id}' is not allowed"
    return None


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in _ALLOWED_MODULES:
        raise ImportError(f"import of '{name}' is not allowed in sandbox")
    import importlib

    return importlib.import_module(name)


def _build_namespace() -> dict:
    raw = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
    safe_builtins = {k: raw[k] for k in _SAFE_BUILTINS if k in raw}
    safe_builtins["__import__"] = _safe_import
    return {
        "__builtins__": safe_builtins,
        "math": math,
        "statistics": statistics,
        "datetime": datetime,
        "date": date,
        "time": time,
        "timedelta": timedelta,
        "timezone": timezone,
    }


@tool
def python_exec(code: str) -> str:
    """Execute Python code for calculations, data transformations, or analysis.

    Available: math, statistics, datetime, re, json, collections, itertools, functools.
    NOT available: file I/O, network, subprocess, os, sys.

    Args:
        code: Python code. Use print() to see output.

    Returns:
        Stdout output from the code, or error message.

    Example:
        python_exec("
        revenues = [100, 120, 145]
        growth = [(revenues[i]-revenues[i-1])/revenues[i-1]*100 for i in range(1, len(revenues))]
        print(f'Growth rates: {growth}')
        ")
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return f"ERROR: SyntaxError: {e}"

    violation = _validate(tree)
    if violation:
        return f"ERROR: sandbox rejected code: {violation}"

    ns = _build_namespace()
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            exec(compile(tree, "<python_exec>", "exec"), ns)  # noqa: S102
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

    out = buf.getvalue().strip()
    if not out:
        return "(no stdout — use print() to see results)"
    return out[:2000]
