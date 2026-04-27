"""Sandboxed Python REPL for calculations."""
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL

_repl = PythonREPL()


@tool
def python_exec(code: str) -> str:
    """Execute Python code for calculations, data transformations, or analysis.

    Available: math, statistics, datetime, re, json, collections.
    NOT available: file I/O, network, os.system.

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
        result = _repl.run(code)
        return str(result)[:2000]  # Cap output
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"
