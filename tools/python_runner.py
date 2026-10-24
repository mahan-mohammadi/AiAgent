

import subprocess
import sys
import tempfile
from pathlib import Path

from langchain_core.tools import tool


@tool
def python_runner(code: str) -> str:
    """Execute a short Python script and return its output.

    Use for data processing, calculations, and testing small snippets
    of Python code. The code must print results to produce visible output.
    """
    if not code.strip():
        return "Execution failed: code cannot be empty."

    # Limit script size to avoid accidental excessive input.
    if len(code) > 10_000:
        return "Execution failed: code exceeds the 10,000-character limit."

    try:
        with tempfile.TemporaryDirectory(prefix="langgraph_runner_") as temp_dir:
            script_path = Path(temp_dir) / "script.py"
            script_path.write_text(code, encoding="utf-8")

            result = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            output = []
            if stdout:
                output.append(f"Output:\n{stdout}")
            if stderr:
                output.append(f"Errors:\n{stderr}")
            if result.returncode != 0:
                output.append(f"Exit code: {result.returncode}")
            if not output:
                output.append("Script completed with no output.")

            response = "\n\n".join(output)
            return response[:8_000]

    except subprocess.TimeoutExpired:
        return "Execution failed: script exceeded the 5-second time limit."
    except Exception as exc:
        return f"Execution failed: {type(exc).__name__}: {exc}"
