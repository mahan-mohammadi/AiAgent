

from pathlib import Path

import pandas as pd
from langchain_core.tools import tool


# Only allow files inside the project's data directory.
DATA_DIR = (Path(__file__).resolve().parent.parent / "data").resolve()
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@tool
def data_analysis(file_path: str) -> str:
    """Analyze a CSV or Excel file stored in the project's data directory.

    Provide the filename, such as 'sales.csv' or 'sales.xlsx'.
    Reports dataset dimensions, column types, missing values,
    duplicates, numerical statistics, and sample rows.
    """
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        requested_path = Path(file_path)

        # Accept filenames or paths relative to the data directory.
        if requested_path.is_absolute():
            return "Error: provide a filename inside the data directory."

        path = (DATA_DIR / requested_path).resolve()

        # Prevent access outside the data directory.
        if not path.is_relative_to(DATA_DIR):
            return "Error: files must be inside the project's data directory."

        if not path.is_file():
            return f"Error: file not found: {requested_path.name}"

        if path.stat().st_size > MAX_FILE_SIZE:
            return "Error: file exceeds the 20 MB limit."

        suffix = path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            return "Error: supported formats are .csv, .xlsx, and .xls."

        report = [
            f"FILE: {path.name}",
            f"Rows: {len(df)}",
            f"Columns: {len(df.columns)}",
            f"Duplicate rows: {df.duplicated().sum()}",
            "",
            "COLUMN TYPES",
            df.dtypes.to_string(),
            "",
            "MISSING VALUES",
            df.isnull().sum().to_string(),
            "",
            "NUMERICAL SUMMARY",
        ]

        numeric = df.select_dtypes(include="number")

        if numeric.empty:
            report.append("No numerical columns found.")
        else:
            report.append(
                numeric.describe().round(3).to_string()
            )

        report.extend([
            "",
            "SAMPLE ROWS",
            df.head(5).to_string(index=False),
        ])

        return "\n".join(report)[:10000]

    except Exception as exc:
        return f"Analysis failed: {type(exc).__name__}: {exc}"
