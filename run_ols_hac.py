#!/usr/bin/env python
"""
run_ols_hac.py – OLS with HAC (Newey–West) and t-tests
Matches the "white" output by:
  • Bartlett kernel (default)
  • user-specified max lags
  • small-sample correction OFF (use_correction=False)
  • t-statistics (df = residual dof)

ADDITION:
  • Prints a TSV (tab-separated) 2-column table: variable, coef
    - Easy to copy/paste into Excel
    - Also attempts to copy to clipboard automatically
"""

# ── imports ────────────────────────────────────────────────────────────
import sys
import pathlib
import tkinter as tk
from tkinter import filedialog as fd

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ── pick file (CLI arg or file-open dialog) ─────────────────────────────
def get_filepath() -> pathlib.Path:
    if len(sys.argv) > 1:  # path given at the CLI
        return pathlib.Path(sys.argv[1])

    root = tk.Tk()
    root.withdraw()
    fname = fd.askopenfilename(
        title="Choose CSV or Excel file",
        filetypes=[
            ("Excel / CSV", "*.csv *.xls *.xlsx"),
            ("CSV files", "*.csv"),
            ("Excel files", "*.xls;*.xlsx"),
            ("All files", "*.*"),
        ],
    )
    if not fname:
        sys.exit("No file selected – exiting.")
    return pathlib.Path(fname)


# ── load CSV or workbook; prompt for sheet if needed ────────────────────
def load_table(path: pathlib.Path) -> pd.DataFrame:
    suf = path.suffix.lower()

    if suf in {".csv", ".txt"}:
        return pd.read_csv(path)

    if suf in {".xls", ".xlsx"}:
        sheets = pd.read_excel(path, sheet_name=None)
        if len(sheets) == 1:
            return next(iter(sheets.values()))

        print("\nSheets in workbook")
        for i, nm in enumerate(sheets, 1):
            print(f"{i:2}) {nm}")
        while True:
            try:
                idx = int(input("Pick a sheet (#): ")) - 1
                return sheets[list(sheets)[idx]]
            except (ValueError, IndexError):
                print("  ❌  enter a valid number.")

    raise ValueError("Unsupported file type (use CSV / xls / xlsx).")


# ── choose dependent & independent variables ────────────────────────────
def choose_vars(df: pd.DataFrame):
    cols = list(df.columns)
    print("\nAvailable columns")
    for i, c in enumerate(cols, 1):
        print(f"{i:2}) {c}")

    while True:  # pick Y
        try:
            y_idx = int(input("\nPick DEPENDENT (Y) (#): ")) - 1
            y = cols[y_idx]
            break
        except (ValueError, IndexError):
            print("  ❌  try again.")

    while True:  # pick X’s
        try:
            x_idx = input("Pick INDEPENDENT X's (#,#,…): ")
            x_idx = [int(i) - 1 for i in x_idx.split(",")]
            x = [cols[i] for i in x_idx if i != y_idx]
            if not x:
                raise ValueError
            break
        except (ValueError, IndexError):
            print("  ❌  need at least one valid number.")
    return y, x


# ── OLS + HAC (matches your expected output) ────────────────────────────
def ols_with_hac(df: pd.DataFrame, y_col: str, x_cols: list[str], lags: int):
    y = df[y_col].astype(float)
    X = df[x_cols].astype(float)
    X = sm.add_constant(X)  # adds 'const', keeps labels

    ols = sm.OLS(y, X).fit()

    # HAC with small-sample correction OFF and t-based inference
    hac = ols.get_robustcov_results(
        cov_type="HAC",
        maxlags=lags,
        use_correction=False,  # match the white output
        use_t=True,
    )

    table = pd.DataFrame(
        {
            "coef": hac.params,
            "std_err_OLS": ols.bse,
            "std_err_HAC": hac.bse,
            "t_HAC": hac.tvalues,
            "p_HAC": hac.pvalues,
        },
        index=hac.model.exog_names,
    )

    return ols, hac, table


# ── main driver ─────────────────────────────────────────────────────────
def main():
    path = get_filepath()
    df = load_table(path)
    y, X = choose_vars(df)

    n = len(df)
    default_lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    try:
        lag_in = input(f"HAC lag [default {default_lag}]: ").strip()
        lags = int(lag_in) if lag_in else default_lag
    except ValueError:
        lags = default_lag

    ols, hac, tbl = ols_with_hac(df, y, X, lags)

    # This summary block should look like your expected (white) output
    print(hac.summary())

    # And keep your detailed coefficient table as well
    print("\n--------------- Coefficients (detailed) ---------------\n")
    print(tbl.to_string(float_format=lambda v: f"{v:10.4f}"))

    # NEW: Coefficients-only TSV for Excel copy/paste
    coef_tbl = tbl[["coef"]].reset_index().rename(columns={"index": "variable", "coef": "coef"})
    tsv_block = coef_tbl.to_csv(sep="\t", index=False)

    print("\n--------------- Coefficients (TSV for Excel) ---------------")
    print("Paste this into Excel (tab-separated):\n")
    print(tsv_block, end="")  # already has trailing newline

    # Try copying to clipboard (Windows/macOS)
    try:
        coef_tbl.to_clipboard(index=False, excel=True)
        print("✅ Coefficients table copied to clipboard. Open Excel and paste.")
    except Exception as exc:  # clipboard backend not available
        print("ⓘ Could not copy to clipboard automatically. You can still copy the TSV above.")
        print(f"   Details: {exc}")

    print()


# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
