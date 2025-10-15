"""Streamlit app for filtering datasets to rows with complete data across selected columns."""

from io import BytesIO

import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st


def ols_with_hac(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    lags: int,
    include_intercept: bool,
):
    """Run OLS with HAC (Newey-West) covariance and return results + summary table."""
    numeric = (
        df[[y_col] + x_cols]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    if numeric.empty:
        raise ValueError("No rows with numeric data remain after conversion. Check your selections.")

    y = numeric[y_col]
    X = numeric[x_cols]
    if include_intercept:
        X = sm.add_constant(X)

    ols = sm.OLS(y, X).fit()
    hac = ols.get_robustcov_results(
        cov_type="HAC",
        maxlags=lags,
        use_correction=False,
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


st.set_page_config(page_title="Dataset Column Filter", page_icon="🧹", layout="wide")
st.title("🧹 Dataset Column Filter for Regression")
st.write(
    "Upload a CSV or Excel file, choose the columns you care about, and I'll keep only rows "
    "where **all** those columns are present (no missing values). Then download the cleaned file."
)

# --- File upload ---
uploaded = st.file_uploader(
    "Upload a CSV or Excel file (drag & drop supported)",
    type=["csv", "xlsx", "xls", "xlsm"],
)

# ✅ Stop immediately if no file has been provided yet
if uploaded is None:
    st.info("Upload a CSV or Excel file to continue.")
    st.stop()

# Safe to reference uploaded now
name = (uploaded.name or "").lower()
read_error = None
sheet_name = None

# --- Load file into a DataFrame ---
try:
    if name.endswith(".csv"):
        # Read CSV using default pandas settings
        df = pd.read_csv(uploaded)
    else:
        # Excel handling — allow sheet selection
        try:
            xls = pd.ExcelFile(uploaded)
        except Exception as exc:
            # Surface a helpful hint for legacy .xls
            raise RuntimeError(
                "Failed to open Excel file. If it's a legacy .xls, install xlrd==1.2.0.\n" + str(exc)
            ) from exc
        if len(xls.sheet_names) > 1:
            sheet_name = st.selectbox("Select a worksheet", options=xls.sheet_names, index=0)
        else:
            sheet_name = xls.sheet_names[0]
        df = xls.parse(sheet_name)
except Exception as exc:
    read_error = str(exc)

if read_error:
    st.error(read_error)
    st.stop()

# --- Show headers and let user choose subset ---
st.subheader("Detected columns")
st.code("\n".join(map(str, df.columns.tolist())) or "<no columns>")

selected_cols = st.multiselect(
    "Which columns do you want to extract?",
    options=df.columns.tolist(),
)

if not selected_cols:
    st.info("Select at least one column to continue.")
    st.stop()

subset = df[selected_cols]

st.caption("Preview of your selected columns (first 20 rows)")
st.dataframe(subset.head(20), use_container_width=True)

# --- Treat blanks/whitespace as missing? ---
with st.expander("Missing-value behavior (advanced)"):
    treat_blank = st.checkbox(
        "Also treat empty strings / whitespace as missing values",
        value=True,
        help="If checked, blank strings like '' or '   ' are treated as NA before dropping rows.",
    )

work = subset.copy()
if treat_blank:
    # Replace blank-only strings with NA so dropna removes them.
    work = work.replace(r"^\s*$", pd.NA, regex=True)

# Keep only rows where ALL selected columns are present (drop if any are missing)
clean = work.dropna(how="any")

# --- Stats ---
orig_rows = len(subset)
clean_rows = len(clean)
removed = orig_rows - clean_rows

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Original rows", orig_rows)
with c2:
    st.metric("Rows kept", clean_rows)
with c3:
    st.metric("Rows removed", removed, delta=-removed)

reset_index = st.checkbox("Reset row index in output", value=True)
if reset_index:
    clean = clean.reset_index(drop=True)

st.subheader("Cleaned dataset preview")
st.dataframe(clean.head(20), use_container_width=True)

# --- Downloads ---
# CSV
csv_bytes = clean.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv_bytes,
    file_name="filtered.csv",
    mime="text/csv",
)

# Excel with a small Summary sheet
output = BytesIO()
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    clean.to_excel(writer, index=False, sheet_name="Filtered")
    summary = pd.DataFrame(
        {
            "Metric": ["Original rows", "Rows kept", "Rows removed"],
            "Value": [orig_rows, clean_rows, removed],
        }
    )
    summary.to_excel(writer, index=False, sheet_name="Summary")
output.seek(0)

st.download_button(
    label="Download Excel",
    data=output,
    file_name="filtered.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.success("Done! Use the download buttons above to save your cleaned data.")

# --- Optional regression with HAC ---
st.markdown("### Optional: Run OLS with HAC (Newey–West) Standard Errors")
if len(clean.columns) < 2:
    st.info("Select at least two columns above to enable regression (one for Y, one or more for X).")
else:
    with st.expander("Configure regression"):
        all_cols = clean.columns.tolist()
        y_col = st.selectbox("Dependent variable (Y)", options=all_cols)
        default_x = [c for c in all_cols if c != y_col]
        x_cols = st.multiselect(
            "Independent variable(s) (X)",
            options=[c for c in all_cols if c != y_col],
            default=default_x,
        )

        n_obs = len(clean)
        default_lag = int(np.floor(4 * (n_obs / 100) ** (2 / 9)))
        lags = st.number_input(
            "HAC lag length",
            min_value=0,
            value=default_lag,
            step=1,
            help="Default follows Newey-West rule of thumb. Set to 0 for no autocorrelation adjustment.",
        )

        include_intercept = st.checkbox(
            "Include intercept",
            value=True,
            help="Adds a constant column to the regression design matrix.",
        )

        run_regression = st.button("Run regression", type="primary")

    if run_regression:
        if not x_cols:
            st.error("Select at least one independent variable.")
        else:
            try:
                _, hac_res, coef_tbl = ols_with_hac(
                    clean,
                    y_col,
                    x_cols,
                    lags,
                    include_intercept=include_intercept,
                )

                used_rows = len(hac_res.model.endog)
                st.caption(f"Regression fit on {used_rows} rows after numeric conversion.")

                st.subheader("Regression summary (HAC)")
                summary_text = hac_res.summary().as_text()
                st.text(summary_text)

                coef_display = coef_tbl.reset_index().rename(columns={"index": "variable"})
                st.subheader("Coefficient details")
                st.dataframe(coef_display, use_container_width=True)

                coef_csv = coef_display.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download coefficients (CSV)",
                    data=coef_csv,
                    file_name="coefficients.csv",
                    mime="text/csv",
                )
                st.download_button(
                    label="Download regression summary (TXT)",
                    data=summary_text.encode("utf-8"),
                    file_name="regression_summary.txt",
                    mime="text/plain",
                )
            except ValueError as err:
                st.error(str(err))
            except Exception as exc:
                st.error(f"Regression failed: {exc}")
