"""Streamlit app for dataset filtering and regression analysis with HAC diagnostics."""

from __future__ import annotations

from io import BytesIO

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api.types import is_string_dtype
from statsmodels.stats.stattools import durbin_watson, jarque_bera
import altair as alt
import streamlit as st


def _strip_or_nan(value: object) -> object:
    """Return stripped string or NA for whitespace-only values."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return pd.NA
        return stripped
    return value


def replace_blank_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Replace empty or whitespace-only strings with NA."""
    cleaned = df.copy()
    for column in cleaned.columns:
        if is_string_dtype(cleaned[column]) or cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].apply(_strip_or_nan)
    return cleaned


def _format_value(value: object) -> str:
    """Pretty-format numerical diagnostics for display."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if pd.isna(value):
        return "NA"
    if isinstance(value, (np.integer, int)):
        return f"{int(value):,}"
    if isinstance(value, (np.floating, float)):
        magnitude = abs(value)
        if magnitude >= 1000:
            return f"{value:,.2f}"
        if magnitude >= 1:
            return f"{value:,.4f}"
        return f"{value:.6f}"
    return str(value)


def _flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert any MultiIndex columns to single-level strings."""
    flattened = frame.copy()
    if isinstance(flattened.columns, pd.MultiIndex):
        flattened.columns = [
            " ".join(part for part in map(str, col) if part.strip()) or f"col_{i}"
            for i, col in enumerate(flattened.columns)
        ]
    else:
        flattened.columns = [
            str(col).strip() or f"col_{i}" for i, col in enumerate(flattened.columns)
        ]
    return flattened


def ols_with_hac(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    lags: int,
    include_intercept: bool,
) -> tuple[sm.regression.linear_model.RegressionResultsWrapper, ...]:
    """Run OLS with HAC (Newey-West) covariance and return core outputs."""
    numeric = (
        df[[y_col] + x_cols]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
    )
    if numeric.empty:
        raise ValueError("No rows with numeric data remain after numeric conversion. Adjust your selections.")

    y = numeric[y_col]
    X = numeric[x_cols]
    if include_intercept:
        X = sm.add_constant(X)

    ols = sm.OLS(y, X).fit()
    hac = ols.get_robustcov_results(
        cov_type="HAC",
        maxlags=int(lags),
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
    return ols, hac, table, numeric


def build_diagnostics(
    ols_res: sm.regression.linear_model.RegressionResultsWrapper,
    hac_res: sm.regression.linear_model.RegressionResultsWrapper,
    hac_lag: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create raw and formatted regression diagnostic tables."""
    resid = ols_res.resid
    jb_stat, jb_pvalue, skew, kurtosis = jarque_bera(resid)

    diagnostics = [
        ("Observations", hac_res.nobs),
        ("Df Residuals", hac_res.df_resid),
        ("Df Model", hac_res.df_model),
        ("R-squared", hac_res.rsquared),
        ("Adj. R-squared", hac_res.rsquared_adj),
        ("F-statistic", hac_res.fvalue),
        ("Prob(F-statistic)", hac_res.f_pvalue),
        ("Log-likelihood", ols_res.llf),
        ("AIC", ols_res.aic),
        ("BIC", ols_res.bic),
        ("Root MSE", np.sqrt(ols_res.scale)),
        ("Durbin-Watson", durbin_watson(resid)),
        ("Jarque-Bera", jb_stat),
        ("Prob(JB)", jb_pvalue),
        ("Skew", skew),
        ("Kurtosis", kurtosis),
        ("Condition number", getattr(ols_res, "condition_number", np.nan)),
        ("Selected HAC lag", hac_lag),
    ]

    diag_df = pd.DataFrame(diagnostics, columns=["Metric", "Value"])
    formatted = diag_df.assign(Value=diag_df["Value"].map(_format_value))
    return diag_df, formatted


def main() -> None:
    st.set_page_config(page_title="Regression Replication Toolkit", page_icon=":bar_chart:", layout="wide")
    st.title("Regression Replication Toolkit")
    st.write(
        "Upload a CSV or Excel file, pick the columns you need, and download a cleaned dataset. "
        "You can then run OLS with HAC (Newey-West) standard errors, choose the dependent and "
        "independent variables, set the lag length, and review detailed diagnostics."
    )

    uploaded = st.file_uploader(
        "Upload a CSV or Excel file (drag and drop supported)",
        type=["csv", "xlsx", "xls", "xlsm"],
    )

    if uploaded is None:
        st.info("Upload a CSV or Excel file to continue.")
        return

    name = (uploaded.name or "").lower()

    try:
        if name.endswith(".csv"):
            uploaded.seek(0)
            df = pd.read_csv(uploaded)
        else:
            uploaded.seek(0)
            workbook = pd.ExcelFile(uploaded)
            sheet_names = workbook.sheet_names
            if not sheet_names:
                raise ValueError("No worksheets found in the Excel file.")
            if len(sheet_names) > 1:
                sheet_name = st.selectbox("Select a worksheet", options=sheet_names, index=0)
            else:
                sheet_name = sheet_names[0]
            df = workbook.parse(sheet_name)
    except Exception as exc:
        st.error(f"Failed to read file: {exc}")
        st.stop()

    columns = df.columns.tolist()
    if not columns:
        st.error("No columns detected in the uploaded file.")
        st.stop()

    st.subheader("Detected columns")
    st.code("\n".join(map(str, columns)))

    selected_cols = st.multiselect(
        "Columns to keep",
        options=columns,
        default=columns,
        help="Rows missing any of the selected columns will be dropped.",
    )

    if not selected_cols:
        st.info("Select at least one column to continue.")
        st.stop()

    subset = df[selected_cols]

    st.caption("Preview of selected columns (first 20 rows)")
    st.dataframe(subset.head(20), use_container_width=True)

    with st.expander("Missing-value handling"):
        treat_blank = st.checkbox(
            "Treat empty strings / whitespace as missing values before filtering",
            value=True,
            help="If enabled, values like '' or '   ' are interpreted as missing.",
        )

    working = subset.copy()
    if treat_blank:
        working = replace_blank_with_nan(working)

    clean = working.dropna(how="any")

    original_rows = len(subset)
    clean_rows = len(clean)
    removed_rows = original_rows - clean_rows

    metrics = st.columns(3)
    metrics[0].metric("Original rows", original_rows)
    metrics[1].metric("Rows kept", clean_rows)
    metrics[2].metric("Rows removed", removed_rows, delta=-removed_rows)

    reset_index = st.checkbox("Reset row index in output", value=True)
    if reset_index:
        clean_data = clean.reset_index(drop=True)
    else:
        clean_data = clean

    if clean_data.empty:
        st.warning("The filtered dataset is empty. Adjust your selections to continue.")
        st.stop()

    st.subheader("Cleaned dataset preview")
    st.dataframe(clean_data.head(20), use_container_width=True)

    csv_bytes = clean_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered CSV",
        data=csv_bytes,
        file_name="filtered.csv",
        mime="text/csv",
    )

    excel_bytes = BytesIO()
    with pd.ExcelWriter(excel_bytes, engine="xlsxwriter") as writer:
        clean_data.to_excel(writer, index=False, sheet_name="Filtered")
        summary = pd.DataFrame(
            {
                "Metric": ["Original rows", "Rows kept", "Rows removed"],
                "Value": [original_rows, clean_rows, removed_rows],
            }
        )
        summary.to_excel(writer, index=False, sheet_name="Summary")
    excel_bytes.seek(0)

    st.download_button(
        label="Download filtered Excel",
        data=excel_bytes,
        file_name="filtered.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.divider()
    st.header("Regression analysis")

    if len(clean_data.columns) < 2:
        st.info("Add at least two columns to run a regression (one for Y, one or more for X).")
        return

    all_cols = clean_data.columns.tolist()

    with st.form("regression_form"):
        y_col = st.selectbox("Dependent variable (Y)", options=all_cols, key="reg_y")
        x_candidates = [col for col in all_cols if col != y_col]
        x_cols = st.multiselect(
            "Independent variable(s) (X)",
            options=x_candidates,
            default=x_candidates,
            key="reg_x",
        )

        n_obs = len(clean_data)
        default_lag = int(np.floor(4 * (n_obs / 100) ** (2 / 9)))
        hac_lag = st.number_input(
            "HAC lag length",
            min_value=0,
            value=max(default_lag, 0),
            step=1,
            help="Newey-West lag rule of thumb is 4 * (n / 100)^(2/9). Set to 0 for no autocorrelation adjustment.",
        )

        include_intercept = st.checkbox(
            "Include intercept",
            value=True,
            help="Adds a constant column to the regression design matrix.",
        )

        submitted = st.form_submit_button("Run regression", type="primary")

    if not submitted:
        return

    if not x_cols:
        st.error("Select at least one independent variable.")
        return

    try:
        ols_res, hac_res, coef_table, numeric_data = ols_with_hac(
            clean_data,
            y_col,
            x_cols,
            lags=int(hac_lag),
            include_intercept=include_intercept,
        )
    except ValueError as err:
        st.error(str(err))
        return
    except Exception as exc:
        st.error(f"Regression failed: {exc}")
        return

    st.success("Regression completed.")
    st.caption(f"Regression fit on {len(numeric_data)} rows after numeric conversion.")

    diag_raw, diag_display = build_diagnostics(ols_res, hac_res, int(hac_lag))
    st.subheader("Diagnostics")
    st.dataframe(diag_display, use_container_width=False)

    coef_display = coef_table.reset_index().rename(columns={"index": "variable"})
    st.subheader("Coefficient estimates")
    st.dataframe(coef_display, use_container_width=True)

    summary = hac_res.summary2()
    if summary.tables:
        overview = _flatten_columns(summary.tables[0]).reset_index()
        overview = overview.rename(columns={"index": "Metric"})
        st.subheader("Model overview")
        st.table(overview)

        if len(summary.tables) > 2:
            residual_table = _flatten_columns(summary.tables[2]).reset_index()
            residual_table = residual_table.rename(columns={"index": "Statistic"})
            st.subheader("Residual summary")
            st.table(residual_table)

    summary_text = hac_res.summary().as_text()
    with st.expander("Full statsmodels summary"):
        st.code(summary_text)

    predicted_col = f"predicted_{y_col}"
    augmented = clean_data.copy()
    augmented[predicted_col] = np.nan
    augmented.loc[numeric_data.index, predicted_col] = hac_res.fittedvalues

    residuals = pd.DataFrame(
        {
            "observation": numeric_data.index,
            f"actual_{y_col}": numeric_data[y_col],
            "predicted": hac_res.fittedvalues,
            "residual": hac_res.resid,
        }
    )

    plot_source = residuals.rename(columns={"observation": "Observation"})
    plot_source = plot_source.reset_index(drop=True)
    plot_df = plot_source.melt(id_vars="Observation", value_vars=[f"actual_{y_col}", "predicted"], var_name="Series", value_name="Value")

    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Observation:N", title="Observation"),
            y=alt.Y("Value:Q", title=y_col),
            color=alt.Color("Series:N", title="Series"),
            tooltip=["Observation", "Series", "Value"],
        )
        .properties(height=300)
    )

    st.subheader("Actual vs. predicted")
    st.altair_chart(chart, use_container_width=True)

    st.download_button(
        label="Download cleaned data + predictions (CSV)",
        data=augmented.to_csv(index=False).encode("utf-8"),
        file_name="filtered_with_predictions.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download diagnostics (CSV)",
        data=diag_raw.to_csv(index=False).encode("utf-8"),
        file_name="regression_diagnostics.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download coefficients (CSV)",
        data=coef_display.to_csv(index=False).encode("utf-8"),
        file_name="regression_coefficients.csv",
        mime="text/csv",
    )
    st.download_button(
        label="Download regression summary (TXT)",
        data=summary_text.encode("utf-8"),
        file_name="regression_summary.txt",
        mime="text/plain",
    )
    st.download_button(
        label="Download fitted values and residuals (CSV)",
        data=residuals.to_csv(index=False).encode("utf-8"),
        file_name="regression_residuals.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
