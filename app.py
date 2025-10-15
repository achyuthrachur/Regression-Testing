"""Streamlit app for filtering datasets to rows with complete data across selected columns."""

from io import BytesIO

import pandas as pd
import streamlit as st


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
