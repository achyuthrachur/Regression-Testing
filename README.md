# Regression Replication Toolkit

This repository provides two utilities for preparing datasets and running ordinary least squares (OLS) regressions with Newey-West (HAC) standard errors.

- `app.py` - Streamlit web application for filtering datasets, exporting cleaned data, and running interactive HAC regressions with configurable dependent and independent variables, lag length selection, and downloadable diagnostics.
- `run_ols_hac.py` - Command-line helper that reproduces the expected HAC output and writes clipboard-friendly coefficient tables.

## Local setup

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Streamlit app

```bash
streamlit run app.py
```

Once running, upload a CSV or Excel file, choose the columns to retain, and download a cleaned dataset. The app also supports:

- Selecting the dependent variable and any number of independent variables.
- Setting the HAC lag length (defaults to the Newey-West rule of thumb).
- Including or excluding an intercept.
- Viewing diagnostics (R-squared, F-statistic, Durbin-Watson, Jarque-Bera, condition number, and more).
- Downloading coefficients, diagnostics, the full regression summary, and residuals/fitted values.

### OLS + HAC script

```bash
python run_ols_hac.py path\to\data.xlsx
```

Omit the `path` argument to open a file picker. If you need to load legacy `.xls` workbooks, install `xlrd==1.2.0`.

## Publish to GitHub

1. Initialize the repository and create the first commit.
2. Create a new GitHub repository (for example, `regression-replication`) and add it as a remote.
3. Push your commits to GitHub.

## Deploy on Streamlit Cloud

1. Push the repository to GitHub.
2. Sign in to [streamlit.io](https://streamlit.io/cloud) and select **New app**.
3. Choose the repository and branch, and set **Main file path** to `app.py`.
4. Streamlit Cloud will install packages from `requirements.txt` and deploy the app automatically.

Share the deployed URL with collaborators. New commits pushed to GitHub will trigger automatic redeploys.
