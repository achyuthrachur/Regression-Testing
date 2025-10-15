# Dataset Column Filter & HAC Regression Utilities

This repository contains two data-preparation tools:

- `app.py` – a Streamlit web app that filters a dataset to rows with complete data across user-selected columns and provides CSV/Excel downloads.
- `run_ols_hac.py` – a command-line helper that runs OLS with Newey–West (HAC) standard errors to reproduce the "white" output, complete with clipboard-friendly coefficient tables.

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

### OLS + HAC script

```bash
python run_ols_hac.py path\to\data.xlsx
```

Omit the `path` argument to trigger a file picker. If you need to open legacy `.xls` workbooks, install `xlrd==1.2.0`.

## Publish to GitHub

1. Initialize the repo and create the first commit:

   ```bash
   git init
   git add .
   git commit -m "Initial commit: Streamlit column filter and HAC utility"
   ```

2. Create a new GitHub repository (e.g., `dataset-column-filter`) and add it as a remote:

   ```bash
   git remote add origin git@github.com:<your-user>/dataset-column-filter.git
   git push -u origin main
   ```

   Replace `git@github.com:...` with your preferred HTTPS or SSH remote URL.

## Deploy on Streamlit Cloud

1. Push the repository to GitHub (as described above).
2. Sign in to [streamlit.io](https://streamlit.io/cloud) and select **New app**.
3. Choose the repository, branch (usually `main`), and set **Main file path** to `app.py`.
4. Confirm; Streamlit Cloud will install packages from `requirements.txt` and deploy the app automatically.

You can now share the Streamlit Cloud URL with collaborators. When you push new commits to GitHub, the deployment will update automatically.
