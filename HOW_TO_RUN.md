# How to run (local, Streamlit Cloud, Render)

This document lists **prerequisites**, **dependencies**, how to run the **Streamlit dashboard** on your machine, and how it maps to **Streamlit Community Cloud** and **Render**.

---

## Prerequisites

| Requirement | Notes |
|-------------|--------|
| **Python** | **3.9+** recommended (3.10 or 3.11 work well with Streamlit and pandas). Check with `python --version`. |
| **pip** | Usually bundled with Python. Upgrade if needed: `python -m pip install --upgrade pip`. |
| **Git** | Needed to clone the repo and to connect GitHub to Streamlit Cloud / Render. |
| **GitHub account** | Required to deploy on [Streamlit Community Cloud](https://share.streamlit.io) or [Render](https://render.com) from a repository. |
| **Browser** | For local app: usually opens automatically; use Chrome/Edge/Firefox for best compatibility. |

**Data files (required for the dashboard):** the app reads CSVs under `data/`. Ensure these exist after clone (e.g. `data/forecasts.csv`, `data/cluster_forecasts.csv`, `data/total_forecasts.csv`, `data/shap_summary.csv`). If anything is missing, regenerate or copy from your team’s source.

---

## Dependencies (`requirements.txt`)

All packages are listed in `requirements.txt`. Install them together; versions are not pinned in the repo (pip resolves compatible releases at install time).

| Package | Role in this project |
|---------|----------------------|
| **streamlit** | Web UI framework; entry point is `dashboard.py`. |
| **pandas** | Loads and filters forecast / SHAP CSV data. |
| **numpy** | Numerical support used with pandas and plots. |
| **plotly** | Interactive charts in the dashboard. |

**Install command:**

```bash
pip install -r requirements.txt
```

Optional but recommended: use a virtual environment before installing (see [Local run](#local-run)).

---

## Local run

1. **Clone the repository** (if you have not already):

   ```bash
   git clone <your-repo-url>
   cd kredl_forecast
   ```

2. **Create and activate a virtual environment** (recommended):

   - Windows (PowerShell):

     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```

   - macOS / Linux:

     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Start the Streamlit app** from the project root (same folder as `dashboard.py`):

   ```bash
   streamlit run dashboard.py
   ```

5. **Open the app:** Streamlit prints a local URL (typically `http://localhost:8501`). Open it in your browser.

To stop the server, press `Ctrl+C` in the terminal.

---

## Streamlit Community Cloud (primary hosted option)

**Console:** [https://share.streamlit.io](https://share.streamlit.io)

**Typical deploy settings:**

- **Repository:** your GitHub repo containing this project.
- **Branch:** e.g. `main`.
- **Main file path:** `dashboard.py`.

After deployment, Streamlit shows your **public app URL** in the app dashboard (shape like `https://<subdomain>.streamlit.app`).

**Fill in your live link after deploy:**

| Item | Value |
|------|--------|
| **Streamlit app URL** | _Paste your URL here after first deploy_ |

---

## Render (fallback hosted option)

**Console:** [https://render.com](https://render.com)

This repo includes `render.yaml` for a **Blueprint** deploy:

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0`

**Manual Web Service** (if not using Blueprint): use the same build and start commands; set the service type to a **Web Service** and connect the same GitHub repo.

After the service is **Live**, Render shows a URL like `https://<service-name>.onrender.com`.

**Fill in your live link after deploy:**

| Item | Value |
|------|--------|
| **Render app URL** | _Paste your URL here after first deploy_ |

**Note:** On Render’s **free** tier, the service may **spin down** when idle. The first visit after idle can take longer while the instance wakes up.

---

## Quick reference

| Where | Command or link |
|-------|------------------|
| **Local** | `streamlit run dashboard.py` → open printed localhost URL |
| **Streamlit Cloud** | [share.streamlit.io](https://share.streamlit.io) → deploy repo → use generated `.streamlit.app` URL |
| **Render** | [render.com](https://render.com) → Blueprint from `render.yaml` or manual web service → use `onrender.com` URL |

---

## Optional: training / pipeline scripts

This repo also includes Python scripts for model training and data prep (e.g. `train_model.py`, `fetch_weather.py`, `aggregate.py`). They are **not** required to **view** the dashboard if `data/` CSVs are already present. Those scripts may need **additional** packages beyond `requirements.txt`; only install what each script’s imports need if you run them locally.
