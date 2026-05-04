# Ai-for-Bharat - Karnataka Renewable Forecast Dashboard

This project is a Streamlit dashboard for AI-generated electricity forecasts for Karnataka solar and wind generation.

## What this app does

- Shows P50 forecast, P10-P90 uncertainty band, actual generation, and naive baseline.
- Supports Plant / Cluster / Karnataka Total views.
- Explains key forecast drivers and operational recommendations.

## Repository structure

- `dashboard.py` — Streamlit app entry point
- `data/` — forecast and SHAP CSV files used by the app
- `requirements.txt` — Python dependencies for deployment
- `render.yaml` — Render deployment config (optional hosted path)
- `HOW_TO_RUN.md` — prerequisites, dependency list, local run, Streamlit Cloud, and Render details

## Local run

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run dashboard.py`

For a step-by-step guide (versions, data files, hosted URLs), see [HOW_TO_RUN.md](HOW_TO_RUN.md).

---

## Deploy on Streamlit Community Cloud (primary)

### GitHub repository

Ensure the repo includes at least:

- `dashboard.py`
- `requirements.txt`
- `data/forecasts.csv`
- `data/cluster_forecasts.csv`
- `data/total_forecasts.csv`
- `data/shap_summary.csv`

### Deploy steps

1. Open [Streamlit Community Cloud](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**.
3. Select your repository, branch (e.g. `main`), and set **Main file path** to `dashboard.py`.
4. Click **Deploy** and wait for the build to finish.
5. Open the public app URL and confirm tabs, sidebar filters, charts, and tables work.

---

## Deploy on Render (alternative)

Use this if you prefer Render or want a second hosting option.

### Blueprint (recommended)

1. Open [Render](https://render.com) and sign in with GitHub.
2. **New +** → **Blueprint** and select this repository.
3. Confirm `render.yaml` is detected, then deploy.
4. When the service is live, open the provided URL.

### Manual Web Service

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0`

---

## Hosted uptime

On free tiers (e.g. Render), services may spin down after idle time; the first load after idle can be slower while the instance starts.
