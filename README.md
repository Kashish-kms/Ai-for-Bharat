# Ai-for-Bharat - Karnataka Renewable Forecast Dashboard

This project is a Streamlit dashboard for AI-generated electricity forecasts for Karnataka solar and wind generation.

## What this app does

- Shows P50 forecast, P10-P90 uncertainty band, actual generation, and naive baseline.
- Supports Plant / Cluster / Karnataka Total views.
- Explains key forecast drivers and operational recommendations.

## Repository structure

- `dashboard.py` - Streamlit app entry point
- `data/` - forecast and SHAP CSV files used by the app
- `requirements.txt` - Python dependencies for deployment
- `render.yaml` - Render deployment config (fallback option)

## Local run

1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run app:
   - `streamlit run dashboard.py`

---

## Exact deploy checklist (Primary: Streamlit Community Cloud)

### A) GitHub prep (required first)

1. Create GitHub repo: `Ai-for-Bharat` (already planned).
2. From this folder, push code:
   - `git add .`
   - `git commit -m "Initial deploy-ready dashboard"`
   - `git push -u origin main`
3. Confirm these files exist in GitHub:
   - `dashboard.py`
   - `requirements.txt`
   - `data/forecasts.csv`
   - `data/cluster_forecasts.csv`
   - `data/total_forecasts.csv`
   - `data/shap_summary.csv`

### B) Deploy on Streamlit Cloud

1. Open [https://share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account.
3. Click **New app**.
4. Select:
   - **Repository**: `Neerajcodes11/Ai-for-Bharat`
   - **Branch**: `main`
   - **Main file path**: `dashboard.py`
5. Click **Deploy**.
6. Wait for build + startup logs to finish.
7. Open the generated public URL and verify:
   - Tabs load correctly
   - Sidebar filters work
   - Charts and tables render

### C) Pre-submission verification

1. Open app URL in incognito/private window.
2. Test at least these cases:
   - Plant view
   - Cluster view
   - Karnataka Total view
3. Confirm no authentication prompt appears.
4. Submit to judges:
   - Live Streamlit URL
   - GitHub repo URL

---

## Fallback deploy checklist (Render)

Use this if Streamlit Cloud is down or slow.

### A) Create Render service

1. Open [https://render.com](https://render.com) and sign in with GitHub.
2. Click **New +** -> **Blueprint** (recommended, uses `render.yaml`).
3. Select repo: `Neerajcodes11/Ai-for-Bharat`.
4. Render detects `render.yaml`; proceed and deploy.
5. Wait until service is live, then open the Render URL.

### B) If not using Blueprint (manual Web Service)

Set:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `streamlit run dashboard.py --server.port $PORT --server.address 0.0.0.0`

Then deploy and verify in browser.

---

## Competition submission package

Submit all 3 for safety:

1. Streamlit live link (primary)
2. Render live link (fallback)
3. GitHub repo link

Optional (recommended): 30-60 second screen recording of app walkthrough.

## Note on uptime

If hosted on Streamlit Cloud or Render, judges can view your app even when your laptop is off.
