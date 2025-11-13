# AyurGenixAI

Lightweight Streamlit app for bilingual (English/Hindi) Ayurveda symptom→disease mapping and treatment recommendations.

Quick local run

```bash
python -m venv .venv
# Windows Git Bash
source .venv/Scripts/activate
pip install -r requirements.txt
python -m streamlit run "app.py"
```

Prepare for Streamlit Cloud deployment

1. Push this repository to GitHub (public or private).
2. Go to https://share.streamlit.io and sign in with your GitHub account.
3. Click "New app", select the repository and branch, and set the `app.py` file path.
4. Streamlit Cloud will install dependencies from `requirements.txt` and start the app. Use the **Secrets** settings in Streamlit Cloud to add API keys (OpenAI, etc.) rather than committing them.

Notes & tips

- Remove large output files (images, large CSV exports) before pushing, or use Git LFS.
- Add any sensitive keys to `secrets.toml` in Streamlit Cloud (not in repo).
- If you need a specific Python version, add a `.python-version` or set it in Streamlit Cloud settings.

If you want, I can:
- Try pushing the prepared repo here (requires credentials / PAT or SSH key), or
- Show exact commands to push from your machine and deploy on Streamlit Cloud.
