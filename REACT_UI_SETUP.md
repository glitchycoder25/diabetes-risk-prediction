# React UI setup

This adds an interactive React frontend on top of the project you already have running. It does not
retrain anything - it reads the model and results your `python -m src.train` run already produced, so
run that first if you haven't.

You'll run **two servers at once**, in **two separate terminal tabs**, both from your project root
(the folder containing `data/`, `src/`, `app/`, and now `backend/` and `frontend/`).

## Terminal 1: backend (serves the model over an API)

```bash
python3 -m pip install -r backend/requirements.txt
python3 -m uvicorn backend.main:app --reload --port 8000
```

Leave this running. Check it worked by opening http://localhost:8000/api/health in your browser - you
should see `{"status":"ok","model_trained":true}`.

## Terminal 2: frontend (the React app)

You need Node.js installed. If `node -v` fails, install it from https://nodejs.org (the LTS version) first.

```bash
cd frontend
npm install
npm run dev
```

It will print a local URL, normally **http://localhost:5173** - open that in your browser.

## What you should see
- **Predict tab** - a form; fill it in and click Predict to get a live probability from your trained model.
- **Model Dashboard tab** - a live chart of cross-validated AUC per model, the full comparison table, and
  the figures your training run saved (ROC/PR curves, confusion matrix, permutation importance).
- **About tab** - methodology and limitations, mirroring the Streamlit app.

## Common issues
- **"Could not reach the backend" in the Predict tab** - Terminal 1 (uvicorn) isn't running, or crashed.
  Check that terminal for errors.
- **CORS error in the browser console** - make sure the frontend really is on port 5173 (the default). If
  you changed the port, add it to `allow_origins` in `backend/main.py`.
- **Dashboard shows "Backend not reachable"** - same as above, or you haven't run `python -m src.train`
  yet, so `results/*.csv` don't exist.
- **`npm: command not found`** - Node.js isn't installed; see the link above.

## Presenting it (e.g. to your professor)
Keep both terminals running side by side during your demo, with the browser open to
`http://localhost:5173`. If you want a single shareable link instead of two local servers, that's a
further step (deploying the FastAPI backend somewhere like Render, and the frontend to Vercel/Netlify) -
ask me and I'll walk you through it once the local version works.
