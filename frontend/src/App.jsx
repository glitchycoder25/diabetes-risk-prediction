import React, { useEffect, useState } from "react";
import PredictForm from "./components/PredictForm.jsx";
import Dashboard from "./components/Dashboard.jsx";
import About from "./components/About.jsx";
import { api } from "./api.js";

const TABS = [
  { id: "predict", label: "Predict" },
  { id: "dashboard", label: "Model Dashboard" },
  { id: "about", label: "About" },
];

export default function App() {
  const [tab, setTab] = useState("predict");
  const [headline, setHeadline] = useState(null);

  useEffect(() => {
    let alive = true;
    api
      .get("/api/final-metrics")
      .then(({ data }) => {
        if (!alive) return;
        const auc = data.test_at_tuned_threshold?.roc_auc;
        setHeadline(auc != null ? { auc, model: data.model } : null);
      })
      .catch(() => alive && setHeadline(null));
    return () => (alive = false);
  }, []);

  return (
    <div className="app">
      <header className="hero">
        <div className="title-block">
          <h1>Diabetes Risk Prediction</h1>
          <p className="subtitle">
            A comparative machine-learning study on the PIMA Indians dataset — leakage-checked,
            cross-validated, and explained.
          </p>
        </div>
        {headline && (
          <div className="hero-stat">
            <div className="num">{headline.auc.toFixed(3)}</div>
            <div className="lbl">test ROC-AUC, {headline.model}</div>
          </div>
        )}
      </header>

      <div className="disclaimer">
        <strong>Educational prototype, not a medical device.</strong> This classifies whether a
        record resembles the diabetic class of a small, cross-sectional research dataset (adult
        women of Pima heritage). It does not diagnose diabetes and does not predict future onset.
        Consult a healthcare professional for any medical decision.
      </div>

      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>

      {tab === "predict" && <PredictForm />}
      {tab === "dashboard" && <Dashboard />}
      {tab === "about" && <About />}

      <footer className="credit">B.Tech minor project — ML pipeline, FastAPI backend, React frontend.</footer>
    </div>
  );
}
