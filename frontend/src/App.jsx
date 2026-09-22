import React, { useState } from "react";
import PredictForm from "./components/PredictForm.jsx";
import Dashboard from "./components/Dashboard.jsx";
import About from "./components/About.jsx";

const TABS = [
  { id: "predict", label: "Predict" },
  { id: "dashboard", label: "Model Dashboard" },
  { id: "about", label: "About" },
];

export default function App() {
  const [tab, setTab] = useState("predict");

  return (
    <div className="app">
      <header className="hero">
        <h1>Diabetes Risk Prediction</h1>
        <p className="muted">ML comparison &amp; explainable prediction on the PIMA Indians dataset</p>
      </header>

      <div className="disclaimer">
        <strong>Educational prototype - not a medical device.</strong> This classifies whether a record
        resembles the diabetic class of a small, cross-sectional research dataset (adult women of Pima
        heritage). It does not diagnose diabetes and does not predict future onset. Consult a healthcare
        professional for any medical decision.
      </div>

      <nav className="tabs">
        {TABS.map((t) => (
          <button
            key={t.id}
            className={tab === t.id ? "active" : ""}
            onClick={() => setTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      {tab === "predict" && <PredictForm />}
      {tab === "dashboard" && <Dashboard />}
      {tab === "about" && <About />}
    </div>
  );
}
