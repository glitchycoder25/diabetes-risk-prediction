import React, { useState } from "react";
import { api } from "../api.js";

const FIELD_DEFS = [
  { key: "Pregnancies", label: "Pregnancies", min: 0, max: 20, step: 1, default: 2 },
  { key: "Glucose", label: "Plasma glucose (mg/dL)", min: 40, max: 300, step: 1, default: 110 },
  { key: "BloodPressure", label: "Diastolic blood pressure (mm Hg)", min: 30, max: 150, step: 1, default: 70 },
  { key: "SkinThickness", label: "Triceps skin-fold thickness (mm)", min: 0, max: 100, step: 1, default: 25, optional: true },
  { key: "Insulin", label: "2-h serum insulin (µU/mL)", min: 0, max: 900, step: 1, default: 80, optional: true },
  { key: "BMI", label: "BMI (kg/m²)", min: 12, max: 70, step: 0.1, default: 28 },
  { key: "DiabetesPedigreeFunction", label: "Diabetes pedigree function", min: 0.05, max: 2.5, step: 0.01, default: 0.4 },
  { key: "Age", label: "Age (years)", min: 18, max: 100, step: 1, default: 30 },
];

export default function PredictForm() {
  const initial = Object.fromEntries(FIELD_DEFS.map((f) => [f.key, f.default]));
  const [values, setValues] = useState(initial);
  const [unavailable, setUnavailable] = useState({ SkinThickness: false, Insulin: false });
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const setField = (key, v) => setValues((prev) => ({ ...prev, [key]: v }));

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = { ...values };
      for (const k of Object.keys(unavailable)) if (unavailable[k]) payload[k] = 0;
      const { data } = await api.post("/api/predict", payload);
      setResult(data);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
          "Could not reach the backend. Is it running? (uvicorn backend.main:app --reload --port 8000)"
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <form onSubmit={submit}>
        <div className="grid-2">
          {FIELD_DEFS.map((f) => (
            <div key={f.key}>
              <label>{f.label}</label>
              <input
                type="number"
                min={f.min}
                max={f.max}
                step={f.step}
                value={values[f.key]}
                disabled={unavailable[f.key]}
                onChange={(e) => setField(f.key, parseFloat(e.target.value))}
              />
              {f.optional && (
                <div className="checkbox-row" style={{ marginTop: -10 }}>
                  <input
                    type="checkbox"
                    id={`na-${f.key}`}
                    checked={unavailable[f.key]}
                    onChange={(e) =>
                      setUnavailable((prev) => ({ ...prev, [f.key]: e.target.checked }))
                    }
                  />
                  <label htmlFor={`na-${f.key}`} style={{ margin: 0 }}>
                    Not available
                  </label>
                </div>
              )}
            </div>
          ))}
        </div>
        <button className="primary" type="submit" disabled={loading}>
          {loading ? "Predicting..." : "Predict"}
        </button>
      </form>

      {error && <p className="error-box" style={{ marginTop: 14 }}>{error}</p>}

      {result && (
        <div className={`result-banner ${result.predicted_class === "diabetic" ? "pos" : "neg"}`}>
          <strong>
            {result.predicted_class === "diabetic"
              ? "Resembles the diabetic class"
              : "Resembles the non-diabetic class"}
          </strong>
          <div className="muted" style={{ marginTop: 4 }}>
            Model: {result.model_name} &middot; decision threshold {result.threshold.toFixed(2)}
          </div>
          <div className="prob-bar-track">
            <div className="prob-bar-fill" style={{ width: `${(result.probability * 100).toFixed(1)}%` }} />
          </div>
          <div className="muted" style={{ marginTop: 6 }}>
            Model-estimated probability: {(result.probability * 100).toFixed(1)}%
          </div>
        </div>
      )}
    </div>
  );
}
