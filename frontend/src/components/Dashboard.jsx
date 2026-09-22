import React, { useEffect, useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ErrorBar,
} from "recharts";
import { api, figureUrl } from "../api.js";

function useApi(path) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    let alive = true;
    api
      .get(path)
      .then((r) => alive && setData(r.data))
      .catch((e) => alive && setError(e.response?.data?.detail || "Backend not reachable."));
    return () => (alive = false);
  }, [path]);
  return { data, error };
}

export default function Dashboard() {
  const { data: comparison, error: cmpErr } = useApi("/api/comparison");
  const { data: finalMetrics, error: fmErr } = useApi("/api/final-metrics");

  if (cmpErr || fmErr) {
    return (
      <div className="card">
        <p className="error-box">{cmpErr || fmErr}</p>
        <p className="muted">
          Run <code>python -m src.train</code> in the project root first, then start the backend with{" "}
          <code>uvicorn backend.main:app --reload --port 8000</code>.
        </p>
      </div>
    );
  }

  const chartData = (comparison || [])
    .map((r) => ({
      model: r.model,
      auc: Number(r.cv_auc_mean),
      err: Number(r.cv_auc_std) || 0,
      selected: r.selected,
    }))
    .sort((a, b) => a.auc - b.auc);

  return (
    <>
      {finalMetrics && (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Selected model: {finalMetrics.model}</h3>
          <p className="muted">
            Decision threshold {finalMetrics.threshold?.toFixed(2)} &middot; trained on{" "}
            {finalMetrics.n_train} rows, tested on {finalMetrics.n_test} held-out rows
          </p>
          <div className="metric-cards">
            {Object.entries(finalMetrics.test_at_tuned_threshold || {}).map(([k, v]) => (
              <div className="metric-card" key={k}>
                <div className="val">{typeof v === "number" ? v.toFixed(3) : v}</div>
                <div className="lbl">{k.replace(/_/g, " ")}</div>
              </div>
            ))}
          </div>
          <p className="muted" style={{ marginTop: 10 }}>
            Test set has {finalMetrics.n_test} rows, so treat differences of a few points between models
            as noise - see the bootstrap 95% CIs in <code>results/final_metrics.json</code>.
          </p>
        </div>
      )}

      {chartData.length > 0 && (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Cross-validated ROC-AUC by model</h3>
          <ResponsiveContainer width="100%" height={Math.max(260, chartData.length * 34)}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 20, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#dad5c8" horizontal={false} />
              <XAxis type="number" domain={[0.5, 1]} stroke="#4d564f" fontSize={12} />
              <YAxis type="category" dataKey="model" width={160} stroke="#4d564f" fontSize={12} fontFamily="IBM Plex Mono, monospace" />
              <Tooltip
                contentStyle={{ background: "#ffffff", border: "1px solid #dad5c8", color: "#1b211f",
                                fontFamily: "IBM Plex Mono, monospace", fontSize: 12 }}
                formatter={(v) => v.toFixed(4)}
              />
              <Bar dataKey="auc" radius={[0, 6, 6, 0]}>
                {chartData.map((d, i) => (
                  <Cell key={i} fill={d.selected ? "#c1524b" : "#2a6f6f"} />
                ))}
                <ErrorBar dataKey="err" width={4} strokeWidth={1.5} stroke="#4d564f" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="muted">Clay bar = model selected by the one-standard-error rule (see About).</p>
        </div>
      )}

      {comparison && (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>Full comparison table</h3>
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>Model</th><th>Origin</th><th className="num">CV AUC</th><th className="num">Accuracy</th>
                  <th className="num">Precision</th><th className="num">Recall</th><th className="num">F1</th>
                </tr>
              </thead>
              <tbody>
                {comparison.map((r) => (
                  <tr key={r.model} className={r.selected ? "selected-row" : ""}>
                    <td>{r.model}</td>
                    <td>{r.origin}</td>
                    <td className="num">{Number(r.cv_auc_mean).toFixed(4)}</td>
                    <td className="num">{Number(r.accuracy).toFixed(3)}</td>
                    <td className="num">{Number(r.precision).toFixed(3)}</td>
                    <td className="num">{Number(r.recall).toFixed(3)}</td>
                    <td className="num">{Number(r.f1).toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="card">
        <h3 style={{ marginTop: 0 }}>Curves &amp; explainability</h3>
        <div className="grid-2">
          {[
            ["roc_pr_curves.png", "ROC / precision-recall curves (test set)"],
            ["confusion_matrix_final.png", "Confusion matrix (selected model)"],
            ["model_comparison_cv_auc.png", "CV AUC comparison (static version)"],
            ["explain_permutation_importance.png", "Permutation importance"],
          ].map(([name, caption]) => (
            <figure key={name} style={{ margin: 0 }}>
              <img className="fig" src={figureUrl(name)} alt={caption}
                   onError={(e) => (e.target.style.display = "none")} />
              <figcaption className="muted" style={{ marginTop: 4 }}>{caption}</figcaption>
            </figure>
          ))}
        </div>
      </div>
    </>
  );
}
