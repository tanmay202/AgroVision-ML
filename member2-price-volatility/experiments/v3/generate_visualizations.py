"""
AgroVision Member 2 — Volatility V3 Visualization Generator

Generates SVG diagrams for:
1. Confusion matrices (7D & 14D) for Persistence, V3 XGBoost, and V2
2. Probability calibration reliability curves (Uncalibrated vs Sigmoid vs Isotonic)
"""

import json
from pathlib import Path
import numpy as np

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def generate_cm_svg(cm, title, filename, classes=["LOW", "MEDIUM", "HIGH"]):
    """Generates an SVG heatmap for a 3x3 confusion matrix."""
    cm = np.array(cm)
    total = cm.sum()
    max_val = cm.max()

    cell_w, cell_h = 100, 70
    start_x, start_y = 120, 80
    w = start_x + 3 * cell_w + 60
    h = start_y + 3 * cell_h + 80

    svg = [
        f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; font-family:sans-serif;">',
        f'<text x="{w/2}" y="35" text-anchor="middle" font-size="16" font-weight="bold" fill="#1e293b">{title}</text>',
        f'<text x="{start_x + 1.5*cell_w}" y="65" text-anchor="middle" font-size="13" font-weight="600" fill="#475569">Predicted Class</text>',
        f'<text x="25" y="{start_y + 1.5*cell_h}" text-anchor="middle" font-size="13" font-weight="600" fill="#475569" transform="rotate(-90 25 {start_y + 1.5*cell_h})">Actual Class</text>',
    ]

    # Column labels (Predicted)
    for j, c in enumerate(classes):
        cx = start_x + j * cell_w + cell_w / 2
        svg.append(f'<text x="{cx}" y="{start_y - 8}" text-anchor="middle" font-size="12" font-weight="bold" fill="#334155">{c}</text>')

    # Rows (Actual)
    for i, r_c in enumerate(classes):
        cy = start_y + i * cell_h + cell_h / 2
        svg.append(f'<text x="{start_x - 15}" y="{cy + 5}" text-anchor="end" font-size="12" font-weight="bold" fill="#334155">{r_c}</text>')

        for j, c_c in enumerate(classes):
            val = cm[i, j]
            pct = val / total * 100.0
            row_pct = val / max(1, cm[i].sum()) * 100.0

            # Color intensity
            intensity = val / max_val
            # Green gradient for diagonal, blue/red for off-diagonal
            if i == j:
                r = int(240 - intensity * 180)
                g = int(253 - intensity * 90)
                b = int(244 - intensity * 180)
                fill_col = f'rgb({r},{g},{b})'
                text_col = '#ffffff' if intensity > 0.55 else '#064e3b'
            else:
                r = int(255 - intensity * 120)
                g = int(245 - intensity * 160)
                b = int(245 - intensity * 160)
                fill_col = f'rgb({r},{g},{b})'
                text_col = '#ffffff' if intensity > 0.6 else '#1e293b'

            x = start_x + j * cell_w
            y = start_y + i * cell_h

            svg.append(f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill_col}" stroke="#cbd5e1" stroke-width="1"/>')
            svg.append(f'<text x="{x + cell_w/2}" y="{y + cell_h/2 - 5}" text-anchor="middle" font-size="14" font-weight="bold" fill="{text_col}">{val:,}</text>')
            svg.append(f'<text x="{x + cell_w/2}" y="{y + cell_h/2 + 15}" text-anchor="middle" font-size="11" fill="{text_col}">({row_pct:.1f}% row)</text>')

    # Support column
    svg.append(f'<text x="{start_x + 3*cell_w + 30}" y="{start_y - 8}" text-anchor="middle" font-size="11" font-weight="bold" fill="#64748b">Total</text>')
    for i, r_c in enumerate(classes):
        cy = start_y + i * cell_h + cell_h / 2
        r_sum = cm[i].sum()
        svg.append(f'<text x="{start_x + 3*cell_w + 30}" y="{cy + 5}" text-anchor="middle" font-size="12" fill="#64748b">{r_sum:,}</text>')

    # Bottom sum
    svg.append(f'<text x="{start_x - 15}" y="{start_y + 3*cell_h + 25}" text-anchor="end" font-size="11" font-weight="bold" fill="#64748b">Total</text>')
    for j in range(3):
        cx = start_x + j * cell_w + cell_w / 2
        c_sum = cm[:, j].sum()
        svg.append(f'<text x="{cx}" y="{start_y + 3*cell_h + 25}" text-anchor="middle" font-size="12" fill="#64748b">{c_sum:,}</text>')

    svg.append(f'<text x="{start_x + 3*cell_w + 30}" y="{start_y + 3*cell_h + 25}" text-anchor="middle" font-size="12" font-weight="bold" fill="#0f172a">{total:,}</text>')
    svg.append('</svg>')

    out_path = FIGURES_DIR / filename
    with open(out_path, "w") as f:
        f.write("\n".join(svg))
    print(f"Generated CM SVG: {out_path}")


def generate_calibration_svg(cal_data, horizon_label, filename):
    """Generates an SVG reliability diagram comparing Uncalibrated vs Sigmoid vs Isotonic."""
    w, h = 550, 420
    ox, oy = 70, 340 # Origin
    pw, ph = 400, 260 # Plot width and height

    svg = [
        f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; font-family:sans-serif;">',
        f'<text x="{w/2}" y="35" text-anchor="middle" font-size="16" font-weight="bold" fill="#1e293b">Probability Calibration: HIGH Volatility ({horizon_label})</text>',
        f'<text x="{ox + pw/2}" y="{oy + 40}" text-anchor="middle" font-size="12" font-weight="600" fill="#475569">Mean Predicted Probability</text>',
        f'<text x="25" y="{oy - ph/2}" text-anchor="middle" font-size="12" font-weight="600" fill="#475569" transform="rotate(-90 25 {oy - ph/2})">Observed Event Frequency</text>',
    ]

    # Grid lines and ticks
    for tick in np.linspace(0, 1.0, 6):
        x = ox + tick * pw
        y = oy - tick * ph
        # Vertical grid
        svg.append(f'<line x1="{x}" y1="{oy}" x2="{x}" y2="{oy - ph}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{x}" y="{oy + 18}" text-anchor="middle" font-size="10" fill="#64748b">{tick:.1f}</text>')
        # Horizontal grid
        svg.append(f'<line x1="{ox}" y1="{y}" x2="{ox + pw}" y2="{y}" stroke="#e2e8f0" stroke-width="1"/>')
        svg.append(f'<text x="{ox - 10}" y="{y + 4}" text-anchor="end" font-size="10" fill="#64748b">{tick:.1f}</text>')

    # Diagonal perfect calibration line
    svg.append(f'<line x1="{ox}" y1="{oy}" x2="{ox + pw}" y2="{oy - ph}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="4,4"/>')

    # Methods and colors
    methods = [
        ("uncalibrated", "Raw XGBoost (Uncalibrated)", "#ef4444", "circle"),
        ("sigmoid", "Platt / Sigmoid", "#3b82f6", "square"),
        ("isotonic", "Isotonic Regression", "#10b981", "diamond"),
    ]

    for m_key, m_name, color, shape in methods:
        table = cal_data[m_key]["reliability_tables"]["HIGH"]
        pts = []
        for row in table:
            if row["count"] > 10:
                px = ox + row["mean_predicted_prob"] * pw
                py = oy - row["observed_frequency"] * ph
                pts.append((px, py))

        if len(pts) > 1:
            d_str = f"M {pts[0][0]:.1f},{pts[0][1]:.1f} " + " ".join([f"L {p[0]:.1f},{p[1]:.1f}" for p in pts[1:]])
            svg.append(f'<path d="{d_str}" fill="none" stroke="{color}" stroke-width="2"/>')

        for px, py in pts:
            if shape == "circle":
                svg.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}"/>')
            elif shape == "square":
                svg.append(f'<rect x="{px-3.5:.1f}" y="{py-3.5:.1f}" width="7" height="7" fill="{color}"/>')
            else:
                svg.append(f'<polygon points="{px:.1f},{py-4:.1f} {px+4:.1f},{py:.1f} {px:.1f},{py+4:.1f} {px-4:.1f},{py:.1f}" fill="{color}"/>')

    # Legend
    leg_x = ox + 15
    leg_y = oy - ph + 20
    svg.append(f'<rect x="{leg_x}" y="{leg_y}" width="220" height="90" fill="#ffffff" fill-opacity="0.9" stroke="#cbd5e1" rx="4"/>')
    svg.append(f'<line x1="{leg_x+10}" y1="{leg_y+15}" x2="{leg_x+30}" y2="{leg_y+15}" stroke="#94a3b8" stroke-dasharray="3,3" stroke-width="2"/>')
    svg.append(f'<text x="{leg_x+35}" y="{leg_y+18}" font-size="10" fill="#334155">Perfect Calibration</text>')

    for idx, (m_key, m_name, color, _) in enumerate(methods):
        ly = leg_y + 35 + idx * 18
        ece_val = cal_data[m_key]["class_ece"]["HIGH"]
        svg.append(f'<circle cx="{leg_x+20}" cy="{ly-3}" r="4" fill="{color}"/>')
        svg.append(f'<text x="{leg_x+35}" y="{ly}" font-size="10" fill="#334155">{m_name} (ECE={ece_val:.3f})</text>')

    svg.append('</svg>')
    out_path = FIGURES_DIR / filename
    with open(out_path, "w") as f:
        f.write("\n".join(svg))
    print(f"Generated Calibration SVG: {out_path}")


def main():
    with open(RESULTS_DIR / "v3_full_pooled_metrics.json") as f:
        metrics = json.load(f)

    with open(RESULTS_DIR / "v3_calibration_report.json") as f:
        cal = json.load(f)

    # 1. Generate Confusion Matrix SVGs
    for m in metrics:
        if m["period"] == "Holdout (2023-2024)":
            h = m["horizon"]
            model = m["model"]
            cm = m["confusion_matrix"]
            if "Calendar Persistence" in model:
                generate_cm_svg(cm, f"Holdout Confusion Matrix: Calendar Persistence ({h})", f"cm_{h.lower()}_persistence.svg")
            elif "Default Thresholds" in model:
                generate_cm_svg(cm, f"Holdout Confusion Matrix: V3 XGBoost Balanced ({h})", f"cm_{h.lower()}_v3_xgboost.svg")
            elif "Tuned Thresholds" in model:
                generate_cm_svg(cm, f"Holdout Confusion Matrix: V3 XGBoost Tuned ({h})", f"cm_{h.lower()}_v3_tuned.svg")
            elif "Normalized V2" in model:
                generate_cm_svg(cm, f"Holdout Confusion Matrix: Production V2 Model ({h})", f"cm_{h.lower()}_v2_production.svg")

    # 2. Generate Calibration SVGs
    generate_calibration_svg(cal["7D"], "7D Horizon", "calibration_curve_7d.svg")
    generate_calibration_svg(cal["14D"], "14D Horizon", "calibration_curve_14d.svg")


if __name__ == "__main__":
    main()
