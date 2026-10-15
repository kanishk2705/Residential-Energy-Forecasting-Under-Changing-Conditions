"""Figures from results/final (no numbers are typed by hand).

Usage: python scripts/make_figures.py [--results DIR] [--out DIR] [--raw CSV]
Palette: validated categorical slots 1-3 (blue, orange, aqua) plus neutral gray;
aqua is never used for text (contrast), identity is carried by legends/direct labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
HZ = {"1": "10 min ahead", "6": "1 h ahead", "144": "1 day ahead"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.solid_capstyle": "round",
})


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def dot_ci(ax, y, est, lo, hi, color, label=None, size=6):
    ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
    ax.plot([est], [y], "o", ms=size, color=color, mec=SURFACE, mew=2, zorder=3, label=label)


def short(name: str) -> str:
    return (name.replace("lgbm|", "LightGBM · ").replace("ridge|", "Ridge · ").replace("rf|", "Random forest · ")
            .replace("B1_persistence", "Persistence").replace("B5_slot_of_day_median", "Slot-of-day median")
            .replace("B2_mean_last_hour", "Last-hour mean").replace("B6_slot_of_week_median", "Slot-of-week median"))


def fig_timeline(raw: Path, out: Path, test_start="2016-04-25", dev_end="2016-04-22 23:50"):
    df = pd.read_csv(raw, parse_dates=["date"], index_col="date")
    df = df.apply(pd.to_numeric, errors="coerce")
    daily_y = df["Appliances"].resample("D").mean()
    node = df["T_out"][df.index.minute == 0].resample("D").mean()
    fig, axs = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw=dict(hspace=0.12))
    axs[0].plot(daily_y.index, daily_y, color=GRAY, lw=2)
    axs[0].set_ylabel("Appliances\n(daily mean, Wh / 10 min)")
    axs[1].plot(node.index, node, color=BLUE, lw=2)
    axs[1].set_ylabel("Outdoor temperature\n(daily mean, °C)")
    folds = [("2016-02-29", "2016-03-13"), ("2016-03-14", "2016-03-27"),
             ("2016-03-28", "2016-04-10"), ("2016-04-11", "2016-04-22")]
    for ax in axs:
        for i, (a, b) in enumerate(folds):
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b) + pd.Timedelta(days=1), color=GRAY,
                       alpha=0.12 if i % 2 == 0 else 0.05, lw=0)
        ax.axvspan(pd.Timestamp(test_start), df.index.max(), color=ORANGE, alpha=0.12, lw=0)
    ymax = axs[0].get_ylim()[1]
    for i, (a, b) in enumerate(folds):
        axs[0].text(pd.Timestamp(a) + pd.Timedelta(days=7), ymax, f"V{i+1}", ha="center", va="bottom",
                    color=INK2, fontsize=9)
    axs[0].text(pd.Timestamp(test_start) + pd.Timedelta(days=16), ymax, "Final test (read once)", ha="center",
                va="bottom", color=INK2, fontsize=9)
    axs[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=3))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axs[0].set_title("The data drifts from winter into spring; validation folds V1-V4 precede the test", pad=18)
    save(fig, out, "fig1_timeline")


def fig_headline(R, out):
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    hs = list(R["horizons"])
    for i, h in enumerate(hs):
        H = R["horizons"][h]
        b, m = H["names"]["B_star"], H["names"]["M_star"]
        mb, mm = H["methods"][b]["mae"], H["methods"][m]["mae"]
        dot_ci(ax, i + 0.14, mb["estimate"], mb["ci_lo"], mb["ci_hi"], ORANGE, "Best baseline (chosen on dev)" if i == 0 else None)
        dot_ci(ax, i - 0.14, mm["estimate"], mm["ci_lo"], mm["ci_hi"], BLUE, "LightGBM, own-history inputs" if i == 0 else None)
    ax.set_yticks(range(len(hs)), [HZ[h] for h in hs]); ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error on the final test (Wh per 10 min), 95% CI")
    ax.set_title("Error on the final test")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, fontsize=8.5)
    ax = axs[1]
    for i, h in enumerate(hs):
        d = R["horizons"][h]["E1_headline"]["delta_mae_M_minus_B"]
        ax.plot(d["ci98_33"], [i, i], color=BLUE, lw=1.2, alpha=0.45)
        dot_ci(ax, i, d["estimate"], d["ci_lo"], d["ci_hi"], BLUE)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(hs)), [HZ[h] for h in hs]); ax.invert_yaxis()
    ax.set_xlabel("Model minus best baseline (Wh); below 0 = model better")
    ax.set_title("Paired difference (thick 95%, thin 98.3% CI)")
    fig.tight_layout()
    save(fig, out, "fig2_headline")


def fig_inputs(R, out):
    fig, ax = plt.subplots(figsize=(8, 3.4))
    hs = list(R["horizons"])
    cols = {"A": BLUE, "B": ORANGE, "C": AQUA}
    lab = {"A": "A: own history", "B": "B: + indoor sensors", "C": "C: + airport weather"}
    for i, h in enumerate(hs):
        for j, f in enumerate("ABC"):
            m = R["horizons"][h]["E2_input_groups"]["mae"][f]
            dot_ci(ax, i + (j - 1) * 0.2, m["estimate"], m["ci_lo"], m["ci_hi"], cols[f], lab[f] if i == 0 else None)
    ax.set_yticks(range(len(hs)), [HZ[h] for h in hs]); ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error on the final test (Wh per 10 min), 95% CI")
    ax.set_title("Error by input group on the final test")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8.5)
    fig.tight_layout()
    save(fig, out, "fig3_inputs")


def fig_policies(R, out):
    names = ["P1_expanding_weekly", "P2_sliding28_weekly", "P3_drift_triggered", "P4_expanding_daily"]
    nice = {"P1_expanding_weekly": "Retrain weekly (expanding)", "P2_sliding28_weekly": "Retrain weekly (28-day window)",
            "P3_drift_triggered": "Retrain when drift flagged", "P4_expanding_daily": "Retrain daily (expanding)"}
    hs = list(R["horizons"])
    fig, axs = plt.subplots(1, len(hs), figsize=(10, 3.3), sharey=True)
    for ax, h in zip(axs, hs):
        P = R["horizons"][h]["E3_policies"]
        for i, n in enumerate(names):
            d = P[n]["delta_mae_vs_P0"]
            dot_ci(ax, i, d["estimate"], d["ci_lo"], d["ci_hi"], BLUE)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(HZ[h], fontsize=10)
        ax.set_xlabel("MAE minus frozen model (Wh)")
    counts = R["horizons"][hs[0]]["E3_policies"]
    axs[0].set_yticks(range(len(names)), [f"{nice[n]} ({counts[n]['retrains']}×)" for n in names])
    axs[0].invert_yaxis()
    fig.suptitle("Does retraining help? Paired MAE difference vs. a frozen model, 95% CI (retrain count at 10 min)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, out, "fig4_policies")


def fig_intervals(R, out):
    hs = list(R["horizons"])
    spec = [("I0_raw", "Raw quantiles", ORANGE), ("I1_static_cqr", "Calibrated once", GRAY),
            ("I2_rolling_cqr", "Recalibrated daily", BLUE), ("I3_aci", "Adaptive (ACI)", AQUA)]
    fig, axs = plt.subplots(1, len(hs), figsize=(10, 3.3), sharey=True)
    for ax, h in zip(axs, hs):
        E = R["horizons"][h]["E4_intervals"]
        for k, lbl, c in spec:
            cw = E[k]["coverage_by_week"]
            w = sorted(cw, key=int)
            ax.plot([int(x) + 1 for x in w], [cw[x] for x in w], "-o", color=c, lw=2, ms=5, mec=SURFACE, mew=1.5,
                    label=lbl)
        ax.axhline(0.8, color=INK2, lw=1)
        ax.set_title(HZ[h], fontsize=10)
        ax.set_xlabel("Test week")
        ax.set_xticks(range(1, 6))
    for ax in axs:
        ax.set_ylim(0.65, 0.95)
    axs[0].set_ylabel("Share of outcomes inside\nthe 80% interval")
    h_, l_ = axs[-1].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Interval coverage by test week (nominal 80%, dark line)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    save(fig, out, "fig5_intervals")


def fig_shift(R, out, h="6"):
    H = R["horizons"][h]
    dd = pd.DataFrame(H["E5_daily"])
    dd["day"] = pd.to_datetime(dd["day"])
    fig, axs = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True, gridspec_kw=dict(hspace=0.12))
    axs[0].plot(dd["day"], dd["shift_auc_monitor"], color=BLUE, lw=2)
    axs[0].axhline(0.65, color=INK2, lw=1)
    axs[0].text(dd["day"].iloc[0], 0.655, "retrain-trigger threshold 0.65", color=INK2, fontsize=8.5, va="bottom")
    axs[0].set_ylabel("Shift monitor\n(domain AUC)")
    axs[1].plot(dd["day"], dd["mae_ratio_L_over_B"], color=BLUE, lw=2)
    axs[1].axhline(1, color=INK2, lw=1)
    axs[1].set_ylabel("Daily MAE: model\n÷ best baseline")
    axs[1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axs[0].set_title("Drift monitor and daily relative error across the test period (exploratory)", pad=8)
    save(fig, out, "fig6_shift")


def fig_example(R, out, results_dir: Path, h="6"):
    p = pd.read_csv(results_dir / f"predictions_h{h}.csv", parse_dates=["origin"], index_col="origin")
    t0 = pd.Timestamp("2016-05-02")
    w = p.loc[t0: t0 + pd.Timedelta(days=2) - pd.Timedelta(minutes=10)]
    # values are for the target time origin + h steps; plot at target time
    x = w.index + pd.Timedelta(minutes=10 * int(h))
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.fill_between(x, w["I2_rolling_cqr_lo"], w["I2_rolling_cqr_hi"], color=BLUE, alpha=0.12, lw=0,
                    label="80% interval (recalibrated daily)")
    ax.plot(x, w["y"], color=GRAY, lw=1.2, label="Actual")
    ax.plot(x, w["L_star"], color=BLUE, lw=2, label="LightGBM forecast")
    ax.set_ylabel("Appliances (Wh per 10 min)")
    ax.set_title(f"Two test days, forecasts issued {HZ[h]}")
    ax.legend(loc="upper left", fontsize=8.5, ncol=3)
    save(fig, out, "fig7_example")


def fig_diagnostics(results_dir: Path, out: Path, h="144"):
    path = results_dir / "diagnostics.json"
    if not path.exists():
        return
    D = json.loads(path.read_text())[f"h{h}_group_ablation"]
    names = [k for k in D if k.startswith("A + ")] + ["C (everything)"]
    base_dev = D["A (own history)"]["dev_mean_fold_mae"]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    for i, k in enumerate(names):
        t = D[k]["test_minus_A"]
        dot_ci(ax, i + 0.13, t["estimate"], t["ci_lo"], t["ci_hi"], BLUE, "Final test, 95% CI (post-hoc)" if i == 0 else None)
        d = D[k]["dev_mean_fold_mae"] - base_dev
        ax.plot([d], [i - 0.13], "o", ms=7, mfc=SURFACE, mec=GRAY, mew=2, label="Validation folds" if i == 0 else None)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(range(len(names)), [n.replace("A + ", "+ ").replace("on_site", "on-site").replace("_", " ") for n in names]); ax.invert_yaxis()
    ax.set_xlabel("MAE added to own-history inputs (Wh); above 0 = worse")
    ax.set_title(f"Which added inputs hurt the {HZ[h]} forecast (post-hoc)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8.5)
    save(fig, out, "fig8_input_groups")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "final")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "figures")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw" / "energydata_complete.csv")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    R = json.loads((a.results / "results.json").read_text())
    fig_timeline(a.raw, a.out)
    fig_headline(R, a.out)
    fig_inputs(R, a.out)
    fig_policies(R, a.out)
    fig_intervals(R, a.out)
    fig_shift(R, a.out)
    fig_example(R, a.out, a.results)
    fig_diagnostics(a.results, a.out)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
