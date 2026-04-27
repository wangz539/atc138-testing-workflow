# comparison_batch.py
# Batch comparison of ATC-138 Python vs MATLAB recovery outputs
# Expects each model folder to contain:
#   - recovery_outputs.json (or recovery_outputs)
#   - recovery_outputs_MATLAB.json (or recovery_outputs_MATLAB)

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------

def safe_makedirs(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    safe_makedirs(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def auc_to_1yr(y, t, T_END=365):
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)

    if t[0] > 0:
        t = np.insert(t, 0, 0.0)
        y = np.insert(y, 0, y[0])

    if t[-1] < T_END:
        t = np.append(t, T_END)
        y = np.append(y, y[-1])

    if t[-1] > T_END:
        mask = t < T_END
        t_trunc = t[mask]
        y_trunc = y[mask]
        y_end = np.interp(T_END, t, y)
        t = np.append(t_trunc, T_END)
        y = np.append(y_trunc, y_end)

    return float(np.trapz(y, t))


def standardize_to_grid(t, y, T_END=365, dt=1.0, extend="last"):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    t_grid = np.arange(0.0, T_END + dt, dt)

    if t.size == 0:
        raise ValueError("Empty time array.")
    if t[0] > 0.0:
        t = np.insert(t, 0, 0.0)
        y = np.insert(y, 0, y[0])

    y_grid = np.interp(t_grid, t, y)

    if t[-1] < T_END:
        if extend == "last":
            y_grid[t_grid > t[-1]] = y[-1]
        elif extend == "zero":
            y_grid[t_grid > t[-1]] = 0.0
        else:
            raise ValueError("extend must be 'last' or 'zero'")

    return t_grid, y_grid


def curve_diff_metrics(y_py, y_mat, t_grid):
    d = np.abs(y_py - y_mat)

    mae = float(np.mean(d))
    rmse = float(np.sqrt(np.mean(d**2)))
    p50 = float(np.quantile(d, 0.50))
    p95 = float(np.quantile(d, 0.95))
    p99 = float(np.quantile(d, 0.99))
    max_abs = float(np.max(d))

    i_max = int(np.argmax(d))
    t_at_max = float(t_grid[i_max])

    return {
        "MAE": mae,
        "RMSE": rmse,
        "P50_abs": p50,
        "P95_abs": p95,
        "P99_abs": p99,
        "Max_abs": max_abs,
        "t_at_max": t_at_max,
    }


def summarize_stats(label, py_vals, mat_vals):
    py_vals = np.array(py_vals, dtype=float)
    mat_vals = np.array(mat_vals, dtype=float)

    stats = {}
    stats["metric"] = label
    stats["py_mean"] = float(np.mean(py_vals))
    stats["mat_mean"] = float(np.mean(mat_vals))
    stats["py_p25"], stats["py_p50"], stats["py_p75"] = map(float, np.percentile(py_vals, [25, 50, 75]))
    stats["mat_p25"], stats["mat_p50"], stats["mat_p75"] = map(float, np.percentile(mat_vals, [25, 50, 75]))

    if stats["mat_mean"] != 0:
        stats["mean_pct_diff"] = float(100.0 * abs(stats["py_mean"] - stats["mat_mean"]) / stats["mat_mean"])
    else:
        stats["mean_pct_diff"] = float("nan")

    stats["pass_mean_3pct"] = (stats["mean_pct_diff"] <= 3.0) if not np.isnan(stats["mean_pct_diff"]) else True
    return stats


def find_outputs(model_dir: str):
    """
    Allow either:
      recovery_outputs.json OR recovery_outputs
      recovery_outputs_MATLAB.json OR recovery_outputs_MATLAB
    """
    candidates_py = ["recovery_outputs.json", "recovery_outputs"]
    candidates_mat = ["recovery_outputs_MATLAB.json", "recovery_outputs_MATLAB"]

    py_path = next((os.path.join(model_dir, f) for f in candidates_py
                    if os.path.isfile(os.path.join(model_dir, f))), None)
    mat_path = next((os.path.join(model_dir, f) for f in candidates_mat
                     if os.path.isfile(os.path.join(model_dir, f))), None)

    return py_path, mat_path


# -----------------------------
# Plotting
# -----------------------------

def plot_system_breakdowns(
    model_name: str,
    py_sys: np.ndarray,
    py_sys_bkdwns: np.ndarray,
    py_days: np.ndarray,
    mat_sys: np.ndarray,
    mat_sys_bkdwns: np.ndarray,
    mat_days: np.ndarray,
    func_tag: str,
    out_path: str
):

    head_order = []
    for i in range(len(mat_sys)):
        idx = np.where(py_sys == mat_sys[i])[0]
        head_order.append(idx[0] if len(idx) else None)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(mat_sys))]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=[24, 8], sharey=True)

    n = len(mat_sys)
    splits = [0, int(np.ceil(n / 3)), int(np.ceil(2 * n / 3)), n]

    for panel in range(3):
        for i in range(splits[panel], splits[panel + 1]):
            idx_py = head_order[i]
            if idx_py is None:
                continue

            ax[panel].plot(py_days, py_sys_bkdwns[idx_py],
                           linestyle="solid", color=colors[i],
                           label=f"{py_sys[idx_py]} - Python", linewidth=1.5)
            ax[panel].plot(mat_days, mat_sys_bkdwns[i],
                           linestyle="dashed", color=colors[i],
                           marker="x", label=f"{mat_sys[i]} - MATLAB", linewidth=1.0)

        ax[panel].set_ylim([0, 1.1])
        ax[panel].set_xlim([0, float(mat_days[-1])])
        ax[panel].set_xlabel("Days")
        ax[panel].grid(True)
        ax[panel].legend()

    ax[0].set_ylabel("Fraction Unresolved")
    fig.suptitle(f"{model_name} - {func_tag}")

    safe_makedirs(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Comparison model by model
# -----------------------------

def compare_one_model(
    model_dir: str,
    out_dir: str | None = None,
    T_END: int = 365,
    dt: float = 1.0,
    tol_p95: float = 0.04,  # system PASS threshold 
    tol_mae: float = 0.02   # system PASS threshold
):



    model_name = os.path.basename(os.path.normpath(model_dir))
    out_dir = out_dir or model_dir

    py_path, mat_path = find_outputs(model_dir)
    if not py_path or not mat_path:
        raise FileNotFoundError(f"Missing recovery outputs in: {model_dir}")

    functionality_python = load_json(py_path)
    functionality_matlab = load_json(mat_path)

    if "functionality" in functionality_matlab:
        functionality_matlab = functionality_matlab["functionality"]

    # --- Extract high-level series (days)
    reoc_py = functionality_python["recovery"]["reoccupancy"]["building_level"]["recovery_day"]
    func_py = functionality_python["recovery"]["functional"]["building_level"]["recovery_day"]
    reoc_mat = functionality_matlab["recovery"]["reoccupancy"]["building_level"]["recovery_day"]
    func_mat = functionality_matlab["recovery"]["functional"]["building_level"]["recovery_day"]

    # full repair (days)
    percent_recovered = np.array(functionality_matlab["recovery"]["reoccupancy"]["recovery_trajectory"]["percent_recovered"])
    if len(percent_recovered) > 2:
        full_py = np.amax(np.array(functionality_python["building_repair_schedule"]["full"]["repair_complete_day"]["per_story"]), axis=1)
        full_mat = np.amax(np.array(functionality_matlab["building_repair_schedule"]["full"]["repair_complete_day"]["per_story"]), axis=1)
    else:
        full_py = np.array(functionality_python["building_repair_schedule"]["full"]["repair_complete_day"]["per_story"])
        full_mat = np.array(functionality_matlab["building_repair_schedule"]["full"]["repair_complete_day"]["per_story"])

    # --- Extract trajectories
    reoc_rec_py = np.mean(functionality_python["recovery"]["reoccupancy"]["recovery_trajectory"]["recovery_day"], axis=0)
    func_rec_py = np.mean(functionality_python["recovery"]["functional"]["recovery_trajectory"]["recovery_day"], axis=0)
    reoc_rec_mat = np.mean(functionality_matlab["recovery"]["reoccupancy"]["recovery_trajectory"]["recovery_day"], axis=0)
    func_rec_mat = np.mean(functionality_matlab["recovery"]["functional"]["recovery_trajectory"]["recovery_day"], axis=0)

    # --- Extract breakdown curves
    bkd_py_reoc = np.array(functionality_python["recovery"]["reoccupancy"]["breakdowns"]["system_breakdowns"], dtype=float)
    bkd_py_func = np.array(functionality_python["recovery"]["functional"]["breakdowns"]["system_breakdowns"], dtype=float)
    bkd_mat_reoc = np.array(functionality_matlab["recovery"]["reoccupancy"]["breakdowns"]["system_breakdowns"], dtype=float)
    bkd_mat_func = np.array(functionality_matlab["recovery"]["functional"]["breakdowns"]["system_breakdowns"], dtype=float)

    sys_py_reoc = np.array(functionality_python["recovery"]["reoccupancy"]["breakdowns"]["system_names"])
    sys_py_func = np.array(functionality_python["recovery"]["functional"]["breakdowns"]["system_names"])
    sys_mat_reoc = np.array(functionality_matlab["recovery"]["reoccupancy"]["breakdowns"]["system_names"])
    sys_mat_func = np.array(functionality_matlab["recovery"]["functional"]["breakdowns"]["system_names"])

    days_py_reoc = np.array(functionality_python["recovery"]["reoccupancy"]["breakdowns"]["perform_targ_days"], dtype=float)
    days_py_func = np.array(functionality_python["recovery"]["functional"]["breakdowns"]["perform_targ_days"], dtype=float)
    days_mat_reoc = np.array(functionality_matlab["recovery"]["reoccupancy"]["breakdowns"]["perform_targ_days"], dtype=float)
    days_mat_func = np.array(functionality_matlab["recovery"]["functional"]["breakdowns"]["perform_targ_days"], dtype=float)

    # -----------------
    # Save high-level table
    # -----------------
    high_rows = [
        summarize_stats("Reoccupancy", reoc_py, reoc_mat),
        summarize_stats("Functional", func_py, func_mat),
        summarize_stats("Full repair", full_py, full_mat),
    ]
    write_csv(
        os.path.join(out_dir, "comparison_outputs", "high_level.csv"),
        high_rows,
        fieldnames=list(high_rows[0].keys())
    )

    # -----------------
    # Plots
    # -----------------
    safe_makedirs(os.path.join(out_dir, "comparison_outputs"))

    plt.figure(figsize=[8, 5])
    plt.plot(reoc_rec_py, percent_recovered * 100, linestyle="solid", linewidth=1.0, label="Mean Reoccupancy - Python")
    plt.plot(reoc_rec_mat, percent_recovered * 100, linestyle="dashed", linewidth=2.0, label="Mean Reoccupancy - MATLAB")
    plt.plot(func_rec_py, percent_recovered * 100, linestyle="solid", linewidth=1.0, label="Mean Functional - Python")
    plt.plot(func_rec_mat, percent_recovered * 100, linestyle="dashed", linewidth=2.0, label="Mean Functional - MATLAB")
    plt.plot([np.mean(full_py), np.mean(full_py)], [0, 100], linestyle="solid", linewidth=1.0, label="Mean Full repair - Python")
    plt.plot([np.mean(full_mat), np.mean(full_mat)], [0, 100], linestyle="dashed", linewidth=2.0, label="Mean Full repair - MATLAB")
    plt.xlim([0, 1.1 * max(float(np.mean(full_mat)), float(np.mean(full_py)))])
    plt.grid(True)
    plt.legend()
    plt.xlabel("Days")
    plt.ylabel("Percent recovered")
    plt.title(f"{model_name} - Recovery trajectory")
    plt.savefig(os.path.join(out_dir, "comparison_outputs", "recovery_trajectory.jpg"), dpi=150, bbox_inches="tight")
    plt.close()

    plot_system_breakdowns(
        model_name, sys_py_reoc, bkd_py_reoc, days_py_reoc,
        sys_mat_reoc, bkd_mat_reoc, days_mat_reoc,
        "Reoccupancy",
        os.path.join(out_dir, "comparison_outputs", "system_breakdowns_reoccupancy.jpg")
    )
    plot_system_breakdowns(
        model_name, sys_py_func, bkd_py_func, days_py_func,
        sys_mat_func, bkd_mat_func, days_mat_func,
        "Functional",
        os.path.join(out_dir, "comparison_outputs", "system_breakdowns_functional.jpg")
    )

    # -----------------
    # AUC tables (kept as before)
    # -----------------
    def compute_system_auc_rows(tag, sys_py, y_py, t_py, sys_mat, y_mat, t_mat):
        rows = []
        for i, name in enumerate(sys_mat):
            idx = np.where(sys_py == name)[0]
            if len(idx) != 1:
                rows.append({
                    "tag": tag, "system": str(name),
                    "auc_py": "", "auc_mat": auc_to_1yr(y_mat[i], t_mat, T_END=T_END),
                    "pct_diff": "", "pass_1pct": ""
                })
                continue
            idx = idx[0]
            auc_py = auc_to_1yr(y_py[idx], t_py, T_END=T_END)
            auc_mat = auc_to_1yr(y_mat[i], t_mat, T_END=T_END)
            pct = (100.0 * abs(auc_py - auc_mat) / auc_mat) if auc_mat != 0 else float("nan")
            rows.append({
                "tag": tag, "system": str(name),
                "auc_py": auc_py, "auc_mat": auc_mat,
                "pct_diff": pct,
                "pass_1pct": (pct <= 1.0) if not np.isnan(pct) else True
            })
        return rows

    auc_reoc = compute_system_auc_rows("Reoccupancy", sys_py_reoc, bkd_py_reoc, days_py_reoc,
                                       sys_mat_reoc, bkd_mat_reoc, days_mat_reoc)
    auc_func = compute_system_auc_rows("Functional", sys_py_func, bkd_py_func, days_py_func,
                                       sys_mat_func, bkd_mat_func, days_mat_func)

    write_csv(os.path.join(out_dir, "comparison_outputs", "auc_reoccupancy.csv"),
              auc_reoc, fieldnames=list(auc_reoc[0].keys()))
    write_csv(os.path.join(out_dir, "comparison_outputs", "auc_functional.csv"),
              auc_func, fieldnames=list(auc_func[0].keys()))

    # -----------------
    # Pointwise curve comparison tables
    # -----------------
    def compute_curve_rows(tag, sys_py, y_py_all, t_py, sys_mat, y_mat_all, t_mat):
        t_grid = np.arange(0.0, T_END + dt, dt)
        rows = []
        for i, name in enumerate(sys_mat):
            idx = np.where(sys_py == name)[0]
            if len(idx) != 1:
                rows.append({"tag": tag, "system": str(name), "missing_in_python": True})
                continue
            idx = idx[0]

            _, y_py = standardize_to_grid(t_py,  y_py_all[idx], T_END=T_END, dt=dt, extend="last")
            _, y_mt = standardize_to_grid(t_mat, y_mat_all[i],   T_END=T_END, dt=dt, extend="last")

            m = curve_diff_metrics(y_py, y_mt, t_grid)
            passed = (m["P95_abs"] <= tol_p95) and (m["MAE"] <= tol_mae)

            rows.append({
                "tag": tag,
                "system": str(name),
                **m,
                "PASS": passed,
                "missing_in_python": False
            })
        return rows

    curve_reoc = compute_curve_rows("Reoccupancy", sys_py_reoc, bkd_py_reoc, days_py_reoc,
                                    sys_mat_reoc, bkd_mat_reoc, days_mat_reoc)
    curve_func = compute_curve_rows("Functional", sys_py_func, bkd_py_func, days_py_func,
                                    sys_mat_func, bkd_mat_func, days_mat_func)

    def fieldnames_union(rows):
        keys = set()
        for r in rows:
            keys |= set(r.keys())
        prefer = ["tag", "system", "MAE", "P95_abs", "Max_abs", "t_at_max", "PASS", "missing_in_python"]
        return [k for k in prefer if k in keys] + sorted([k for k in keys if k not in prefer])

    write_csv(os.path.join(out_dir, "comparison_outputs", "curve_reoccupancy.csv"),
              curve_reoc, fieldnames=fieldnames_union(curve_reoc))
    write_csv(os.path.join(out_dir, "comparison_outputs", "curve_functional.csv"),
              curve_func, fieldnames=fieldnames_union(curve_func))

    # -----------------
    # Build GLOBAL-MODEL metrics 
    # -----------------
    # Mean functional recovery times (days)
    mean_func_py = high_rows[1]["py_mean"]
    mean_func_mat = high_rows[1]["mat_mean"]

    func_mean_abs_diff_days = float(abs(mean_func_py - mean_func_mat))
    func_mean_rel_diff_pct = float((100.0 * func_mean_abs_diff_days / mean_func_mat) if mean_func_mat != 0 else float("nan"))

    # System-wise worst MAE / worst P95 across ALL systems (reoc + func)
    valid_rows = [r for r in (curve_reoc + curve_func) if not r.get("missing_in_python")]

    # If somehow empty, default to nan
    if valid_rows:
        worst_system_MAE = float(np.nanmax([r.get("MAE", np.nan) for r in valid_rows]))
        worst_system_P95_abs = float(np.nanmax([r.get("P95_abs", np.nan) for r in valid_rows]))
    else:
        worst_system_MAE = float("nan")
        worst_system_P95_abs = float("nan")


    # We use MATLAB mean functional recovery time as the baseline shown.
    mean_func_recovery_time_days = float(mean_func_mat)

    # -----------------
    # System-level rows for global system tables
    # -----------------
    sys_rows = []
    for r in curve_reoc:
        if r.get("missing_in_python"):
            continue
        sys_rows.append({
            "model": model_name,
            "tag": "Reoccupancy",
            "system": r.get("system", ""),
            "MAE": r.get("MAE", float("nan")),
            "P95_abs": r.get("P95_abs", float("nan")),
            "PASS": r.get("PASS", False),
        })
    for r in curve_func:
        if r.get("missing_in_python"):
            continue
        sys_rows.append({
            "model": model_name,
            "tag": "Functional",
            "system": r.get("system", ""),
            "MAE": r.get("MAE", float("nan")),
            "P95_abs": r.get("P95_abs", float("nan")),
            "PASS": r.get("PASS", False),
        })

    # Summary returned (model_number will be assigned in run_batch)
    summary = {
        "model_label": model_name,
        "func_mean_abs_diff_days": func_mean_abs_diff_days,   # thr=2 days
        "func_mean_rel_diff_pct": func_mean_rel_diff_pct,     # thr=4%
        "worst_system_MAE": worst_system_MAE,                 # thr=0.02
        "worst_system_P95_abs": worst_system_P95_abs,         # thr=0.04
        "mean_func_recovery_time_days": mean_func_recovery_time_days,
    }

    return summary, sys_rows


# -----------------------------
# Batch runner
# -----------------------------

def discover_models(parent_dir: str) -> list[str]:
    """
    Auto-detect model folders containing both outputs (with or without .json)
    """
    out = []
    for name in sorted(os.listdir(parent_dir)):
        p = os.path.join(parent_dir, name)
        if not os.path.isdir(p):
            continue
        py_path, mat_path = find_outputs(p)
        if py_path and mat_path:
            out.append(p)
    return out


def agg_system_table(rows: list[dict], tag: str) -> list[dict]:
    """
    Global system table across models for a given tag.
    Outputs ONLY:
      System, # Models, % PASS (Abs), Median MAE, Max MAE
    (Removed 95th %ile MAE per your request.)
    """
    rr = [r for r in rows if r.get("tag") == tag]
    systems = sorted(set(r["system"] for r in rr if r.get("system")))

    out = []
    for s in systems:
        rs = [r for r in rr if r.get("system") == s]
        n = len(rs)
        if n == 0:
            continue

        maes = np.array([r.get("MAE", np.nan) for r in rs], dtype=float)
        passes = np.array([bool(r.get("PASS", False)) for r in rs], dtype=bool)

        out.append({
            "System": s,
            "# Models": int(n),
            "% PASS (Abs)": float(100.0 * np.mean(passes)) if n > 0 else float("nan"),
            "Median MAE": float(np.nanmedian(maes)),
            "Max MAE": float(np.nanmax(maes)),
        })

    return out


def run_batch(parent_dir: str, model_names: list[str] | None = None, out_parent: str | None = None):
    if model_names:
        model_dirs = [os.path.join(parent_dir, m) for m in model_names]
    else:
        model_dirs = discover_models(parent_dir)

    global_model_rows = []
    global_system_rows = []

    for mdir in model_dirs:
        mname = os.path.basename(os.path.normpath(mdir))
        out_dir = os.path.join(out_parent, mname) if out_parent else mdir
        try:
            summary, sys_rows = compare_one_model(mdir, out_dir=out_dir)
            global_model_rows.append(summary)
            global_system_rows.extend(sys_rows)
            print(f"[OK] {mname}")
        except Exception as e:
            print(f"[FAIL] {mname}: {e}")

    # -------------------------
    # Write GLOBAL MODELS table 
    # -------------------------
    if global_model_rows:

        model_table = []
        for i, r in enumerate(global_model_rows, start=1):
            model_table.append({
                "model_number": i,
                "model_label": r.get("model_label", ""),
                "func_mean_abs_diff_days": r.get("func_mean_abs_diff_days", ""),
                "func_mean_rel_diff_pct": r.get("func_mean_rel_diff_pct", ""),
                "worst_system_MAE": r.get("worst_system_MAE", ""),
                "worst_system_P95_abs": r.get("worst_system_P95_abs", ""),
                "mean_func_recovery_time_days": r.get("mean_func_recovery_time_days", ""),
            })

        fieldnames = [
            "model_number",
            "model_label",
            "func_mean_abs_diff_days",
            "func_mean_rel_diff_pct",
            "worst_system_MAE",
            "worst_system_P95_abs",
            "mean_func_recovery_time_days",
        ]

        out_path = os.path.join(out_parent or parent_dir, "comparison_outputs_global_models.csv")
        write_csv(out_path, model_table, fieldnames=fieldnames)
        print(f"\nWrote global models table: {out_path}")

    # -------------------------
    # Write GLOBAL SYSTEMS tables (Functional + Reoccupancy)
    # -------------------------
    if global_system_rows:
        for tag in ["Reoccupancy", "Functional"]:
            table = agg_system_table(global_system_rows, tag)
            if table:
                out_path = os.path.join(out_parent or parent_dir, f"comparison_outputs_global_systems_{tag}.csv")
                write_csv(out_path, table, fieldnames=list(table[0].keys()))
                print(f"Wrote global systems table ({tag}): {out_path}")


if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = SCRIPT_DIR

    # MODELS = ["ICSB"]
    MODELS = None # will auto-discover all model folders that contain both JSONs

    run_batch(PARENT_DIR, model_names=MODELS, out_parent=None)
