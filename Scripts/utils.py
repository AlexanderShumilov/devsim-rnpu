#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit("matplotlib is required: pip install matplotlib") from exc


def bias_token(v: float) -> str:
    return f"{v:.6g}".replace("-", "m").replace("+", "p").replace(".", "p")


def figure_tag(run_number: int) -> str:
    return f"{datetime.now().strftime('%B%d')}_run{run_number}"


def write_csv(path: Path, header: list[str], rows: list[tuple]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def plot_iv(path: Path, iv: list[tuple[float, float]], xmin: float, xmax: float) -> None:
    arr = np.array(iv, dtype=float)
    v = arr[:, 0]
    i = arr[:, 1]
    vis = (v >= xmin) & (v <= xmax)
    v = v[vis]
    i = i[vis]
    m = float(np.max(np.abs(i))) if i.size else 1.0
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(v, i, color="#1f77b4", lw=2.0)
    plt.axhline(0.0, color="k", lw=0.8)
    plt.grid(True, alpha=0.25)
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [A/cm]")
    plt.title("IV characteristic")
    plt.xlim(xmin, xmax)
    plt.ylim(-1.1 * m, 1.1 * m)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_slope(path: Path, rows: list[tuple[float, float, str]], xmin: float, xmax: float) -> None:
    if not rows:
        return
    arr = np.array([(x, y) for x, y, _ in rows], dtype=float)
    v = arr[:, 0]
    s = arr[:, 1]
    vis = (v >= xmin) & (v <= xmax)
    v = v[vis]
    s = s[vis]
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(v, s, color="#d62728", lw=2.0)
    plt.grid(True, alpha=0.25)
    plt.xlabel("Voltage [V]")
    plt.ylabel("d(log10|I|) / d(log10|V|)")
    plt.title("Slope plot")
    plt.xlim(xmin, xmax)
    if s.size:
        lo, hi = np.nanmin(s), np.nanmax(s)
        if np.isfinite(lo) and np.isfinite(hi):
            pad = 0.15 * max(1e-6, hi - lo)
            plt.ylim(lo - pad, hi + pad)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_combined_iv(path: Path, cases: list[tuple[float, list[tuple[float, float]]]], xmin: float, xmax: float) -> None:
    plt.figure(figsize=(8.0, 5.0))
    for width_nm, iv in cases:
        arr = np.array(iv, dtype=float)
        v = arr[:, 0]
        i = arr[:, 1]
        vis = (v >= xmin) & (v <= xmax)
        plt.plot(v[vis], i[vis], lw=1.8, label=f"{width_nm:g} nm")
    plt.axhline(0.0, color="k", lw=0.8)
    plt.grid(True, alpha=0.25)
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [A/cm]")
    plt.title("IV comparison")
    plt.xlim(xmin, xmax)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_combined_slope(path: Path, cases: list[tuple[float, list[tuple[float, float, str]]]], xmin: float, xmax: float) -> None:
    plt.figure(figsize=(8.0, 5.0))
    for width_nm, rows in cases:
        if not rows:
            continue
        arr = np.array([(x, y) for x, y, _ in rows], dtype=float)
        v = arr[:, 0]
        s = arr[:, 1]
        vis = (v >= xmin) & (v <= xmax)
        plt.plot(v[vis], s[vis], lw=1.8, label=f"{width_nm:g} nm")
    plt.grid(True, alpha=0.25)
    plt.xlabel("Voltage [V]")
    plt.ylabel("d(log10|I|) / d(log10|V|)")
    plt.title("Slope comparison")
    plt.xlim(xmin, xmax)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_combined_iv_labeled(path: Path, cases: list[tuple[str, list[tuple[float, float]]]], xmin: float, xmax: float) -> None:
    plt.figure(figsize=(8.2, 5.2))
    for label, iv in cases:
        arr = np.array(iv, dtype=float)
        v = arr[:, 0]
        i = arr[:, 1]
        vis = (v >= xmin) & (v <= xmax)
        plt.plot(v[vis], i[vis], lw=1.8, label=str(label))
    plt.axhline(0.0, color="k", lw=0.8)
    plt.grid(True, alpha=0.25)
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [A/cm]")
    plt.title("IV comparison")
    plt.xlim(xmin, xmax)
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_combined_slope_labeled(path: Path, cases: list[tuple[str, list[tuple[float, float, str]]]], xmin: float, xmax: float) -> None:
    plt.figure(figsize=(8.2, 5.2))
    for label, rows in cases:
        if not rows:
            continue
        arr = np.array([(x, y) for x, y, _ in rows], dtype=float)
        v = arr[:, 0]
        s = arr[:, 1]
        vis = (v >= xmin) & (v <= xmax)
        plt.plot(v[vis], s[vis], lw=1.8, label=str(label))
    plt.grid(True, alpha=0.25)
    plt.xlabel("Voltage [V]")
    plt.ylabel("d(log10|I|) / d(log10|V|)")
    plt.title("Slope comparison")
    plt.xlim(xmin, xmax)
    plt.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def parse_elapsed_seconds(raw: str) -> float:
    raw = raw.strip()
    if not raw:
        return 0.0
    if "(" in raw:
        raw = raw.split("(", 1)[0].strip()
    if ":" in raw:
        parts = raw.split(":")
        try:
            if len(parts) == 3:
                h = float(parts[0])
                m = float(parts[1])
                s = float(parts[2])
                return 3600.0 * h + 60.0 * m + s
            if len(parts) == 2:
                m = float(parts[0])
                s = float(parts[1])
                return 60.0 * m + s
        except Exception:
            return 0.0
    try:
        return float(raw)
    except Exception:
        return 0.0


def parse_stability_log(stdout_text: str) -> tuple[list[dict[str, float | int | str]], dict[float, list[tuple[int, float]]]]:
    points: list[dict[str, float | int | str]] = []
    curves: dict[float, list[tuple[int, float]]] = {}

    rel_abs_re = re.compile(r"RelError:\s*([0-9eE+\-.]+)\s*\tAbsError:\s*([0-9eE+\-.]+)")
    target_v: float | None = None
    iter_index: int | None = None
    rel_start: float | None = None
    rel_end: float | None = None
    abs_start: float | None = None
    abs_end: float | None = None
    fallback_count = 0
    current_curve: list[tuple[int, float]] = []
    current_v: float | None = None

    def flush_point(converged: int) -> None:
        nonlocal target_v, iter_index, rel_start, rel_end, abs_start, abs_end, fallback_count, current_curve, current_v
        if target_v is None:
            return
        points.append(
            {
                "voltage_V": float(target_v),
                "converged": int(converged),
                "newton_iterations": int(iter_index + 1) if iter_index is not None else 0,
                "rel_error_start": float(rel_start) if rel_start is not None else np.nan,
                "rel_error_end": float(rel_end) if rel_end is not None else np.nan,
                "abs_error_start": float(abs_start) if abs_start is not None else np.nan,
                "abs_error_end": float(abs_end) if abs_end is not None else np.nan,
                "fallback_retries": int(fallback_count),
            }
        )
        if current_v is not None and current_curve:
            curves[current_v] = current_curve.copy()
        target_v = None
        iter_index = None
        rel_start = None
        rel_end = None
        abs_start = None
        abs_end = None
        fallback_count = 0
        current_curve = []
        current_v = None

    for line in stdout_text.splitlines():
        line = line.strip()
        if line.startswith("BIAS_TARGET "):
            flush_point(converged=0)
            try:
                target_v = float(line.split()[1])
            except Exception:
                target_v = None
            current_v = target_v
            continue
        if line.startswith("FALLBACK_RETRY "):
            fallback_count += 1
            continue
        if line.startswith("Iteration:"):
            try:
                iter_index = int(line.split(":", 1)[1].strip())
            except Exception:
                iter_index = None
            continue
        if line.startswith('Device:') and current_v is not None:
            m = rel_abs_re.search(line)
            if m:
                rel = float(m.group(1))
                ab = float(m.group(2))
                if rel_start is None:
                    rel_start = rel
                rel_end = rel
                if abs_start is None:
                    abs_start = ab
                abs_end = ab
                if iter_index is not None:
                    current_curve.append((iter_index, rel))
            continue
        if line.startswith("BIAS_DONE "):
            flush_point(converged=1)

    flush_point(converged=0)
    return points, curves


def parse_time_metrics(stderr_text: str) -> tuple[float, float]:
    elapsed_s = 0.0
    peak_rss_kb = 0.0
    for line in stderr_text.splitlines():
        line = line.strip()
        if line.startswith("Elapsed (wall clock) time"):
            _, val = line.split(":", 1)
            elapsed_s = parse_elapsed_seconds(val.strip())
        elif line.startswith("Maximum resident set size (kbytes)"):
            _, val = line.split(":", 1)
            try:
                peak_rss_kb = float(val.strip())
            except Exception:
                peak_rss_kb = 0.0
    return elapsed_s, peak_rss_kb
