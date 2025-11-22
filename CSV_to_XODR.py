#!/usr/bin/env python3
"""
9/30/2025
Script made by Charles Levine and his good friend, ChatGPT


track2xodr_single_lane.py — Build a single-lane OpenDRIVE (.xodr) from a CSV of X/Y[/Z] points.

- Converts feet->meters (default) or passes meters through if --units meters.
- Orders scattered points into a continuous path (NN + turn penalty), unless --no-order.
- 'endurance' closes the loop; 'autocross' keeps it open.
- Resamples at uniform spacing; optional moving-average smoothing.
- Writes a single right driving lane (id = -1). --lane-width is the TOTAL track width.
"""

import argparse
import math
import os
import numpy as np
import pandas as pd
from xml.etree.ElementTree import Element, SubElement, ElementTree

# ----------------------------- helpers -------------------------------- #

def order_points_continuous(pts_xy: np.ndarray, k_near: int = 80) -> np.ndarray:
    N = len(pts_xy)
    unused = set(range(N))
    start_idx = int(np.argmin(pts_xy[:, 0] + 1e-6 * pts_xy[:, 1]))
    order = [start_idx]; unused.remove(start_idx)

    def cost(prev_idx, cand_idx, prev_prev_idx=None):
        p = pts_xy[prev_idx]; c = pts_xy[cand_idx]
        d = float(np.hypot(*(c - p)))
        if prev_prev_idx is None:
            return d
        v1 = pts_xy[prev_idx] - pts_xy[prev_prev_idx]
        v2 = c - p
        n1 = np.hypot(*v1) + 1e-9
        n2 = np.hypot(*v2) + 1e-9
        cosang = float((v1 @ v2) / (n1 * n2))
        turn_pen = (1.0 - cosang) * 2.0
        return d * (1.0 + turn_pen)

    while unused:
        last = order[-1]
        prev = order[-2] if len(order) >= 2 else None
        cand_list = np.array(list(unused))
        dists = np.hypot(*(pts_xy[cand_list].T - pts_xy[last][:, None]))
        cand_list = cand_list[np.argsort(dists)[: min(k_near, len(cand_list))]]

        best = None; best_cost = 1e18
        for ci in cand_list:
            c = cost(last, int(ci), prev)
            if c < best_cost:
                best_cost = c; best = int(ci)
        order.append(best); unused.remove(best)

    return np.array(order, dtype=int)

def resample_polyline(x, y, z, spacing):
    dx = np.diff(x); dy = np.diff(y)
    seg_len = np.hypot(dx, dy)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(s[-1])
    if total < 1e-6:
        raise ValueError("Polyline length is ~0 after ordering/cleanup.")
    s_target = np.arange(0.0, total, spacing)
    if total - s_target[-1] > 0.3 * spacing:
        s_target = np.append(s_target, total)
    x_res = np.interp(s_target, s, x)
    y_res = np.interp(s_target, s, y)
    z_res = np.interp(s_target, s, z) if z is not None else np.zeros_like(x_res)
    return x_res, y_res, z_res, s_target

def _moving_average(arr: np.ndarray, window_m: float, spacing: float) -> np.ndarray:
    if window_m <= 0:
        return arr
    w = max(1, int(round(window_m / max(spacing, 1e-9))))
    if w <= 1: return arr
    if w % 2 == 0: w += 1
    pad = w // 2
    arr_pad = np.pad(arr, (pad, pad), mode="edge")
    ker = np.ones(w) / float(w)
    return np.convolve(arr_pad, ker, mode="valid")

def write_xodr(path: str, name: str,
               x: np.ndarray, y: np.ndarray, z: np.ndarray,
               lane_width_total: float = 3.5,
               track_type: str = "endurance"):
    """Single-lane OpenDRIVE: one right driving lane (id=-1), width = total track width."""
    dx = np.diff(x); dy = np.diff(y)
    L = np.hypot(dx, dy)
    mask = L > 1e-6
    x0 = x[:-1][mask]; y0 = y[:-1][mask]
    hdg = np.arctan2(dy[mask], dx[mask])
    s_vals = np.concatenate([[0.0], np.cumsum(L[mask][:-1])])
    total_len = float(np.sum(L[mask]))

    root = Element('OpenDRIVE')
    SubElement(root, 'header', {
        'revMajor': '1', 'revMinor': '4',
        'name': name, 'version': '1.00',
        'north': '0', 'south': '0', 'east': '0', 'west': '0'
    })
    road = SubElement(root, 'road', {
        'name': name, 'length': f'{total_len:.6f}', 'id': '1', 'junction': '-1'
    })

    # Road-level link (helps CarMaker generate routes)
    rlink = SubElement(road, 'link')
    if track_type == "endurance":
        SubElement(rlink, 'predecessor', {
            'elementType': 'road', 'elementId': '1', 'contactPoint': 'start'
        })
        SubElement(rlink, 'successor', {
            'elementType': 'road', 'elementId': '1', 'contactPoint': 'end'
        })
    # Autocross: leave open (no predecessor/successor)

    plan = SubElement(road, 'planView')
    for s, xi, yi, psi, Li in zip(s_vals, x0, y0, hdg, L[mask]):
        g = SubElement(plan, 'geometry', {
            's': f'{s:.6f}', 'x': f'{xi:.6f}', 'y': f'{yi:.6f}',
            'hdg': f'{psi:.10f}', 'length': f'{Li:.6f}'
        })
        SubElement(g, 'line')

    # elevation (flat at start z)
    elev = SubElement(road, 'elevationProfile')
    SubElement(elev, 'elevation', {
        's': '0.000000', 'a': f'{float(z[0]):.6f}', 'b': '0.0', 'c': '0.0', 'd': '0.0'
    })
    SubElement(road, 'lateralProfile')

    # Single-lane layout: center (id 0) + one right driving lane (id -1) with total width
    lanes = SubElement(road, 'lanes')
    ls = SubElement(lanes, 'laneSection', {'s': '0.000000'})
    SubElement(ls, 'left')  # no left lanes
    center = SubElement(ls, 'center'); right = SubElement(ls, 'right')

    SubElement(center, 'lane', {'id': '0', 'type': 'none', 'level': 'false'})
    lane_r = SubElement(right, 'lane', {'id': '-1', 'type': 'driving', 'level': 'false'})
    SubElement(lane_r, 'width', {
        'sOffset': '0.0',
        'a': f'{lane_width_total:.3f}', 'b': '0.0', 'c': '0.0', 'd': '0.0'
    })

    SubElement(road, 'objects'); SubElement(road, 'signals')
    ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)

# ----------------------------- main ----------------------------------- #

def main():
    ap = argparse.ArgumentParser(description="Convert CSV of X/Y[/Z] to single-lane OpenDRIVE (.xodr)")
    ap.add_argument("csv", help="Input CSV with columns for X,Y,[Z] (feet by default)")
    ap.add_argument("--xcol", default=None, help="Name of X column (default: auto-detect)")
    ap.add_argument("--ycol", default=None, help="Name of Y column (default: auto-detect)")
    ap.add_argument("--zcol", default=None, help="Name of Z column (optional)")
    ap.add_argument("--units", choices=["feet", "meters"], default="feet",
                    help="Units in the input CSV (default: feet)")
    ap.add_argument("--type", choices=["endurance", "autocross"], default="endurance",
                    help="Track type: 'endurance' = closed loop, 'autocross' = open")
    ap.add_argument("--spacing", type=float, default=1.0, help="Resample spacing in meters (default: 1.0)")
    ap.add_argument("--smooth-window", type=float, default=0.0,
                    help="Moving-average window in meters after resampling (0 = off)")
    ap.add_argument("--lane-width", type=float, default=3.5,
                    help="TOTAL track width in meters (single lane)")
    ap.add_argument("--name", default="Track_FROM_CSV", help="Name used inside the .xodr")
    ap.add_argument("--out", default="track_from_csv.xodr", help="Output .xodr filename")
    ap.add_argument("--preview", default=None, help="Optional: write ordered+resampled CSV here")
    ap.add_argument("--no-order", action="store_true",
                    help="Skip ordering (use rows as-is if already in path order)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    cols_lower = {c.strip().lower(): c for c in df.columns}
    x_col = args.xcol or cols_lower.get('x') or cols_lower.get('x_m') or list(df.columns)[0]
    y_col = args.ycol or cols_lower.get('y') or cols_lower.get('y_m') or list(df.columns)[1]
    z_col = args.zcol or cols_lower.get('z') or cols_lower.get('z_m')

    X = df[x_col].to_numpy(dtype=float)
    Y = df[y_col].to_numpy(dtype=float)
    Z = df[z_col].to_numpy(dtype=float) if (z_col and z_col in df.columns) else None

    if args.units == "feet":
        ft2m = 0.3048
        X *= ft2m; Y *= ft2m
        if Z is not None: Z *= ft2m

    valid = np.isfinite(X) & np.isfinite(Y)
    if Z is not None: valid &= np.isfinite(Z)
    X, Y = X[valid], Y[valid]
    if Z is not None: Z = Z[valid]

    pts = np.column_stack([X, Y])
    if len(pts) < 2:
        raise SystemExit("Not enough valid points.")
    dxy = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    keep = np.concatenate([[True], dxy > 1e-9])
    pts = pts[keep]
    if Z is not None: Z = Z[keep]

    if not args.no_order:
        idx = order_points_continuous(pts)
        pts = pts[idx]
        if Z is not None: Z = Z[idx]

    if args.type == "endurance":
        if math.hypot(*(pts[-1] - pts[0])) > 0.0:
            pts = np.vstack([pts, pts[0]])
            if Z is not None: Z = np.append(Z, Z[0])

    x_res, y_res, z_res, s_res = resample_polyline(pts[:, 0], pts[:, 1], Z, args.spacing)

    if args.smooth_window and args.smooth_window > 0.0:
        x_res = _moving_average(x_res, args.smooth_window, args.spacing)
        y_res = _moving_average(y_res, args.smooth_window, args.spacing)

    if args.preview:
        pd.DataFrame({"x_m": x_res, "y_m": y_res, "z_m": z_res}).to_csv(args.preview, index=False)

    write_xodr(args.out, args.name, x_res, y_res, z_res,
               lane_width_total=args.lane_width, track_type=args.type)

    abs_out = os.path.abspath(args.out)
    shape_txt = "closed loop" if args.type == "endurance" else "open track"
    print(f"Wrote {abs_out}  |  points: {len(x_res)}  |  length ≈ {s_res[-1]:.2f} m  |  {shape_txt}")

# ------------------ Thonny-friendly launcher ------------------ #
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        import tkinter as tk
        from tkinter.filedialog import askopenfilename
        root = tk.Tk(); root.withdraw()
        csv_path = askopenfilename(
            title="Select track CSV (X,Y[,Z])",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        root.update(); root.destroy()
        if not csv_path:
            print("No CSV selected. Exiting."); sys.exit(0)

        # Editable defaults for Thonny runs
        NAME       = "Track_FROM_CSV"
        OUT        = "track_from_csv.xodr"
        TYPE       = "autocross"          # "endurance" or "autocross"
        UNITS      = "meters"               # "feet" or "meters"
        SPACING    = "0.5"                # meters
        SMOOTHWIN  = "1.75"                # meters (use "0" to disable)
        LANE_WIDTH = "3.5"                # TOTAL track width (single lane)
        PREVIEW    = ""                   # e.g., "preview_ordered_m.csv"
        XCOL = ""; YCOL = ""; ZCOL = ""
        NO_ORDER   = True                 # set True if your CSV is already in path order

        argv = [sys.argv[0], csv_path,
                "--name", NAME,
                "--out", OUT,
                "--type", TYPE,
                "--units", UNITS,
                "--spacing", SPACING,
                "--smooth-window", SMOOTHWIN,
                "--lane-width", LANE_WIDTH]
        if PREVIEW.strip(): argv += ["--preview", PREVIEW]
        if XCOL.strip():    argv += ["--xcol", XCOL]
        if YCOL.strip():    argv += ["--ycol", YCOL]
        if ZCOL.strip():    argv += ["--zcol", ZCOL]
        if NO_ORDER:        argv += ["--no-order"]
        sys.argv = argv
        main()
