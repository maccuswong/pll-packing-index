import glob
import os
import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from scipy import ndimage


def load_image(path):
    img = tiff.imread(path)
    if img.ndim == 3:
        if img.shape[-1] in (3, 4):
            channel_means = img[..., :3].astype(float).mean(axis=(0, 1))
            img = img[..., int(np.argmax(channel_means))]
        elif img.shape[0] <= 8:
            img = img[0]
        else:
            img = img[..., 0]
    return img.astype(float)


def parse_filename(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    parts = stem.split("_")
    phenotype = parts[0] if parts else ""
    region = ""
    channel = ""
    for p in parts:
        if "-Proj" in p:
            region = p.split("-Proj")[0]
        if p.lower().startswith("c") and p[1:].isdigit():
            channel = p
    return phenotype, region, channel


def hampel_filter(trace, window_size=251, n_sigmas=2.0):
    trace = np.asarray(trace, dtype=float)
    n = len(trace)
    half = window_size // 2
    rolling_median = np.array([np.median(trace[max(0, i - half):min(n, i + half + 1)]) for i in range(n)])
    abs_dev = np.abs(trace - rolling_median)
    rolling_mad = np.array([np.median(abs_dev[max(0, i - half):min(n, i + half + 1)]) for i in range(n)])
    mad_threshold = np.maximum(n_sigmas * 1.4826 * rolling_mad, 1.0)
    is_outlier = abs_dev > mad_threshold
    good = ~is_outlier
    if good.sum() < 2:
        return trace, is_outlier
    xs = np.arange(n)
    cleaned = trace.copy()
    cleaned[is_outlier] = np.interp(xs[is_outlier], xs[good], trace[good])
    return cleaned, is_outlier


def robust_median_run_width(raw_mask):
    widths = []
    for xi in range(raw_mask.shape[1]):
        idx = np.where(raw_mask[:, xi])[0]
        if len(idx) == 0:
            continue
        splits = np.where(np.diff(idx) > 1)[0] + 1
        runs = np.split(idx, splits)
        widths.append(len(max(runs, key=len)))
    return float(np.median(widths)) if widths else np.nan


def per_column_run_width(raw_mask):
    n_cols = raw_mask.shape[1]
    widths = np.full(n_cols, np.nan)
    for xi in range(n_cols):
        idx = np.where(raw_mask[:, xi])[0]
        if len(idx) == 0:
            continue
        splits = np.where(np.diff(idx) > 1)[0] + 1
        runs = np.split(idx, splits)
        widths[xi] = len(max(runs, key=len))
    return widths


def local_tube_halfwidth(raw_mask, width_safety_factor, width_window=101):
    col_w = per_column_run_width(raw_mask)
    xs_all = np.arange(len(col_w))
    valid = ~np.isnan(col_w)
    if valid.sum() < 2:
        return np.full(len(col_w), np.nanmedian(col_w) * width_safety_factor if valid.any() else 1.0)
    col_w_interp = np.interp(xs_all, xs_all[valid], col_w[valid])
    col_w_smoothed = ndimage.median_filter(col_w_interp, size=width_window, mode='nearest')
    return col_w_smoothed * width_safety_factor


def _coverage_profile(raw_mask, coverage_frac, coverage_window):
    """Shared computation behind low_confidence_columns() and the coverage
    profile plot in save_coverage_profile() — kept in one place so both
    always agree on the same numbers instead of recomputing separately."""
    col_w = per_column_run_width(raw_mask)
    xs_all = np.arange(len(col_w))
    valid = ~np.isnan(col_w)
    if valid.sum() < 2:
        z = np.zeros(len(col_w))
        return dict(col_w_raw=z, col_w_smoothed=z, coverage_floor=0.0,
                    low_conf_col=np.zeros(len(col_w), dtype=bool))
    col_w_interp = np.interp(xs_all, xs_all[valid], col_w[valid])
    col_w_smoothed = ndimage.median_filter(col_w_interp, size=coverage_window, mode='nearest')
    peak_ref = np.percentile(col_w_smoothed, 90)
    coverage_floor = coverage_frac * peak_ref
    low_conf_col = col_w_smoothed < coverage_floor
    return dict(col_w_raw=col_w_interp, col_w_smoothed=col_w_smoothed,
                coverage_floor=coverage_floor, low_conf_col=low_conf_col)


def low_confidence_columns(raw_mask, coverage_frac=0.4, coverage_window=41):
    """Flag columns whose local raw-signal coverage (longest contiguous run
    per column) is too weak, relative to that image's own best-observed
    (90th percentile) coverage, to trust for quantification.

    This is deliberately computed from `raw_mask` alone, independent of the
    adaptive tube (local_tube_halfwidth) that is derived from the same
    per-column run width. Gating on the tube itself would inherit its
    feedback loop: weak signal narrows the tube, which then rejects more of
    the already-weak signal, making a dim region look artificially even
    worse. Gating on raw coverage first avoids that.
    """
    return _coverage_profile(raw_mask, coverage_frac, coverage_window)["low_conf_col"]


def auto_roi(img, margin=70, exclude_frac=(0.78, 1.0, 0.73, 1.0)):
    H, W = img.shape
    r0f, r1f, c0f, c1f = exclude_frac
    mask = np.ones(img.shape, dtype=bool)
    mask[int(H * r0f):int(H * r1f), int(W * c0f):int(W * c1f)] = False
    thr = threshold_otsu(img[mask])
    coverage = ((img > thr) & mask).sum(axis=1)
    peak = int(np.argmax(coverage))
    y0 = max(0, peak - margin)
    y1 = min(H, peak + margin)
    return y0, y1, 0, W


def tube_isolate_and_measure(image, roi, threshold=None, width_safety_factor=2.0,
                              width_window=101, close_size=0,
                              centroid_median_size=41, centerline_sigma=10,
                              hampel_window=251, hampel_n_sigmas=0,
                              step=4, min_width=2,
                              coverage_gate_frac=0.4, coverage_gate_window=41):
    y0, y1, x0, x1 = roi
    crop = image[y0:y1, x0:x1]
    thr = threshold if threshold is not None else threshold_otsu(crop)
    raw_thresh = crop > thr
    raw = ndimage.binary_closing(raw_thresh, structure=np.ones((close_size, close_size))) if close_size else raw_thresh

    tube_halfwidth = local_tube_halfwidth(raw, width_safety_factor, width_window)
    coverage_profile = _coverage_profile(raw, coverage_gate_frac, coverage_gate_window)
    low_conf_col = coverage_profile["low_conf_col"]

    yy = np.arange(crop.shape[0])[:, None]
    weights = crop * raw
    col_sum = weights.sum(axis=0)
    raw_centroid = np.where(col_sum > 0, (yy * weights).sum(axis=0) / np.maximum(col_sum, 1e-9), np.nan)

    xs_all = np.arange(crop.shape[1])
    valid = ~np.isnan(raw_centroid)
    centroid_interp = np.interp(xs_all, xs_all[valid], raw_centroid[valid])
    centroid_interp, outliers = hampel_filter(centroid_interp, hampel_window, hampel_n_sigmas)
    centroid_smoothed = ndimage.median_filter(centroid_interp, size=centroid_median_size, mode='nearest')
    centerline = ndimage.gaussian_filter1d(centroid_smoothed, sigma=centerline_sigma)

    tube = np.abs(yy - centerline[None, :]) <= tube_halfwidth[None, :]
    clean_mask = raw & tube

    widths, packings, cvs = [], [], []
    n_cols_total, n_cols_lowconf = 0, 0
    for xi in range(0, clean_mask.shape[1], step):
        col_mask = clean_mask[:, xi]
        idx = np.where(col_mask)[0]
        if len(idx) < min_width:
            continue
        n_cols_total += 1
        if low_conf_col[xi]:
            n_cols_lowconf += 1
            continue
        lo, hi = idx.min(), idx.max()
        local_mask = col_mask[lo:hi + 1]
        local_intensity = crop[lo:hi + 1, xi]
        widths.append(hi - lo + 1)
        packings.append(float(local_mask.mean()))
        cvs.append(float(local_intensity.std() / local_intensity.mean()) if local_intensity.mean() > 0 else np.nan)

    px_removed = int((raw & ~clean_mask).sum())
    return {
        "threshold_used": thr,
        "mean_packing_index": float(np.mean(packings)) if packings else np.nan,
        "mean_width_px": float(np.mean(widths)) if widths else np.nan,
        "mean_cv": float(np.mean(cvs)) if cvs else np.nan,
        "tube_halfwidth_mean_px": float(np.mean(tube_halfwidth)),
        "tube_halfwidth_min_px": float(np.min(tube_halfwidth)),
        "tube_halfwidth_max_px": float(np.max(tube_halfwidth)),
        "px_removed_pct": float(100 * px_removed / raw.sum()) if raw.sum() > 0 else np.nan,
        "n_hampel_outlier_columns": int(outliers.sum()),
        "pct_columns_excluded_lowconf": float(100 * n_cols_lowconf / n_cols_total) if n_cols_total else np.nan,
        "centerline": centerline,
        "crop": crop,
        "raw": raw,
        "clean_mask": clean_mask,
        "low_conf_col": low_conf_col,
        "col_w_raw": coverage_profile["col_w_raw"],
        "col_w_smoothed": coverage_profile["col_w_smoothed"],
        "coverage_floor": coverage_profile["coverage_floor"],
    }


def save_overlay(res, out_path):
    crop = res["crop"]
    raw = res["raw"]
    clean_mask = res["clean_mask"]
    low_conf_col = res["low_conf_col"]
    removed = raw & ~clean_mask
    confident_map = np.broadcast_to(~low_conf_col[None, :], crop.shape)
    overlay = np.stack([crop, crop, crop], axis=-1)
    overlay = overlay / max(overlay.max(), 1)
    overlay = np.clip(overlay * 1.6, 0, 1)
    overlay[clean_mask & confident_map] = [0, 1, 0]        # green: counted as measured signal
    overlay[removed & confident_map] = [1, 0.5, 0]          # orange: real signal, excluded by the tube
    overlay[raw & ~confident_map] = [0.2, 0.5, 1.0]         # blue: whole column excluded, low-confidence coverage
    fig, ax = plt.subplots(figsize=(20, 4))
    ax.imshow(overlay, aspect="auto")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def save_coverage_profile(res, out_path, title=""):
    """Diagnostic companion to save_overlay(): plots the per-column raw
    signal coverage (the same curve the adaptive tube and the QC gate are
    both derived from), the low-confidence gate floor, which stretches fall
    below it, and the centerline row-position (to confirm a dip isn't
    caused by the centerline drifting toward the ROI edge)."""
    col_w_raw = res["col_w_raw"]
    col_w_smoothed = res["col_w_smoothed"]
    coverage_floor = res["coverage_floor"]
    low_conf_col = res["low_conf_col"]
    centerline = res["centerline"]
    pct_excluded = res["pct_columns_excluded_lowconf"]
    xs = np.arange(len(col_w_smoothed))

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.plot(col_w_raw, alpha=0.3, label="raw per-column run width")
    ax.plot(col_w_smoothed, label="smoothed", color="tab:orange")
    ax.fill_between(xs, 0, max(col_w_smoothed.max(), coverage_floor) * 1.1, where=low_conf_col,
                     color="tab:blue", alpha=0.15, label="low-confidence (gated out)")
    ax.axhline(coverage_floor, color="k", ls="--", linewidth=1, label=f"gate floor ({coverage_floor:.1f}px)")
    ax2 = ax.twinx()
    ax2.plot(centerline, color="red", alpha=0.4, linewidth=1, label="centerline (row pos.)")
    ax2.set_ylabel("centerline row", color="red")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylabel("per-column run width (px)")
    ax.set_title(f"{title}  |  {pct_excluded:.1f}% columns excluded as low-confidence", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


def run_batch(image_dir, out_dir, pattern="*.tif", threshold=None, roi_margin=70):
    os.makedirs(out_dir, exist_ok=True)
    profile_dir = os.path.join(out_dir, "Profile")
    os.makedirs(profile_dir, exist_ok=True)
    rows = []
    for path in sorted(glob.glob(os.path.join(image_dir, pattern))):
        img = load_image(path)
        roi = auto_roi(img, roi_margin)
        res = tube_isolate_and_measure(img, roi, threshold=threshold)
        phenotype, region, channel = parse_filename(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        out_png = os.path.join(out_dir, stem + ".png")
        save_overlay(res, out_png)
        profile_png = os.path.join(profile_dir, stem + "_colw_profile.png")
        save_coverage_profile(res, profile_png, title=stem)
        rows.append({
            "file": os.path.basename(path),
            "phenotype": phenotype,
            "region": region,
            "channel": channel,
            "threshold_used": res["threshold_used"],
            "packing_index": res["mean_packing_index"],
            "width_px": res["mean_width_px"],
            "cv": res["mean_cv"],
            "tube_halfwidth_mean_px": res["tube_halfwidth_mean_px"],
            "tube_halfwidth_min_px": res["tube_halfwidth_min_px"],
            "tube_halfwidth_max_px": res["tube_halfwidth_max_px"],
            "px_removed_pct": res["px_removed_pct"],
            "n_hampel_outliers": res["n_hampel_outlier_columns"],
            "pct_columns_excluded_lowconf": res["pct_columns_excluded_lowconf"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "packing_density_results.csv"), index=False)
    return df


if __name__ == "__main__":
    run_batch("data", "results")
