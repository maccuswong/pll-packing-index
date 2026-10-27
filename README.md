# pll-packing-index

Image-analysis script used to quantify the **packing index** and **width** of the zebrafish posterior lateral line (pll) axon bundle from confocal maximum-intensity projections of anti-acetylated tubulin staining.

This code accompanies:

> Wong HJ, Matsui T, Bessho Y, Akiyama R. *Fgf3 and Fgf10a regulate neuronal fasciculation through Schwann cell proliferation and infiltration in zebrafish posterior lateral line.* bioRxiv (2026). doi:[10.64898/2026.04.05.716528](https://doi.org/10.64898/2026.04.05.716528)

## What it measures

The packing index describes how tightly axons are packed **within** the bundle, independently of how wide the bundle is. For each sampled position along the nerve it is the fraction of signal-positive pixels between the outermost axonal fibres, so wider gaps between axons give a lower value.

## How it works

1. **Load** a TIFF maximum-intensity projection (`load_image`). For a 3- or 4-channel image the brightest channel is used; for a small image stack the first plane is used.
2. **Locate the nerve** (`auto_roi`). After Otsu thresholding, the image row with the most signal-positive pixels is taken as the centre of the nerve, and a region extending 70 px above and below it (140 px tall) is cropped. A corner region (by default the lower-right part of the image) is excluded from this search so that bright off-target structures do not dominate.
3. **Isolate the bundle** (`tube_isolate_and_measure`):
   - Otsu thresholding of the crop gives the raw mask.
   - The local bundle width is measured column by column, smoothed with a 101-px median filter, and multiplied by a safety factor of 2.0 to give a local half-width.
   - The intensity-weighted centroid of each column defines a centre line, cleaned with a Hampel filter, a 41-px median filter and a Gaussian filter (sigma 10).
   - Only pixels lying within the local half-width of the centre line are kept, which removes signal from neighbouring structures.
4. **Exclude unreliable stretches** (`low_confidence_columns`). The longest contiguous run of signal in each column is smoothed with a 41-px median filter and compared with the 90th percentile of that same profile for that image. Columns falling below **40%** of that value carry too little signal to measure and are dropped.

   This gate is computed from the raw mask alone, deliberately not from the adaptive tube in step 3. Gating on the tube would feed back on itself: weak signal narrows the tube, the narrowed tube rejects more of the already weak signal, and a dim stretch then looks worse than it is.
5. **Measure** every 4 px along the anteroposterior axis: bundle width (distance between the outermost fibres), packing index (fraction of signal-positive pixels between them) and the coefficient of variation of intensity. Columns with fewer than 2 signal-positive pixels are skipped.
6. **Report** one row per image in `results/packing_density_results.csv`, plus two diagnostic images per input:
   - an **overlay** showing the pixels counted (green), real signal excluded by the tube (orange) and whole columns dropped by the confidence gate (blue);
   - a **coverage profile** in `results/Profile/`, plotting the per-column signal width, the gate floor, the stretches below it, and the centre-line position, so that a dip can be checked against the centre line drifting towards the edge of the region.

All distances are in **pixels**.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Put the TIFF projections in `data/` and run:

```bash
python run_packing_density_QC.py
```

Results are written to `results/`. To use other folders, call the function directly:

```python
from run_packing_density_QC import run_batch
run_batch("my_images", "my_results", pattern="*.tif")
```

### Input file naming

Metadata is read from the file name, with fields separated by underscores:

- **phenotype** — the first field
- **region** — the field before `-Proj`
- **channel** — a field such as `c2`

For example `WT_anterior-Proj_c2.tif` is read as phenotype `WT`, region `anterior`, channel `c2`.

### Output columns

| Column | Meaning |
| --- | --- |
| `file`, `phenotype`, `region`, `channel` | parsed from the file name |
| `threshold_used` | Otsu threshold applied to the crop |
| `packing_index` | mean packing index for the image |
| `width_px` | mean bundle width, in pixels |
| `cv` | mean coefficient of variation of intensity |
| `tube_halfwidth_mean_px`, `_min_px`, `_max_px` | local half-width used to isolate the bundle |
| `px_removed_pct` | percentage of thresholded pixels removed as off-target |
| `n_hampel_outliers` | number of columns corrected by the Hampel filter |
| `pct_columns_excluded_lowconf` | percentage of measured columns dropped by the confidence gate |

## Licence

MIT, see [LICENSE](LICENSE).
