# DEVSIM RNPU Pipeline (Planar + Etched-Like Proxy)

This repository contains a CLI simulation script:

- `Scripts/DEVSIM_test.py`
- `Scripts/utils.py` (plotting, CSV/JSON I/O, stability-log parsing helpers)

It generates:

- IV curves (`voltage_V`, `current_A_per_cm`)
- slope curves (`d(log10|I|)/d(log10|V|)`)
- optional band-diagram exports
- optional stability metrics for multi-width runs

## 1. Environment Setup

## Prerequisites

- Python 3.10+
- A working DEVSIM installation/import in your Python environment

`DEVSIM_test.py` imports:

- `devsim`
- `devsim.python_packages.model_create`
- `devsim.python_packages.simple_physics`

If `import devsim` fails, install DEVSIM first (wheel/package/build) and ensure the same Python interpreter is used for runs.

## Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `pip install -r requirements.txt` cannot resolve `devsim` on your platform, install DEVSIM manually (wheel/build from the official DEVSIM distribution), then install the remaining packages:

```bash
pip install numpy matplotlib pandas
```

## Verify dependencies

```bash
python - << 'PY'
import numpy, matplotlib, pandas
print('numpy', numpy.__version__)
print('matplotlib', matplotlib.__version__)
print('pandas', pandas.__version__)
try:
    import devsim
    print('devsim OK')
except Exception as e:
    print('devsim import failed:', e)
PY
```

## 2. Output Folder Behavior

No local hardcoded paths are required.

- By default, script outputs are created next to `Scripts/DEVSIM_test.py`:
  - `Scripts/output`
  - `Scripts/figures`
- You can override these with:
  - `--output-dir <path>`
  - `--figures-dir <path>`

## 3. Silvaco-Oriented Structure in `DEVSIM_test.py`

The script is organized with Silvaco-style blocks:

- Process/geometry setup (2D mesh, regions, oxide/sidewall/metal stack proxies)
- Doping block (abrupt or Gaussian implant)
- Device physics initialization (Poisson, DD)
- Bias sweep and post-processing (IV, slope, optional band diagram)

## 4. Minimal Example Runs

## A) Planar-like quick run (fixed temperature)

Reasonable short sweep for fast verification:

```bash
python Scripts/DEVSIM_test.py \
  --prefix planar_quick_300K \
  --temp-k 300 \
  --geometry-mode silvaco_window \
  --metal-width-nm 800 \
  --contact-spacing-um 0.5 \
  --depth-um 2.0 \
  --doping-mode gaussian_implant \
  --implant-species boron \
  --implant-parameter-mode deck_by_species \
  --nd 1e15 \
  --na 1e14 \
  --vmin -0.001 \
  --vmax 0.1 \
  --sweep-mode silvaco_short \
  --contact-mode spec_ohmic \
  --save-band-diagram
```

## B) Etched-like preset run (fixed temperature)

`--preset etched_like` applies Silvaco-oriented etched proxy defaults (oxide + sidewall + Ti/Pd + geometry/contact defaults):

```bash
python Scripts/DEVSIM_test.py \
  --preset etched_like \
  --prefix etched_like_300K \
  --temp-k 300 \
  --metal-width-nm 800 \
  --nd 1e15 \
  --na 1e14 \
  --vmin -0.001 \
  --vmax 0.1 \
  --sweep-mode silvaco_short \
  --save-band-diagram
```

## C) Custom output directories

```bash
python Scripts/DEVSIM_test.py \
  --prefix custom_paths \
  --output-dir ./output \
  --figures-dir ./figures
```

## 5. Batch Example (width sweep)

```bash
python Scripts/DEVSIM_test.py \
  --prefix etched_width_sweep \
  --preset etched_like \
  --batch-metal-widths-nm 300 500 800 \
  --temp-k 300 \
  --nd 1e15 \
  --na 1e14 \
  --vmin -0.001 \
  --vmax 0.1 \
  --sweep-mode silvaco_short
```

## 6. Notes

- This is a DEVSIM proxy workflow; it is not a one-to-one replacement for full Silvaco process TCAD.
- Contact modes include:
  - `neutral`
  - `spec_ohmic` (Silvaco-aligned workfunction convention)
  - `schottky_approx` (barrier-based proxy)
- For reproducible runs, store command lines used in your report/log.

## 7) DEVSIM Algorithm Steps (Concise)

1. Parse CLI parameters (geometry, doping/implant, contacts, sweep, temperature, output paths).
2. Build 2D mesh and regions (silicon + optional oxide/sidewall/metal-stack proxy layers).
3. Create two contacts/electrodes (`left`, `right`) on silicon boundaries.
4. Set material/model parameters (temperature, lifetimes, optional oxide parameters).
5. Define doping (background `NA/ND` + optional Gaussian implant for boron/arsenic).
6. Solve Poisson-only initialization for a stable electrostatic starting point.
7. Enable drift-diffusion (electron/hole equations, optional trap block) and solve DC initialization.
8. Execute voltage sweep with continuation/backtracking/fallback retries for convergence.
9. Record current at each bias point and build IV data.
10. Compute slope curve `d(log10|I|)/d(log10|V|)` from IV points.
11. Export outputs (CSV + PNG; optional band diagram, manifest, stability metrics).
12. In batch/stability modes, repeat over parameter grids and generate combined summaries/plots.
