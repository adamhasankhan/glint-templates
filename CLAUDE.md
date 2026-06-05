# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A launch template for running **gravitational "glint" parameter estimation (PE)** on confirmed
binary black hole mergers from the GWOSC catalog, using [Bilby](https://lscsoft.docs.ligo.org/bilby/).
"Glint" / "echo" refers to a hypothesized lensing perturber that adds a delayed echo
(`echo_delta_t`, `echo_amp`) to the standard BBH waveform. The point of each run is to do PE
with these echo parameters either sampled or pinned to zero, and compare.

This is **not** a conventional software package — there is no build, no test suite, no
`requirements.txt`, no entry-point/module structure. `glint_pe.py` is a self-contained,
top-to-bottom analysis script driven by argparse. Treat it as a runnable scientific template.

## Hard dependency: the grav-glint Bilby branch

The echo parameters (`echo_delta_t`, `echo_amp`) only exist in a **custom Bilby fork**, not
upstream Bilby. You must install:
- https://github.com/adamhasankhan/bilby-adamhasankhan/tree/grav-glint  (current)
- https://gitlab.com/mattcarney106/bilby-mattcarney106  (original)

Other deps used at import: `gwpy`, `gwosc`, `h5py`, `astropy`, `numpy`. Running against stock
PyPI `bilby` will fail or silently ignore the echo priors.

## Running

```bash
python3 glint_pe.py \
    --npool 12 \
    --outdir ~/public_html/glint/GW150914 \
    --label GW150914-run1 \
    --event GW150914
```

Key flags (see `argparse` block near the top of the script for the full list):
- `--event` — GWOSC event name (e.g. `GW150914`, `GW231123_135430`). Drives data download,
  trigger time, and which GWTC release file is used.
- `--noglintrec` — `store_false`; **presence pins echoes to 0** (no-glint run). Default samples echoes.
- `--perturber_params` — sample physical perturber params
  (`b`, `f_P`, `M_P`) instead of waveform echo params.
- `--npool`, `--nlive`, `--nact`, `--maxmcmc` — dynesty sampler resources/hyperparameters.
- `--gwtc_dir` / `--gwtc_file` — where to find / override the GWTC PE data release HDF5.

Or use the launcher `run_glint_pe.sh`, which wraps a single invocation in `time` and tees stdout/
stderr to `glint_pe.log` / `time.log` — edit the arguments in it for the run you want.

There is no single-test command; a "run" *is* a long-running nested-sampling job (hours, with
`check_point_delta_t=3600` checkpointing). Outputs (result HDF5, corner plots incl.
`echo_params.pdf`, data/PSD plots, logs) all land in `--outdir`.

## The files

- `glint_pe.py` — the analysis program (the single source of truth for the science).
- `run_glint_pe.sh` — a thin bash launcher: one `python3 glint_pe.py ...` invocation with a
  concrete set of arguments, wrapped in `time` and redirecting output to `glint_pe.log` /
  `time.log`. Edit it to set the event/flags for a given run, or copy it per-run.

**Naming convention:** the program is `glint_pe.py`; launchers/wrappers share that base name with a
`run_` prefix (`run_glint_pe.sh`). Don't put a version number in the filename (no `template5`-style
counters — versioning lives in git), and don't give a shell launcher a `.py`-script's body.

## Execution flow (top to bottom)

1. Parse args, seed RNG, set up Bilby logger into `--outdir`.
2. `bilby.gw.utils.get_event_time(event)` → trigger time; analysis segment is `duration=12`s with
   `post_trigger_duration=6`s.
3. **`get_gwtc_filename`** resolves the GWTC PE data release HDF5:
   explicit `--gwtc_file` → local file in `--gwtc_dir` containing the event name → Zenodo download
   for O4 events (`GW23*`→GWTC-4, `GW24*`/`GW25*`→GWTC-5, record IDs hard-coded near the top) →
   canonical filename convention for O1–O3 (must be pre-downloaded), chosen by GPS-time thresholds.
4. **`get_analysis_key`** picks the HDF5 waveform group (prefers `IMRPhenomXPHM`+`SpinTaylor`).
5. **`get_prior_bounds`** derives chirp-mass and geocentric-time prior ranges from the published
   posterior samples (posterior range expanded 50% each side). This is how PE priors stay
   event-appropriate without manual tuning.
6. **`fetch_strain_data`** per detector (`H1`, `L1`, `V1`): queries GWOSC `locate` for
   deglitched/cleaned frames (CLEAN/CLN channels, `DCS-`/`GDS-` prefixes) and prefers them;
   otherwise falls back to `TimeSeries.fetch_open_data`. The big comment block of "Known glitched
   events" is **reference only** — glitch handling is automatic via this function.
7. Build `InterferometerList`, attaching downloaded strain + the **PSD read from the GWTC file**
   (not estimated locally). Detectors with no data / no PSD are skipped; zero detectors → hard error.
8. Build `BBHPriorDict`, add echo (or perturber) priors per `--noglintrec`/`--perturber_params`,
   set time prior to `geocent_time` (multi-detector) or `<det>_time` (single detector).
9. `IMRPhenomXPHM` waveform generator → `GravitationalWaveTransient` likelihood
   (distance-marginalized) → `run_sampler(sampler='dynesty', ...)` → corner plots.

## Gotchas when modifying

- **Zenodo record IDs and the GPS-time release thresholds in `get_gwtc_filename` are hard-coded.**
  New observing runs / catalog versions require updating these constants, not just CLI args.
- The script reads PSDs *from the GWTC HDF5*, so the chosen `analysis_key` must contain a `psds`
  group with the right detector keys, or detectors get dropped via the `KeyError` path.
- `set_start_method('fork')` is set at import — relevant on macOS / non-fork platforms.
- The waveform generator uses the **standard** `lal_binary_black_hole` source model; the echo
  parameters are consumed inside the grav-glint Bilby fork's conversion/likelihood, not here.
  A comment notes the perturber-parameter path still needs a waveform generator that supports
  perturber→waveform conversion.
