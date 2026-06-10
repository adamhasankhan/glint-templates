#!/usr/bin/env python
"""
Generate a bilby_pipe .ini (plus companion prior + PSD files) that reproduces the
gravitational-glint parameter-estimation run defined by gradar_template5.py.

gradar_template5.py builds and launches the run programmatically (custom data
fetch, GWTC-release PSDs, posterior-derived priors, glint echo parameters,
dynesty).  This script extracts the *same* settings and writes them out as a
bilby_pipe configuration so the identical analysis can be driven through the
bilby_pipe workflow (HTCondor or --local) instead.

What is reproduced faithfully:
  * 12 s analysis segment, 6 s post-trigger, 4096 Hz, f_low = f_ref = 20 Hz
  * Detectors = those with a PSD in the GWTC release for the chosen waveform
  * PSDs taken directly from the GWTC PE data-release HDF5 (not estimated)
  * chirp-mass and geocentric-time priors derived from the GWTC posteriors
    (+/-50% of the posterior range; time +/-0.2 s) -- via get_prior_bounds()
  * IMRPhenomXPHM, lal_binary_black_hole source model
  * Glint echo priors echo_delta_t ~ U(0.1, 1), echo_amp ~ U(0, 1)
    (or both pinned to 0.0 with --noglintrec)
  * GravitationalWaveTransient likelihood, distance marginalisation ON,
    phase/time marginalisation OFF
  * dynesty: nlive=2048, nact=20, maxmcmc=8192, dlogz=0.1, sample='act-walk'

Known, deliberate differences (bilby_pipe limitations -- see comments inline):
  * Deglitched/cleaned-frame auto-selection is not reproduced; data are pulled as
    GWOSC open data via channel-dict={det:GWOSC}.  Override channel-dict if you
    have cleaned frames.
  * luminosity_distance uses cosmology='Planck15' (H0=67.74, Om0=0.3075) rather
    than the template's custom LambdaCDM(H0=67.9, Om0=0.3065); the distance prior
    difference is sub-percent and negligible.  Edit the prior file if you need it
    bit-for-bit.

Requires the same environment as gradar_template5.py (bilby grav-glint branch,
gwpy, h5py).  The custom branch must be installed wherever bilby_pipe runs so
that frequency-domain-source-model=lal_binary_black_hole understands the echo
parameters.

Sample launch:
    python3 make_bilby_pipe_ini.py --event GW241225_082815 \\
        --outdir ~/public_html/glint/pe/GW241225_082815/bilby_pipe_glint
    bilby_pipe <outdir>/<label>.ini            # then submit, or add --local

The helper functions below are copied (not imported) from gradar_template5.py on
purpose: importing that module would execute its top-level run.
"""
import argparse
import json
import os
import sys
import urllib.request

import h5py
import numpy as np


def log(msg):
    print(msg, file=sys.stderr, flush=True)


# --- GWTC file resolution (mirrors gradar_template5.py) -----------------------

_GWTC4_ZENODO_ID = '17602505'                  # https://doi.org/10.5281/zenodo.17602505
_GWTC5_ZENODO_IDS = ['20276106', '20291740']   # https://doi.org/10.5281/zenodo.20276106 and .20291740


def _zenodo_find_and_download(event, record_ids, download_dir):
    """Find/download an HDF5 from Zenodo whose name contains the event string."""
    os.makedirs(download_dir, exist_ok=True)
    for record_id in record_ids:
        files_url = f'https://zenodo.org/api/records/{record_id}/files'
        try:
            req = urllib.request.Request(files_url, headers={'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            for entry in data.get('entries', []):
                key = entry.get('key', '')
                if event in key and key.endswith(('.hdf5', '.h5')):
                    local_path = os.path.join(download_dir, key)
                    if os.path.exists(local_path):
                        log(f"Using cached file: {local_path}")
                        return local_path
                    content_url = entry['links']['content']
                    log(f"Downloading {key} from Zenodo record {record_id}...")
                    urllib.request.urlretrieve(content_url, local_path)
                    log(f"Downloaded to {local_path}")
                    return local_path
        except Exception as e:
            log(f"Zenodo query for record {record_id} failed: {e}")
    return None


def get_gwtc_filename(event, trigger_time, gwtc_dir='.', gwtc_file_override=None):
    """Resolve the GWTC PE data-release HDF5 filename for a given event."""
    if gwtc_file_override:
        return gwtc_file_override

    try:
        for entry in os.scandir(gwtc_dir):
            if entry.is_file() and entry.name.endswith(('.hdf5', '.h5')) and event in entry.name:
                log(f"Found HDF5 file on disk: {entry.path}")
                return entry.path
    except OSError:
        pass

    if event.startswith('GW23'):
        record_ids = [_GWTC4_ZENODO_ID]
    elif event.startswith(('GW24', 'GW25')):
        record_ids = _GWTC5_ZENODO_IDS + [_GWTC4_ZENODO_ID]
    else:
        record_ids = []

    if record_ids:
        path = _zenodo_find_and_download(event, record_ids, gwtc_dir)
        if path:
            return path
        log(f"Zenodo download failed for {event}. Pass --gwtc_file to specify the file manually.")

    if trigger_time > 1256655618:      # O3b -> GWTC-3
        fallback = f'IGWN-GWTC3p0-v1-{event}_PEDataRelease_mixed_cosmo.h5'
    elif trigger_time > 1238112018:    # O3a -> GWTC-2.1
        fallback = f'IGWN-GWTC2p1-v2-{event}_PEDataRelease_mixed_cosmo.h5'
    else:                              # O1/O2 -> GWTC-1
        fallback = f'GWTC-1_sample_release/{event}_GWTC-1.hdf5'

    return os.path.join(gwtc_dir, fallback)


def get_analysis_key(gwtc_file_handle):
    """Return the HDF5 group key for the preferred waveform analysis."""
    keys = list(gwtc_file_handle.keys())
    for k in keys:
        if 'IMRPhenomXPHM' in k and 'SpinTaylor' in k:
            return k
    for k in keys:
        if 'IMRPhenom' in k or 'SEOBNR' in k:
            return k
    log(f"No standard waveform key found in HDF5. Available: {keys}. Using first key.")
    return keys[0]


def get_prior_bounds(gwtc_file_handle, analysis_key):
    """Derive chirp-mass and geocentric-time bounds from the GWTC posteriors."""
    samples = gwtc_file_handle[analysis_key]['posterior_samples']

    chirp_masses = samples['chirp_mass'][:]
    cm_range = chirp_masses.max() - chirp_masses.min()
    buffer_cm = max(0.5 * cm_range, 1.0)  # at least 1 M_sun buffer
    min_chirp = max(1.0, float(chirp_masses.min()) - buffer_cm)
    max_chirp = float(chirp_masses.max()) + buffer_cm

    geo_key = 'geocent_time'
    if geo_key not in samples.dtype.names:
        for candidate in ('H1_time', 'L1_time', 'V1_time'):
            if candidate in samples.dtype.names:
                geo_key = candidate
                break
    geo_times = samples[geo_key][:]
    min_geo = float(geo_times.min()) - 0.2
    max_geo = float(geo_times.max()) + 0.2

    log(f"Prior bounds from posteriors -- chirp mass: [{min_chirp:.4f}, {max_chirp:.4f}] M_sun, "
        f"time: [{min_geo:.3f}, {max_geo:.3f}] s")
    return min_chirp, max_chirp, min_geo, max_geo


def get_detectors_and_psds(gwtc_file_handle, analysis_key, outdir):
    """Write GWTC-release PSDs to .dat files; return (detectors, psd_dict)."""
    psds = gwtc_file_handle[analysis_key]['psds']
    detectors, psd_dict = [], {}
    for det in ('H1', 'L1', 'V1'):
        if det not in psds.keys():
            continue
        arr = psds[det][:]  # columns: frequency, PSD
        psd_path = os.path.abspath(os.path.join(outdir, f'{det}_psd.dat'))
        np.savetxt(psd_path, arr, header='frequency PSD')
        detectors.append(det)
        psd_dict[det] = psd_path
        log(f"Wrote {det} PSD ({arr.shape[0]} bins) -> {psd_path}")
    return detectors, psd_dict


# --- prior file ---------------------------------------------------------------

def write_prior_file(path, min_chirp, max_chirp, min_geo, max_geo, glint_rec):
    """Write a bilby prior file matching gradar_template5.py's priors."""
    chirp_line = (
        "chirp_mass = bilby.gw.prior.UniformInComponentsChirpMass("
        f"minimum={min_chirp}, maximum={max_chirp}, "
        r"name='chirp_mass', latex_label='$\mathcal{M}$', unit='$M_{\odot}$', boundary=None)"
    )
    geo_line = (
        "geocent_time = Uniform("
        f"minimum={min_geo}, maximum={max_geo}, "
        r"name='geocent_time', latex_label='$t_c$', unit='$s$', boundary=None)"
    )
    # Template used a custom LambdaCDM(H0=67.9, Om0=0.3065); Planck15 differs
    # negligibly and is resolvable inside a bilby prior file.
    dist_line = (
        "luminosity_distance = bilby.gw.prior.UniformSourceFrame("
        "minimum=10.0, maximum=10000.0, cosmology='Planck15', "
        r"name='luminosity_distance', latex_label='$d_L$', unit='Mpc', boundary=None)"
    )

    fixed = r"""mass_ratio = bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.05, maximum=1.0, name='mass_ratio', latex_label='$q$', unit=None, boundary=None)
mass_1 = bilby.gw.prior.Constraint(minimum=1, maximum=1000, name='mass_1', latex_label='$m_1$', unit=None)
mass_2 = bilby.gw.prior.Constraint(minimum=1, maximum=1000, name='mass_2', latex_label='$m_2$', unit=None)
a_1 = Uniform(minimum=0.0, maximum=0.99, name='a_1', latex_label='$a_1$', unit=None, boundary=None)
a_2 = Uniform(minimum=0.0, maximum=0.99, name='a_2', latex_label='$a_2$', unit=None, boundary=None)
tilt_1 = Sine(minimum=0, maximum=3.141592653589793, name='tilt_1', latex_label='$\theta_1$', unit=None, boundary=None)
tilt_2 = Sine(minimum=0, maximum=3.141592653589793, name='tilt_2', latex_label='$\theta_2$', unit=None, boundary=None)
phi_12 = Uniform(minimum=0.0, maximum=6.283185307179586, name='phi_12', latex_label='$\Delta\phi$', unit=None, boundary='periodic')
phi_jl = Uniform(minimum=0.0, maximum=6.283185307179586, name='phi_jl', latex_label='$\phi_{JL}$', unit=None, boundary='periodic')
dec = Cosine(minimum=-1.5707963267948966, maximum=1.5707963267948966, name='dec', latex_label='$\mathrm{DEC}$', unit=None, boundary=None)
ra = Uniform(minimum=0, maximum=6.283185307179586, name='ra', latex_label='$\mathrm{RA}$', unit=None, boundary='periodic')
theta_jn = Sine(minimum=0.0, maximum=3.141592653589793, name='theta_jn', latex_label='$\theta_{JN}$', unit=None, boundary=None)
psi = Uniform(minimum=0.0, maximum=3.141592653589793, name='psi', latex_label='$\psi$', unit=None, boundary='periodic')
phase = Uniform(minimum=0.0, maximum=6.283185307179586, name='phase', latex_label='$\phi$', unit=None, boundary='periodic')"""

    if glint_rec:
        echo_lines = (
            "echo_delta_t = Uniform(minimum=0.1, maximum=1.0, name='echo_delta_t', "
            r"latex_label='$\Delta t_{glint}$', unit='$s$')"
            "\n"
            "echo_amp = Uniform(minimum=0.0, maximum=1.0, name='echo_amp', "
            r"latex_label='$\epsilon_{glint}$')"
        )
    else:
        echo_lines = "echo_delta_t = 0.0\necho_amp = 0.0"

    content = "\n".join([
        "# Prior file auto-generated by make_bilby_pipe_ini.py",
        "# Mirrors the priors in gradar_template5.py.",
        chirp_line,
        fixed,
        dist_line,
        geo_line,
        "# --- gravitational glint echo parameters ---",
        echo_lines,
        "",
    ])
    with open(path, 'w') as f:
        f.write(content)
    log(f"Wrote prior file -> {path}")


# --- ini file -----------------------------------------------------------------

def write_ini(path, *, label, outdir, trigger_time, detectors, channel_dict,
              psd_dict, prior_file, sampling_frequency, duration,
              post_trigger_duration, minimum_frequency, reference_frequency,
              nlive, nact, maxmcmc, dlogz, sampling_seed, request_cpus,
              n_parallel, time_reference, accounting, local):
    def fmt_dict(d):
        return "{" + ", ".join(f"{k}:{v}" for k, v in d.items()) + "}"

    sampler_kwargs = (
        "{"
        f"'nlive': {nlive}, 'nact': {nact}, 'maxmcmc': {maxmcmc}, "
        f"'sample': 'act-walk', 'dlogz': {dlogz}"
        "}"
    )

    lines = [
        "################################################################################",
        "## bilby_pipe configuration auto-generated by make_bilby_pipe_ini.py",
        "## Reproduces the run defined by gradar_template5.py",
        "################################################################################",
        "",
        "## ---- Job / output ----",
        f"label = {label}",
        f"outdir = {os.path.abspath(outdir)}",
        f"accounting = {accounting}",
        f"local = {local}",
        "transfer-files = False",
        f"request-cpus = {request_cpus}",
        "request-memory = 8.0",
        f"n-parallel = {n_parallel}",
        "create-plots = True",
        "plot-corner = True",
        "",
        "## ---- Data ----",
        f"trigger-time = {trigger_time}",
        f"detectors = {detectors}",
        f"duration = {duration}",
        f"sampling-frequency = {sampling_frequency}",
        f"post-trigger-duration = {post_trigger_duration}",
        f"minimum-frequency = {minimum_frequency}",
        f"maximum-frequency = {sampling_frequency // 2}",
        f"reference-frequency = {reference_frequency}",
        "# GWOSC open data (deglitched-frame auto-selection from the template is",
        "# NOT reproduced here -- edit channel-dict if you have cleaned frames).",
        f"channel-dict = {fmt_dict(channel_dict)}",
        "# PSDs taken straight from the GWTC PE data-release HDF5 (not estimated).",
        f"psd-dict = {fmt_dict(psd_dict)}",
        "",
        "## ---- Waveform ----",
        "frequency-domain-source-model = lal_binary_black_hole",
        "waveform-approximant = IMRPhenomXPHM",
        "",
        "## ---- Prior ----",
        f"prior-file = {os.path.abspath(prior_file)}",
        "# Bounds for chirp_mass / geocent_time were derived from the GWTC",
        "# posteriors and baked into the prior file above.",
        "",
        "## ---- Likelihood ----",
        "likelihood-type = GravitationalWaveTransient",
        "distance-marginalization = True",
        "phase-marginalization = False",
        "time-marginalization = False",
        f"time-reference = {time_reference}",
        "",
        "## ---- Sampler ----",
        "sampler = dynesty",
        f"sampling-seed = {sampling_seed}",
        f"sampler-kwargs = {sampler_kwargs}",
        "",
    ]
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    log(f"Wrote ini file -> {path}")


# --- main ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a bilby_pipe .ini reproducing the gradar_template5.py run.")
    parser.add_argument('--event', required=True, help="event name, e.g. GW241225_082815")
    parser.add_argument('--outdir', default='.', help="output directory for ini/prior/psd + results")
    parser.add_argument('--label', default=None, help="run label (default: <event>-glint_rec / -no_glint_rec)")
    parser.add_argument('--noglintrec', action='store_false', dest='glint_rec',
                        help="pin echo params to 0 (baseline), matching gradar_template5.py")
    parser.add_argument('--nlive', type=int, default=2048)
    parser.add_argument('--nact', type=int, default=20)
    parser.add_argument('--maxmcmc', type=int, default=8192)
    parser.add_argument('--dlogz', type=float, default=0.1)
    parser.add_argument('--samplerate', type=int, default=4096)
    parser.add_argument('--duration', type=int, default=12)
    parser.add_argument('--post-trigger-duration', type=int, default=6)
    parser.add_argument('--minimum-frequency', type=int, default=20)
    parser.add_argument('--reference-frequency', type=int, default=20)
    parser.add_argument('--rand', type=int, default=88170235, help="-> sampling-seed")
    parser.add_argument('--npool', type=int, default=12, help="-> request-cpus")
    parser.add_argument('--n-parallel', type=int, default=1)
    parser.add_argument('--gwtc_dir', default=None, help="dir to find/download GWTC HDF5 (default: outdir)")
    parser.add_argument('--gwtc_file', default=None, help="explicit GWTC HDF5 path (overrides auto-detection)")
    parser.add_argument('--trigger-time', type=float, default=None,
                        help="override trigger GPS time (default: looked up from event)")
    parser.add_argument('--accounting', default='ligo.dev.o4.cbc.pe.bilby',
                        help="HTCondor accounting tag (set to your group)")
    parser.add_argument('--local', action='store_true', help="run locally instead of via HTCondor")
    args = parser.parse_args()

    glint_rec = args.glint_rec
    label = args.label or f"{args.event}-{'glint_rec' if glint_rec else 'no_glint_rec'}"
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    gwtc_dir = os.path.abspath(args.gwtc_dir) if args.gwtc_dir else outdir

    # Trigger time (same call as the template).
    if args.trigger_time is not None:
        trigger_time = args.trigger_time
    else:
        import bilby
        log("Looking up trigger time...")
        trigger_time = bilby.gw.utils.get_event_time(args.event)
    log(f"Trigger time: {trigger_time}")

    # GWTC release file -> analysis key -> prior bounds + PSDs.
    file_name = get_gwtc_filename(args.event, trigger_time, gwtc_dir, args.gwtc_file)
    log(f"Using GWTC file: {file_name}")
    with h5py.File(file_name, 'r') as gwtc_file:
        analysis_key = get_analysis_key(gwtc_file)
        log(f"Using analysis key: {analysis_key}")
        min_chirp, max_chirp, min_geo, max_geo = get_prior_bounds(gwtc_file, analysis_key)
        detectors, psd_dict = get_detectors_and_psds(gwtc_file, analysis_key, outdir)

    if not detectors:
        raise RuntimeError("No detectors with PSDs found in the GWTC file; cannot build ini.")

    # Single detector -> use that detector's time reference (mirrors template).
    time_reference = 'geocent' if len(detectors) > 1 else detectors[0]
    channel_dict = {det: 'GWOSC' for det in detectors}

    prior_file = os.path.join(outdir, f"{label}.prior")
    write_prior_file(prior_file, min_chirp, max_chirp, min_geo, max_geo, glint_rec)

    ini_file = os.path.join(outdir, f"{label}.ini")
    write_ini(
        ini_file,
        label=label, outdir=outdir, trigger_time=trigger_time,
        detectors=detectors, channel_dict=channel_dict, psd_dict=psd_dict,
        prior_file=prior_file, sampling_frequency=args.samplerate,
        duration=args.duration, post_trigger_duration=args.post_trigger_duration,
        minimum_frequency=args.minimum_frequency,
        reference_frequency=args.reference_frequency,
        nlive=args.nlive, nact=args.nact, maxmcmc=args.maxmcmc, dlogz=args.dlogz,
        sampling_seed=args.rand, request_cpus=args.npool, n_parallel=args.n_parallel,
        time_reference=time_reference, accounting=args.accounting, local=args.local,
    )

    log("")
    log("Done. Detectors: " + ", ".join(detectors) + f" | glint_rec={glint_rec}")
    log(f"Next: bilby_pipe {ini_file}" + ("" if args.local else "   # then condor_submit the generated dag"))


if __name__ == '__main__':
    main()
