#!/usr/bin/env python
"""
This launch script runs gravitational glint parameter estimation on a confirmed binary blackhole merger from the GSWOC database.
Requires bilby grav-glint branch: https://gitlab.com/mattcarney106/bilby-mattcarney106
Updated branch: https://github.com/adamhasankhan/bilby-adamhasankhan/tree/grav-glint
Uses gwpy [1] to download gravitational wave data.

sample launch:
    python3 gradar_template5.py --npool 12 --outdir ~/public_html/glint/GW150914 --label GW150914-run1 --event GW150914

[1] https://gwpy.github.io/docs/stable/timeseries/remote-access.html
"""
import bilby
from bilby.core.utils import logger
from bilby.gw.conversion import component_masses_to_chirp_mass
#from bilby.gw.utils import calculate_time_to_merger
from gwpy.timeseries import TimeSeries
import h5py
#from gwosc.datasets import event_gps
import datetime
import argparse
import numpy as np
import urllib.request
import json

from multiprocessing import set_start_method



#--------------------------------------
#Ethan Addition:
import astropy
from astropy.cosmology import LambdaCDM

import gwosc
from gwosc.datasets import event_gps
from gwosc.datasets import run_at_gps
import os
#------------------------------------


set_start_method('fork')

#parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--noglintrec', action='store_false', help="does not recover in echo model params")
parser.add_argument('--npool', type=int, default=1, help="number of cores to run on")
parser.add_argument('--label', type=str, default ='glint_test', help="label for run")
parser.add_argument('--outdir', type=str, default='.', help = "outdir for run")
parser.add_argument('--rand', type=int, default=88170235, help="sets the random seed for noise instance")
parser.add_argument('--nlive', type=int, default=2048, help="number of live points to use in dynesty sampler")
parser.add_argument('--nact', type=int, default=50, help="nact dynesty sampler hyperparameter")
parser.add_argument('--event', type=str, default='bleh', help="name of event to run analysis on")
parser.add_argument('--maxmcmc', type=int, default=10000, help="max_mcmc dynesty hyper paramater")
parser.add_argument('--samplerate', type=int, default=4096, help="sample rate for data")
parser.add_argument('--gwtc_dir', type=str, default='.', help="directory containing GWTC PE data release HDF5 files")
parser.add_argument('--gwtc_file', type=str, default=None, help="explicit path to GWTC HDF5 file (overrides auto-detection)")

args = parser.parse_args()

glint_rec=args.noglintrec
outdir=args.outdir
npool=args.npool
label=args.label
rand=args.rand
nlive=args.nlive
nact=args.nact
event=args.event
maxmcmc=args.maxmcmc
samplerate=args.samplerate
gwtc_dir=args.gwtc_dir
gwtc_file_override=args.gwtc_file

# Set up the logger and log command line args
bilby.core.utils.setup_logger(outdir=outdir, label=label)
np.random.seed(rand)
logger.info(datetime.datetime.now())
logger.info(f"Performing gravitational wave analysis on event {event}")
logger.info(f"Using random seed: {rand}")

# Known glitched events (reference only — detection is handled automatically via GWOSC)
# GWTC-5.0 H1: GW240919_061559, GW240629_145256, GW240922_142106
# GWTC-5.0 L1: GW241127_061008, GW250119_190238, GW240413_022019, GW240930_035959,
#              GW241111_111552, GW241113_163507, GW240515_005301, GW240520_213616,
#              GW241102_144729, GW241114_235258
# GWTC-5.0 V1: GW240705_053215, GW241130_034908
# GWTC-4   H1: GW231123_135430, GW230707_124047, GW230606_004305, GW231118_090602
# GWTC-3   H1: GW191109_010717, GW191113_071753, GW191127_050227, GW191219_163120
# GWTC-3   L1: GW191109_010717, GW191219_163120, GW200105_162426, GW200115_042309
# GWTC-3   V1: GW191105_143521
# GWTC-2.1 L1: GW190413_134308, GW190425_081805, GW190503_185404, GW190513_205428,
#              GW190514_065416, GW190701_203306, GW190924_021846
# GWTC-1   L1: GW170817

# Zenodo record IDs for GWTC PE data releases.
_GWTC4_ZENODO_ID = '17602505'    # https://doi.org/10.5281/zenodo.17602505
_GWTC5_ZENODO_IDS = ['20276106', '20291740']  # https://doi.org/10.5281/zenodo.20276106 and .20291740


def _zenodo_find_and_download(event, record_ids, download_dir):
    """
    Query Zenodo records for an HDF5 file whose name contains the event string.
    Downloads the file to download_dir if not already present.
    Returns the local path on success, None on failure.
    """
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
                        logger.info(f"Using cached file: {local_path}")
                        return local_path
                    content_url = entry['links']['content']
                    logger.info(f"Downloading {key} from Zenodo record {record_id}...")
                    urllib.request.urlretrieve(content_url, local_path)
                    logger.info(f"Downloaded to {local_path}")
                    return local_path
        except Exception as e:
            logger.warning(f"Zenodo query for record {record_id} failed: {e}")
    return None


def get_gwtc_filename(event, trigger_time, gwtc_dir='.', gwtc_file_override=None):
    """
    Resolve the GWTC PE data release HDF5 filename for a given event.

    Checks for an explicit override first, then scans gwtc_dir for an existing
    file whose name contains the event string, then downloads from Zenodo for
    O4 events.  Falls back to the canonical filename convention for O1–O3.
    """
    if gwtc_file_override:
        return gwtc_file_override

    # Check local directory for an already-downloaded file.
    try:
        for entry in os.scandir(gwtc_dir):
            if entry.is_file() and entry.name.endswith(('.hdf5', '.h5')) and event in entry.name:
                logger.info(f"Found HDF5 file on disk: {entry.path}")
                return entry.path
    except OSError:
        pass

    # For O4 events, download the exact file from Zenodo.
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
        logger.warning(
            f"Zenodo download failed for {event}. Pass --gwtc_file to specify the file manually."
        )

    # O1/O2/O3 fall back to the canonical filename convention (must be pre-downloaded).
    if trigger_time > 1256655618:  # O3b -> GWTC-3
        fallback = f'IGWN-GWTC3p0-v1-{event}_PEDataRelease_mixed_cosmo.h5'
    elif trigger_time > 1238112018:  # O3a -> GWTC-2.1
        fallback = f'IGWN-GWTC2p1-v2-{event}_PEDataRelease_mixed_cosmo.h5'
    else:  # O1/O2 -> GWTC-1
        fallback = f'GWTC-1_sample_release/{event}_GWTC-1.hdf5'

    return os.path.join(gwtc_dir, fallback)


def get_analysis_key(gwtc_file_handle):
    """
    Return the HDF5 group key for the preferred waveform analysis.

    Prefers IMRPhenomXPHM-SpinTaylor; falls back to any IMRPhenom or SEOBNR
    key, then the first available key.
    """
    keys = list(gwtc_file_handle.keys())
    for k in keys:
        if 'IMRPhenomXPHM' in k and 'SpinTaylor' in k:
            return k
    for k in keys:
        if 'IMRPhenom' in k or 'SEOBNR' in k:
            return k
    logger.warning(f"No standard waveform key found in HDF5. Available: {keys}. Using first key.")
    return keys[0]


def get_prior_bounds(gwtc_file_handle, analysis_key):
    """
    Derive chirp-mass and geocentric-time prior bounds from the GWTC posterior
    samples by expanding the posterior range by 50% on each side.

    Returns (Min_Chirp_Mass, Max_Chirp_Mass, Min_Geo, Max_Geo).
    """
    samples = gwtc_file_handle[analysis_key]['posterior_samples']

    chirp_masses = samples['chirp_mass'][:]
    cm_range = chirp_masses.max() - chirp_masses.min()
    buffer_cm = max(0.5 * cm_range, 1.0)  # at least 1 M_sun buffer
    Min_Chirp_Mass = max(1.0, float(chirp_masses.min()) - buffer_cm)
    Max_Chirp_Mass = float(chirp_masses.max()) + buffer_cm

    # Try geocent_time first; fall back to any detector-specific time column.
    geo_key = 'geocent_time'
    if geo_key not in samples.dtype.names:
        for candidate in ('H1_time', 'L1_time', 'V1_time'):
            if candidate in samples.dtype.names:
                geo_key = candidate
                break
    geo_times = samples[geo_key][:]
    Min_Geo = float(geo_times.min()) - 0.2
    Max_Geo = float(geo_times.max()) + 0.2

    logger.info(
        f"Prior bounds from posteriors — chirp mass: [{Min_Chirp_Mass:.4f}, {Max_Chirp_Mass:.4f}] M_sun, "
        f"time: [{Min_Geo:.3f}, {Max_Geo:.3f}] s"
    )
    return Min_Chirp_Mass, Max_Chirp_Mass, Min_Geo, Max_Geo


def fetch_strain_data(det, start_time, end_time, event, samplerate):
    """
    Fetch strain data for a detector, automatically preferring deglitched/cleaned
    data from GWOSC when available.

    Returns (TimeSeries, is_deglitched: bool).
    """
    from gwosc import locate

    # Cleaned channel name candidates ordered by preference.
    # LIGO uses DCS- prefix (O3+) or GDS- prefix (O2 and earlier).
    clean_channels = [
        f"{det}:DCS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",
        f"{det}:DCS-CALIB_STRAIN_CLEAN_C01",
        f"{det}:GDS-CALIB_STRAIN_CLEAN_SUB60HZ_C01",
        f"{det}:GDS-CALIB_STRAIN_CLEAN_C01",
    ]

    # Query GWOSC for frame URLs available in the time window, then filter
    # for cleaned/deglitched frames by filename convention.
    try:
        all_urls = locate.get_urls(det, int(start_time), int(end_time))
        clean_urls = [
            u for u in all_urls
            if any(tag in u for tag in ("CLEAN", "CLN", "clean", "cln"))
        ]

        if clean_urls:
            logger.info(
                f"Found {len(clean_urls)} deglitched frame(s) for {det} on GWOSC — "
                f"attempting to use cleaned data"
            )
            for channel in clean_channels:
                try:
                    data = TimeSeries.read(
                        clean_urls, channel,
                        start=start_time, end=end_time,
                        format="gwf.lalframe",
                    )
                    logger.info(f"Using deglitched data for {det} (channel: {channel})")
                    return data.resample(samplerate), True
                except Exception:
                    continue
            # Cleaned frames found but no channel matched — warn and fall through.
            logger.warning(
                f"Deglitched frames found for {det} but no known channel could be read. "
                f"Falling back to standard data. Frame URLs: {clean_urls}"
            )
    except Exception as e:
        logger.info(f"GWOSC locate query for {det} raised {type(e).__name__}: {e}")

    # Standard open-data fallback.
    logger.info(f"No glitches found in {det}")
    if event.startswith(("GW23", "GW24", "GW25")):
        data = TimeSeries.fetch_open_data(det, start_time, end_time, sample_rate=4096, verbose=True)
    else:
        gps = event_gps(event)
        dataset = run_at_gps(gps)
        data = TimeSeries.fetch_open_data(det, start_time, end_time, sample_rate=4096, verbose=True, dataset=dataset)

    return data.resample(samplerate), False


#logic to set a good trigger time
#trigger_time = event_gps(event)
print('Getting trigger time...')
trigger_time = bilby.gw.utils.get_event_time(event)
print('Got trigger time...')
logger.info(f"Running analysis with event time: {trigger_time}")

duration = 12  # Analysis segment duration
post_trigger_duration = 6  # Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

logger.info("Verifying PE data release file...")
file_name = get_gwtc_filename(event, trigger_time, gwtc_dir, gwtc_file_override)
logger.info(f"Using GWTC file: {file_name}")
gwtc_file = h5py.File(file_name, 'r')
#only need r not r+ because we are only reading, not reading and writing the GWTC file

analysis_key = get_analysis_key(gwtc_file)
logger.info(f"Using analysis key: {analysis_key}")

Min_Chirp_Mass, Max_Chirp_Mass, Min_Geo, Max_Geo = get_prior_bounds(gwtc_file, analysis_key)


#download and cache gravitational wave data
#set up each ifo and calculate psd's
#lastly, plots strain data and psd in outdir
ifo_list = bilby.gw.detector.InterferometerList([])
for det in ["H1", "L1", "V1"]:
    try:
        logger.info("Downloading analysis data for ifo {}".format(det))
        ifo = bilby.gw.detector.get_empty_interferometer(det)

        data, is_deglitched = fetch_strain_data(det, start_time, end_time, event, samplerate)
        if is_deglitched:
            logger.info(f"Glitch detected and cleaned data used for {det}")

        logger.info("Loading GWTC PSD data for ifo {} from file".format(det))
        freq_array=gwtc_file[analysis_key]['psds'][det][:,0]
        psd_array=gwtc_file[analysis_key]['psds'][det][:,1]
        ifo.strain_data.set_from_gwpy_timeseries(data)

        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
              frequency_array=freq_array, psd_array=psd_array)
        ifo_list.append(ifo)
    except (ValueError, ExceptionGroup):
        logger.info(f"No data available for {det} removing {det} from analysis...")
    except KeyError:
        logger.info(f"PSD not available for {det} removing {det} from analysis...")

if len(ifo_list) == 0:
    logger.error("No detectors loaded successfully! Cannot proceed.")
    gwtc_file.close()
    raise RuntimeError("Analysis cannot proceed without detector data")


gwtc_file.close()
logger.info("Saving data plots to {}".format(outdir))
bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
ifo_list.plot_data(outdir=outdir, label=label)

priors = bilby.gw.prior.BBHPriorDict()
#set variable priors
#priors['geocent_time'] = bilby.core.prior.Uniform(
#        minimum = their_trigger - 0.1,
#        maximum = their_trigger + 0.1,
#        name='geocent_time', latex_label='$t_c$', unit='$s$')
#priors['luminosity_distance'] = bilby.core.prior.PowerLaw(alpha=2, name='luminosity_distance', minimum=100, maximum=10000, latex_label='$d_L$', unit='Mpc', boundary=None)
#priors['mass_1'] = bilby.gw.prior.Constraint(minimum=1, maximum=1000)
#priors['mass_2'] = bilby.gw.prior.Constraint(minimum=1, maximum=1000)
#priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(minimum=min_chirp, maximum=max_chirp, name='chirp_mass', boundary=None)
#priors['mass_ratio'] = bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.05, maximum=1.)

#add glint parameters for us to sample in or pin glint params to 0
if glint_rec:
    logger.info("Recovering with gravitational glint model...")
    priors['echo_delta_t'] = bilby.core.prior.Uniform(
       minimum=0.1,
       maximum=1.,
       name='echo_delta_t', latex_label='$\\Delta t_{glint}$', unit='$s$')
    priors['echo_amp'] = bilby.core.prior.Uniform(
       minimum=0.0,
       maximum=0.1,
       name='echo_amp', latex_label='$\\epsilon_{glint}$')
else:
    logger.info("Recovering with glint parameters pinned at 0...")
    priors['echo_delta_t'] = 0.0
    priors['echo_amp'] = 0.0


available_dets = [ifo.name for ifo in ifo_list]

if len(available_dets) > 1:
    # Multi-detector: use geocentric time (or check GWTC file for which one they used)
    time_ref = 'geocent'
    priors['geocent_time'] = bilby.core.prior.Uniform(
        minimum=Min_Geo, maximum=Max_Geo,
        name='geocent_time', latex_label='$t_c$', unit='$s$')
else:
    # Single detector: use that detector's time
    det = available_dets[0]  # 'L1' or 'H1'
    time_ref = det
    priors[f'{det}_time'] = bilby.core.prior.Uniform(
        minimum=Min_Geo, maximum=Max_Geo,
        name=f'{det}_time', latex_label=f'$t_{{{det}}}$', unit='$s$')


# GWTC Priors
priors['a_1'] = bilby.core.prior.Uniform(minimum=0.1, maximum=0.8, name='a_1', latex_label='$a_1$', unit=None, boundary=None)
priors['a_2'] = bilby.core.prior.Uniform(minimum=0.1, maximum=0.8, name='a_2', latex_label='$a_2$', unit=None, boundary=None)
priors['chirp_mass'] = bilby.gw.prior.UniformInComponentsChirpMass(minimum = Min_Chirp_Mass, maximum = Max_Chirp_Mass, name='chirp_mass', latex_label='$\\mathcal{M}$', unit='$M_{\\odot}$', boundary=None)
priors['luminosity_distance'] = bilby.gw.prior.UniformSourceFrame(
    minimum=10.0, maximum=10000.0,
    cosmology=LambdaCDM(name=None, H0=67.9, Om0=0.3065, Ode0=0.6935, Tcmb0=0., Neff=3.04, m_nu=None, Ob0=None),
    name='luminosity_distance', latex_label='$d_L$', unit='Mpc', boundary=None)
priors['mass_1'] = bilby.gw.prior.Constraint(minimum=1, maximum=1000, name='mass_1', latex_label='$m_1$', unit=None)
priors['mass_2']=bilby.gw.prior.Constraint(minimum=1, maximum=1000, name='mass_2', latex_label='$m_2$', unit=None)
priors['mass_ratio']=bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.05, maximum=1.0, name='mass_ratio', latex_label='$q$', unit=None, boundary=None)
priors['phase']=bilby.core.prior.Uniform(minimum=0.8, maximum=3.85, name='phase', latex_label='$\\phi$', unit=None, boundary='periodic')
priors['phi_12']=bilby.core.prior.Uniform(minimum=1.0, maximum=5.15, name='phi_12', latex_label='$\\Delta\\phi$', unit=None, boundary='periodic')
priors['phi_jl']=bilby.core.prior.Uniform(minimum=1.6, maximum=5.8, name='phi_jl', latex_label='$\\phi_{JL}$', unit=None, boundary='periodic')
priors['psi']=bilby.core.prior.Uniform(minimum=0.75, maximum=2.7, name='psi', latex_label='$\\psi$', unit=None, boundary='periodic')
priors['theta_jn']=bilby.core.prior.Sine(minimum=0.3, maximum=2.85, name='theta_jn', latex_label='$\\theta_{JN}$', unit=None, boundary=None)
priors['tilt_1']=bilby.core.prior.Sine(minimum=0, maximum=3.141592653589793, name='tilt_1', latex_label='$\\theta_1$', unit=None, boundary=None)
priors['tilt_2']=bilby.core.prior.Sine(minimum=0, maximum=3.141592653589793, name='tilt_2', latex_label='$\\theta_2$', unit=None, boundary=None)


#define a waveform_generator to make a frequency domain strain with bbh source model
waveform_generator = bilby.gw.WaveformGenerator(
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments={'waveform_approximant': 'IMRPhenomXPHM',
                        'reference_frequency': 20,
                        'minimum_frequency': 20})


#uses standard transient gwave likelihood function
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    ifo_list, waveform_generator, priors=priors,
    time_marginalization=False,
    phase_marginalization=False,
    distance_marginalization=True,
    time_reference=time_ref)


#passing the priors, likelihood and command line args to the sampler
#using the dynesty sampler
#converts params to generate all bbh parameters
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', outdir=outdir, label=label,
    nlive=nlive, maxmcmc = maxmcmc, nact=nact, dlogz = 0.1, check_point_plot=True, npool=npool,
    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters, check_point_delta_t=3600)
logger.info("Generating corner plots...")
result.plot_corner()
result.plot_corner(parameters=['echo_delta_t', 'echo_amp'], filename="{}/echo_params.pdf".format(outdir))
logger.info("Run completed.")
