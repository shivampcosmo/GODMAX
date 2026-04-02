"""
Utility functions for the paste_backlight_maps_analytic_test notebook.
Contains configuration loading, halo processing, map generation, and analysis helpers.
"""

import os
import sys
import pathlib
import gc
import warnings
import time
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore")
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import numpy as np
import yaml
import pickle as pk
import h5py as h5
import healpy as hp
from scipy import interpolate
from scipy.interpolate import interp1d
from deepmerge import always_merger

import jax
import jax.numpy as jnp
import jax.scipy.integrate as jsi
import jax_cosmo.background as bkgrd
from jax_cosmo import Cosmology
from jax_cosmo.background import radial_comoving_distance

from halo_processing_utils import process_halo


# ---------------------------------------------------------------------------
# Cosmological volume helpers (nbar <-> nz_lens conversion)
# ---------------------------------------------------------------------------

def compute_dV_dz_per_sr(cosmo_jax, z_array):
    """
    Compute the comoving volume element per steradian per unit redshift:
        dV/(dz dOmega) = chi(z)^2 * dchi/dz

    where chi is the comoving radial distance and dchi/dz = c/H(z).
    Returns array of shape (len(z_array),) in units of (Mpc/h)^3 / sr.
    """
    a_array = 1.0 / (1.0 + z_array)
    chi = np.array(radial_comoving_distance(cosmo_jax, a_array))
    # dchi/dz via finite differences on the chi(z) relation
    dchi_dz = np.gradient(chi, z_array)
    return chi**2 * dchi_dz


def nbar_to_nz_lens(nbar_comoving, z_array, cosmo_jax):
    """
    Convert comoving number density nbar(z) [(Mpc/h)^-3] to a normalized
    angular redshift distribution nz_lens(z) [per unit z], suitable for
    the lens kernel in Cl calculations.

    The unnormalized angular density per unit z per steradian is:
        dn/dz/dOmega = nbar(z) * dV/(dz dOmega) = nbar(z) * chi^2 * dchi/dz

    nz_lens is this quantity normalized to unit integral over z.
    """
    dVdz = compute_dV_dz_per_sr(cosmo_jax, z_array)
    unnorm = np.array(nbar_comoving) * dVdz
    norm = np.trapz(unnorm, z_array)
    if norm > 0:
        return unnorm / norm
    return unnorm


def nz_lens_to_nbar(nz_lens, z_array, cosmo_jax, nbar_target):
    """
    Convert a normalized nz_lens(z) back to comoving nbar(z) such that
    the mean comoving density within the support equals nbar_target.

    nbar(z) = nbar_target  where nz_lens > 0, scaled so that the
    round-trip nbar -> nz -> nbar is consistent.
    """
    dVdz = compute_dV_dz_per_sr(cosmo_jax, z_array)
    safe_dVdz = np.where(dVdz > 0, dVdz, 1.0)
    # Invert:  nz_lens(z) ∝ nbar(z) * dVdz  =>  nbar(z) ∝ nz_lens(z) / dVdz
    nbar_shape = np.where(dVdz > 0, nz_lens / safe_dVdz, 0.0)
    # Normalize so that the volume-weighted mean equals nbar_target
    if nbar_shape.max() > 0:
        nbar_shape = nbar_shape / nbar_shape.max() * nbar_target
    return nbar_shape


def check_nz_nbar_consistency(nbar_comoving, nz_lens_stored, z_array, cosmo_jax,
                              rtol=0.05, raise_error=False):
    """
    Check that nbar_comoving and nz_lens are mutually consistent by
    reconstructing nz_lens from nbar and comparing.

    Parameters
    ----------
    rtol : float
        Relative tolerance for the max fractional mismatch (default 5%).
    raise_error : bool
        If True, raise AssertionError on mismatch; otherwise print warning.

    Returns
    -------
    max_frac_mismatch : float
    """
    nz_reconstructed = nbar_to_nz_lens(nbar_comoving, z_array, cosmo_jax)
    nz_stored = np.array(nz_lens_stored)

    # Only compare where both are appreciably nonzero
    mask = (nz_stored > 1e-10) | (nz_reconstructed > 1e-10)
    if not mask.any():
        print("[nz consistency] Both arrays are zero everywhere — trivially consistent.")
        return 0.0

    denom = np.maximum(np.abs(nz_stored[mask]), np.abs(nz_reconstructed[mask]))
    denom = np.where(denom > 0, denom, 1.0)
    frac_diff = np.abs(nz_stored[mask] - nz_reconstructed[mask]) / denom
    max_mismatch = float(frac_diff.max())
    mean_mismatch = float(frac_diff.mean())

    print(f"[nz consistency] max fractional mismatch = {max_mismatch:.4e}, "
          f"mean = {mean_mismatch:.4e}  (rtol={rtol})")

    if max_mismatch > rtol:
        msg = (f"nbar/nz_lens inconsistency: max fractional mismatch "
               f"{max_mismatch:.4e} exceeds rtol={rtol}")
        if raise_error:
            raise AssertionError(msg)
        else:
            print(f"  WARNING: {msg}")

    return max_mismatch


def update_nz_from_mock_catalog(mock_gals, analysis_dict, zarray_lens, cosmo_jax,
                                zmin=None, zmax=None, smooth=True, smooth_sigma=1.0):
    """
    Replace the analytic nz_lens in analysis_dict with one measured from the
    mock galaxy catalog.  Useful for exact comparison diagnostics.

    Parameters
    ----------
    mock_gals : array, shape (N, >=3)
        Galaxy catalog; column 2 is redshift.
    analysis_dict : dict
        Will be modified in-place.
    zarray_lens : array
        Redshift grid.
    cosmo_jax : Cosmology
    zmin, zmax : float or None
        If None, inferred from the nz_lens support in analysis_dict.
    smooth : bool
        Gaussian-smooth the histogram before storing.
    smooth_sigma : float
        Smoothing kernel width in bins.

    Returns
    -------
    nz_realized : array
        The realized (possibly smoothed) nz_lens.
    """
    from scipy.ndimage import gaussian_filter1d

    gal_z = np.array(mock_gals[:, 2])
    dz = zarray_lens[1] - zarray_lens[0]
    edges = np.concatenate([
        [zarray_lens[0] - dz / 2],
        0.5 * (zarray_lens[1:] + zarray_lens[:-1]),
        [zarray_lens[-1] + dz / 2],
    ])

    if zmin is not None:
        gal_z = gal_z[(gal_z >= zmin) & (gal_z <= zmax)]

    counts, _ = np.histogram(gal_z, bins=edges)
    nz_raw = counts.astype(float)
    if smooth and len(nz_raw) > 3:
        nz_raw = gaussian_filter1d(nz_raw, sigma=smooth_sigma)

    norm = np.trapz(nz_raw, zarray_lens)
    nz_realized = nz_raw / norm if norm > 0 else nz_raw

    # Update analysis_dict in place
    nz_info = analysis_dict['nz_lens_info_dict']
    nz_info['nz0'] = np.maximum(nz_realized, 1e-10)
    analysis_dict['nz_lens_info_dict'] = nz_info

    print(f"[update_nz_from_mock] Replaced nz_lens with realized n(z) from "
          f"{len(gal_z)} galaxies (smooth={smooth})")

    return nz_realized


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

def get_project_paths():
    """Return standard project paths relative to the notebooks/pasting directory."""
    curr_path = pathlib.Path(__file__).resolve().parent
    project_base = curr_path.parents[1]
    paths = {
        'data': project_base / "data",
        'src': project_base / "src",
        'results': project_base / "results",
        'params': project_base / "param_files",
    }
    # Ensure src is on sys.path
    for p in [curr_path, paths['src'], paths['data'], paths['results'], paths['params']]:
        sp = str(p)
        if sp not in sys.path:
            sys.path.append(sp)
    return paths


# ---------------------------------------------------------------------------
# YAML / config helpers
# ---------------------------------------------------------------------------

def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def generate_dicts(data):
    return (
        data.get('sim_params', {}),
        data.get('halo_params', {}),
        data.get('analysis', {}),
        data.get('other_params', {}),
    )


# ---------------------------------------------------------------------------
# Full configuration builder
# ---------------------------------------------------------------------------

def build_config(abs_path_params, abs_path_data, params_file='pasting/params.yaml',
                 nbar_comoving=5e-7, gal_zmin=0.005, gal_zmax=0.3):
    """
    Load YAML configs, build cosmology, construct all parameter dicts
    needed by the GODMAX pipeline.

    The galaxy redshift distribution is built from a **single canonical source**:
    a constant comoving number density ``nbar_comoving`` within [gal_zmin, gal_zmax].
    Both ``nbar_gal_comoving_val`` (used by the HOD / Mthresh path in
    get_radial_profiles) and ``nz_lens`` (used by the lens kernel in get_Cls)
    are derived from this same specification, guaranteeing mutual consistency.

    Parameters
    ----------
    abs_path_params : Path
    abs_path_data : Path
    params_file : str
    nbar_comoving : float
        Constant comoving galaxy number density [(Mpc/h)^-3] within the
        galaxy redshift window.
    gal_zmin, gal_zmax : float
        Redshift bounds of the galaxy sample.

    Returns
    -------
    sim_params_dict, halo_params_dict, analysis_dict, other_params_dict,
    cosmo_jax, zarray_lens, nz_lens, gal_zrange
        ``gal_zrange`` is the tuple ``(gal_zmin, gal_zmax)`` so downstream
        code (maps, plots) can use the same window without hard-coding.
    """
    default_data = read_yaml(str(abs_path_params / 'params_default.yaml'))
    new_data = read_yaml(str(abs_path_params / params_file))
    merged_data = always_merger.merge(default_data, new_data)

    sim_params_dict, halo_params_dict, analysis_dict, other_params_dict = generate_dicts(merged_data)

    # Cosmology
    cp = sim_params_dict['cosmo']
    cosmo_jax = Cosmology(
        Omega_c=cp['Om0'] - cp['Ob0'],
        Omega_b=cp['Ob0'],
        h=cp['H0'] / 100.,
        sigma8=cp['sigma8'],
        n_s=cp['ns'],
        Omega_k=0., w0=cp['w0'], wa=0.,
    )

    # ------------------------------------------------------------------
    # Canonical galaxy n(z): constant nbar within [gal_zmin, gal_zmax]
    # ------------------------------------------------------------------
    ks = np.geomspace(3e-1, 10, 10)
    zarray_lens = np.linspace(0.001, 0.8, 200)
    nbins_lens = 1

    # Step 1: build nbar(z) — the single canonical source
    nbar_of_z = np.where(
        (zarray_lens >= gal_zmin) & (zarray_lens <= gal_zmax),
        nbar_comoving, 0.0
    )

    # Step 2: derive nz_lens from nbar via the volume element
    nz_lens_arr = nbar_to_nz_lens(nbar_of_z, zarray_lens, cosmo_jax)
    nz_lens = {0: nz_lens_arr}

    # Step 3: populate analysis_dict from the same canonical arrays
    analysis_dict['nbar_gal_comoving_zarray'] = zarray_lens
    analysis_dict['nbar_gal_comoving_val'] = nbar_of_z

    nz_info_lens = {
        'z_array_lens': zarray_lens,
        'nbins_lens': nbins_lens,
        'nz0': np.maximum(nz_lens_arr, 1e-10),
    }
    analysis_dict['nz_lens_info_dict'] = nz_info_lens

    # Verify round-trip consistency
    check_nz_nbar_consistency(nbar_of_z, nz_lens_arr, zarray_lens, cosmo_jax)

    # Source n(z) info (CMB lensing placeholder)
    analysis_dict['is_cmb_lensing'] = True
    nz_info_src = {'z_array_source': jnp.ones(1), 'nbins': 1, 'nz0': jnp.ones(1)}
    analysis_dict['nz_source_info_dict'] = nz_info_src
    other_params_dict['Delta_z_bias_array'] = jnp.zeros(1)
    other_params_dict['mult_shear_bias_array'] = jnp.zeros(1)

    # ell / k arrays
    analysis_dict['k_array_survey'] = jnp.array(ks / (cp['H0'] / 100.))
    lmin, lmax, dl_log = 80.0, 8800.0, 0.23025851
    l_all = np.exp(np.arange(np.log(lmin), np.log(lmax), dl_log))
    dl = l_all[1:] - l_all[:-1]
    l_survey = (l_all[1:] + l_all[:-1]) / 2.
    halo_params_dict['ell_array'] = jnp.array(l_survey)
    analysis_dict['l_array_survey'] = jnp.array(l_survey)
    analysis_dict['dl_array_survey'] = jnp.array(dl)
    # analysis_dict['symbolic_pk'] = True
    # analysis_dict['symbolic_hmf'] = True

    analysis_dict['symbolic_pk'] = False
    analysis_dict['symbolic_hmf'] = False


    # Halo parameter overrides
    halo_params_dict.update({
        'rmin': 0.005, 'rmax': 10.0, 'nr': 48,
        # 'zmin': 0.005, 'zmax': 0.8, 'nz': 52,
        'zmin': 0.3, 'zmax': 0.7, 'nz': 22,
        'lg10_Mmin': 13.0, 'lg10_Mmax': 15.75, 'nM': 42,
    })

    # Override cosmology to backlight sim cosmology
    sim_params_dict['cosmo'] = {
        'w0': -1.0, 'flat': True, 'H0': 67.11,
        'Om0': 0.3175, 'Ob0': 0.049, 'sigma8': 0.834, 'ns': 0.9624,
    }

    gal_zrange = (gal_zmin, gal_zmax)
    return (sim_params_dict, halo_params_dict, analysis_dict, other_params_dict,
            cosmo_jax, zarray_lens, nz_lens, gal_zrange)


# ---------------------------------------------------------------------------
# Array / data helpers
# ---------------------------------------------------------------------------

def split_array_power_ratio(arr, num_parts, power_val=0.5):
    n = arr.shape[0]
    j_vals = np.arange(1, num_parts + 1)
    weights = j_vals ** power_val
    ideal = (n / weights.sum()) * weights
    int_sizes = np.floor(ideal).astype(int)
    remainder = n - int_sizes.sum()
    frac = ideal - int_sizes
    int_sizes[np.argsort(frac)[-remainder:]] += 1
    return np.split(arr, np.cumsum(int_sizes)[:-1])


# ---------------------------------------------------------------------------
# Halo catalog I/O
# ---------------------------------------------------------------------------

def load_halo_catalog(filepath):
    """Load halo catalog from HDF5 and return (ra, dec, z, M200c, vlos)."""
    with h5.File(filepath, 'r') as f:
        ra = f['ra'][()]
        dec = f['dec'][()]
        z = f['z'][()]
        M200c = f['M200c'][()]
        vlos = f['vlos'][()]
    return ra, dec, z, M200c, vlos


# ---------------------------------------------------------------------------
# Parallel halo processing → pixel finding
# ---------------------------------------------------------------------------

def _concatenate_batch_results(results, M_chunk, z_chunk, vlos_chunk,
                               halo_cat_DV_np, halo_cat_R200c_np,
                               max_paint_R200c_factor, pixel_dtype):
    if not results:
        return None
    lengths = np.array([r[3] for r in results], dtype=np.int32)
    total = lengths.sum()

    nearby_pix_all = np.empty(total, dtype=pixel_dtype)
    distances_pix_all = np.empty(total, dtype=np.float32)
    halo_indices = np.empty(total, dtype=np.int32)

    ends = np.cumsum(lengths)
    starts = np.concatenate([[0], ends[:-1]])
    for i, (s, e, r) in enumerate(zip(starts, ends, results)):
        nearby_pix_all[s:e] = r[0]
        distances_pix_all[s:e] = r[1]
        halo_indices[s:e] = r[2]

    orig_halo_idx = np.array([r[2] for r in results], dtype=np.int32)
    logM = np.log(M_chunk[halo_indices]).astype(np.float32)
    z_ind = z_chunk[halo_indices].astype(np.float32)
    vlos_ind = vlos_chunk[halo_indices].astype(np.float32)
    ang_dist = halo_cat_DV_np[orig_halo_idx].astype(np.float32)
    rp_max = (max_paint_R200c_factor * halo_cat_R200c_np[orig_halo_idx]).astype(np.float32)

    return (nearby_pix_all, distances_pix_all, starts, ends,
            logM, z_ind, vlos_ind, ang_dist, rp_max)


def _final_concatenate_batches(all_results, pixel_dtype):
    total_pix = sum(len(r[0]) for r in all_results)
    total_halos = sum(len(r[2]) for r in all_results)

    fp = np.empty(total_pix, dtype=pixel_dtype)
    fd = np.empty(total_pix, dtype=np.float32)
    flm = np.empty(total_pix, dtype=np.float32)
    fz = np.empty(total_pix, dtype=np.float32)
    fv = np.empty(total_pix, dtype=np.float32)
    fs = np.empty(total_halos, dtype=np.int32)
    fe = np.empty(total_halos, dtype=np.int32)
    fa = np.empty(total_halos, dtype=np.float32)
    fr = np.empty(total_halos, dtype=np.float32)

    po, ho = 0, 0
    for res in all_results:
        npb, nhb = len(res[0]), len(res[2])
        pix_b, dist_b, s_b, e_b, lm_b, z_b, v_b, a_b, r_b = res
        fp[po:po + npb] = pix_b
        fd[po:po + npb] = dist_b
        flm[po:po + npb] = lm_b
        fz[po:po + npb] = z_b
        fv[po:po + npb] = v_b
        fs[ho:ho + nhb] = s_b + po
        fe[ho:ho + nhb] = e_b + po
        fa[ho:ho + nhb] = a_b
        fr[ho:ho + nhb] = r_b
        po += npb
        ho += nhb

    return (fp, fd, fs, fe, flm, fz, fv, fa, fr)


def process_halos_in_batches(M_chunk, ra_chunk, dec_chunk, z_chunk, vlos_chunk,
                             R200c, DA, max_paint_R200c_factor, nside, batch_size=1000):
    pixel_dtype = np.int32 if nside <= 8192 else np.int64
    ra_np = np.clip(np.array(ra_chunk, dtype=np.float32), 0.01, 359.99)
    dec_np = np.clip(np.array(dec_chunk, dtype=np.float32), -89.99, 89.99)
    R200c_np = np.array(R200c, dtype=np.float32)
    DA_np = np.array(DA, dtype=np.float32)

    n_halos = len(z_chunk)
    all_results = []

    with Pool(cpu_count()) as pool:
        for bs in range(0, n_halos, batch_size):
            be = min(bs + batch_size, n_halos)
            print(f"Processing halo batch {bs // batch_size + 1}/{(n_halos - 1) // batch_size + 1}...")
            args = [(j, ra_np, dec_np, R200c_np, DA_np,
                     max_paint_R200c_factor, nside, pixel_dtype) for j in range(bs, be)]
            batch_res = [r for r in pool.map(process_halo, args) if r is not None]
            if batch_res:
                bd = _concatenate_batch_results(
                    batch_res, M_chunk, z_chunk, vlos_chunk,
                    DA_np, R200c_np, max_paint_R200c_factor, pixel_dtype)
                if bd is not None:
                    all_results.append(bd)
            del batch_res, args
            gc.collect()

    if not all_results:
        return None
    return _final_concatenate_batches(all_results, pixel_dtype)


# ---------------------------------------------------------------------------
# Map generation pipeline
# ---------------------------------------------------------------------------

def generate_maps(ra_all, dec_all, z_all, M200c_all, vlos_all,
                  Prof_test, mock_params_dict_setup, nside,
                  sim_params_dict, halo_params_dict, analysis_dict, other_params_dict,
                  save_path=None, profile_timing=False):
    """
    Run pixel-finding + JAX map generation for a set of halos.
    Returns dict with keys: mock_gals_all, map_rhom, map_ymap, map_ksz, map_tau.
    """
    from get_sim_maps import get_sim_map
    import helpers.constants as constants

    # Force baryonified map so rhom_dmo_map_final is always populated.
    # Without this, rhommap_final is the DMO map and rhom_dmo_map_final
    # is undefined, making the baryonic kappa correction impossible.
    mock_params_dict_setup = dict(mock_params_dict_setup)
    mock_params_dict_setup['get_baryonifiedmap'] = True

    nh_max_map = {8192: 4e3, 4096: 5e4, 2048: 5e5, 1024: 1e7, 512: 5e7}
    nh_max = nh_max_map.get(nside, 1e5)
    nsel = len(M200c_all)
    num_chunks = int(np.ceil(nsel / nh_max)) if nsel > nh_max else 1

    npix = 12 * nside ** 2
    #map_rhom = np.zeros(npix, dtype=np.float32)
    map_rhom_dmb = np.zeros(npix, dtype=np.float32)  # baryonified halo matter
    map_rhom_dmo = np.zeros(npix, dtype=np.float32)  # DMO halo matter

    map_ymap = np.zeros(npix, dtype=np.float32)
    map_ksz = np.zeros(npix, dtype=np.float32)
    map_tau = np.zeros(npix, dtype=np.float32)
    mock_gals_all = {}

    for i in range(num_chunks):
        start, end = 0, len(ra_all)
        M_c = M200c_all[start:end]
        ra_c = ra_all[start:end]
        dec_c = dec_all[start:end]
        z_c = z_all[start:end]
        vlos_c = vlos_all[start:end]

        scale_fac = 1. / (1. + z_c)
        rho_c_z = constants.RHO_CRIT_0_KPC3 * bkgrd.Esqr(Prof_test.cosmo_jax, scale_fac) * 1e9
        R200c = (M_c * 3.0 / (4.0 * jnp.pi * 200 * rho_c_z)) ** (1.0 / 3.0)
        DA = bkgrd.angular_diameter_distance(Prof_test.cosmo_jax, scale_fac)

        t0 = time.perf_counter() if profile_timing else None
        max_paint_R200c_factor = 8.0
        batch_size = len(ra_all)
        result = process_halos_in_batches(
            M_c, ra_c, dec_c, z_c, vlos_c,
            R200c, DA, max_paint_R200c_factor, nside, batch_size)

        if profile_timing:
            print(f"[PROFILE] Chunk {i + 1}/{num_chunks} - Pixel finding: {time.perf_counter() - t0:.2f}s")

        if result:
            mock_params_dict = {
                **mock_params_dict_setup,
                'halo_z': jnp.array(z_c, dtype=jnp.float32),
                'halo_ra': jnp.array(ra_c, dtype=jnp.float32),
                'halo_dec': jnp.array(dec_c, dtype=jnp.float32),
                'halo_M': jnp.array(M_c, dtype=jnp.float64),
                'halo_vlos': jnp.array(vlos_c, dtype=jnp.float32),
                'nearby_pix_all': jnp.array(result[0]),
                'start_ind': jnp.array(result[2], dtype=jnp.int32),
                'end_ind': jnp.array(result[3], dtype=jnp.int32),
                'pix_prop_all': (jnp.array([
                    np.log(result[1]), result[5], result[4], result[6]
                ]).T).astype(jnp.float32),
                'ang_distance_all': jnp.array(result[7]),
                'rp_max_all': jnp.array(result[8]),
                'profile_timing': profile_timing,
            }

            t1 = time.perf_counter() if profile_timing else None
            mock_map = get_sim_map(
                sim_params_dict, halo_params_dict, analysis_dict,
                other_params_dict, mock_params_dict, Profiles_obj=Prof_test)

            #map_rhom += np.nan_to_num(mock_map.rhommap_final)
            map_rhom_dmb += np.nan_to_num(mock_map.rhommap_final)
            map_rhom_dmo += np.nan_to_num(mock_map.rhom_dmo_map_final)
            map_ymap += np.nan_to_num(mock_map.ymap_final)
            map_ksz += np.nan_to_num(mock_map.kszmap_final)
            map_tau += np.nan_to_num(mock_map.taumap_final)
            mock_gals_all[i] = mock_map.final_galaxy_catalog
            print(f"Chunk {i + 1}/{num_chunks}: galaxy catalog shape = {mock_gals_all[i].shape}")

            if profile_timing:
                print(f"[PROFILE] Chunk {i + 1}/{num_chunks} - JAX map gen: {time.perf_counter() - t1:.2f}s")

            jax.clear_caches()
            gc.collect()

    saved_data = {
    'mock_gals_all': mock_gals_all,
    'map_rhom_dmb':  map_rhom_dmb,  # baryonified (halos only, no N-body sheet)
    'map_rhom_dmo':  map_rhom_dmo,  # DMO        (halos only, no N-body sheet)
    'map_ymap':      map_ymap,
    'map_ksz':       map_ksz,
    'map_tau':        map_tau,
    # map_kappa absent intentionally — call compute_kappa_map() after this
    }

    if save_path is not None:
        with open(save_path, 'wb') as f:
            pk.dump(saved_data, f)
        print(f"Saved maps to {save_path}")

    return saved_data


# ---------------------------------------------------------------------------
# HOD diagnostics: measure Ncen/Nsat from mock vs theory
# ---------------------------------------------------------------------------

def measure_hod_from_catalog(mock_gals, halo_M200c, halo_z, Cls_test,
                             nM_bins=8, nz_bins=5):
    """
    Measure mean Ncen and Nsat from the generated mock catalog as a function
    of (M, z) bins, and compare against theoretical Ncen_mat / Nsat_mat.

    Parameters
    ----------
    mock_gals : array, shape (N, >=5)
        Galaxy catalog: [ra, dec, z, M_host, is_central, valid].
    halo_M200c : array
        Host halo masses (all halos used in simulation).
    halo_z : array
        Host halo redshifts.
    Cls_test : object
        Has .M_array, .z_array, .Ncen_mat, .Nsat_mat, .hmf_Mz_mat.
    nM_bins, nz_bins : int
        Number of bins in mass and redshift.

    Returns
    -------
    dict with keys:
        M_cen, z_cen : bin centers
        Ncen_meas, Nsat_meas : measured (nz_bins, nM_bins)
        Ncen_th, Nsat_th : theory interpolated to same bins
        frac_mismatch_Ncen, frac_mismatch_Nsat : fractional mismatches
    """
    gal_M = np.array(mock_gals[:, 3])
    gal_z = np.array(mock_gals[:, 2])
    is_cen = np.array(mock_gals[:, 4])

    logM_min = np.log10(halo_M200c.min())
    logM_max = np.log10(halo_M200c.max())
    z_min = halo_z.min()
    z_max = halo_z.max()

    logM_edges = np.linspace(logM_min, logM_max, nM_bins + 1)
    z_edges = np.linspace(z_min, z_max, nz_bins + 1)
    logM_cen = 0.5 * (logM_edges[1:] + logM_edges[:-1])
    z_cen = 0.5 * (z_edges[1:] + z_edges[:-1])

    Ncen_meas = np.full((nz_bins, nM_bins), np.nan)
    Nsat_meas = np.full((nz_bins, nM_bins), np.nan)
    Ncen_th = np.full((nz_bins, nM_bins), np.nan)
    Nsat_th = np.full((nz_bins, nM_bins), np.nan)

    halo_logM = np.log10(halo_M200c)
    gal_logM = np.log10(np.clip(gal_M, 1.0, None))

    for iz in range(nz_bins):
        z_lo, z_hi = z_edges[iz], z_edges[iz + 1]
        # Halos in this z-bin
        halo_sel = (halo_z >= z_lo) & (halo_z < z_hi)
        # Galaxies in this z-bin
        gal_z_sel = (gal_z >= z_lo) & (gal_z < z_hi)

        for im in range(nM_bins):
            lm_lo, lm_hi = logM_edges[im], logM_edges[im + 1]
            n_halos_bin = ((halo_logM >= lm_lo) & (halo_logM < lm_hi) & halo_sel).sum()
            if n_halos_bin == 0:
                continue

            gal_in_bin = gal_z_sel & (gal_logM >= lm_lo) & (gal_logM < lm_hi)
            n_cen = (gal_in_bin & (is_cen > 0.5)).sum()
            n_sat = (gal_in_bin & (is_cen < 0.5)).sum()

            Ncen_meas[iz, im] = n_cen / n_halos_bin
            Nsat_meas[iz, im] = n_sat / n_halos_bin

            # Theory: interpolate Ncen_mat, Nsat_mat to bin center
            jz_th = np.argmin(np.abs(np.array(Cls_test.z_array) - z_cen[iz]))
            jM_th = np.argmin(np.abs(np.log10(np.array(Cls_test.M_array)) - logM_cen[im]))
            Ncen_th[iz, im] = float(Cls_test.Ncen_mat[jz_th, jM_th])
            Nsat_th[iz, im] = float(Cls_test.Nsat_mat[jz_th, jM_th])

    # Fractional mismatch (where both are nonzero)
    valid = np.isfinite(Ncen_meas) & (Ncen_th > 1e-6)
    frac_Ncen = np.full_like(Ncen_meas, np.nan)
    frac_Nsat = np.full_like(Nsat_meas, np.nan)
    if valid.any():
        frac_Ncen[valid] = np.abs(Ncen_meas[valid] - Ncen_th[valid]) / Ncen_th[valid]
    valid_sat = np.isfinite(Nsat_meas) & (Nsat_th > 1e-6)
    if valid_sat.any():
        frac_Nsat[valid_sat] = np.abs(Nsat_meas[valid_sat] - Nsat_th[valid_sat]) / Nsat_th[valid_sat]

    # Print summary
    print("--- HOD Consistency Diagnostics ---")
    if valid.any():
        print(f"  Ncen: max frac mismatch = {np.nanmax(frac_Ncen):.3f}, "
              f"mean = {np.nanmean(frac_Ncen):.3f} "
              f"(over {valid.sum()} populated bins)")
    else:
        print("  Ncen: no populated bins to compare")
    if valid_sat.any():
        print(f"  Nsat: max frac mismatch = {np.nanmax(frac_Nsat):.3f}, "
              f"mean = {np.nanmean(frac_Nsat):.3f} "
              f"(over {valid_sat.sum()} populated bins)")
    else:
        print("  Nsat: no populated bins to compare")
    print("-----------------------------------")

    return dict(
        logM_cen=logM_cen, z_cen=z_cen,
        Ncen_meas=Ncen_meas, Nsat_meas=Nsat_meas,
        Ncen_th=Ncen_th, Nsat_th=Nsat_th,
        frac_mismatch_Ncen=frac_Ncen, frac_mismatch_Nsat=frac_Nsat,
    )


# ---------------------------------------------------------------------------
# Shot noise for Cl_gg
# ---------------------------------------------------------------------------

def compute_shot_noise_Cl(mock_gals, nside, zmin, zmax):
    """
    Compute the Poisson shot noise level for Cl_gg from the galaxy surface
    density. For a galaxy overdensity map delta = n/nbar - 1, the shot noise
    power spectrum is:

        Cl_shot = Omega_pix / N_gal_per_pix  (for full sky)
                = 4*pi / N_gal               (for full-sky uniform)

    More precisely, Cl_shot = 1 / n_bar_sr where n_bar_sr is the mean
    galaxy surface density per steradian.

    Parameters
    ----------
    mock_gals : array
        Galaxy catalog (column 2 = redshift).
    nside : int
    zmin, zmax : float

    Returns
    -------
    Cl_shot : float
        Constant shot noise power spectrum level.
    n_gal : int
        Number of galaxies in the selection.
    nbar_sr : float
        Mean angular number density [galaxies / sr].
    """
    z = np.array(mock_gals[:, 2])
    sel = (z > zmin) & (z < zmax)
    n_gal = int(sel.sum())

    # For full-sky: total solid angle = 4*pi sr
    omega_sky = 4 * np.pi  # sr
    nbar_sr = n_gal / omega_sky
    Cl_shot = 1.0 / nbar_sr if nbar_sr > 0 else 0.0

    print(f"[Shot noise] N_gal = {n_gal}, nbar = {nbar_sr:.1f} /sr, "
          f"Cl_shot = {Cl_shot:.4e}")
    return Cl_shot, n_gal, nbar_sr


def compute_Cl_ratio_in_bands(ell_meas, Cl_meas, ell_th, Cl_th,
                              bands=((100, 300), (300, 700), (700, 1200))):
    """
    Compute the mean ratio Cl_meas / Cl_th in ell bands.

    Parameters
    ----------
    ell_meas, Cl_meas : arrays (measured spectrum, integer ells)
    ell_th, Cl_th : arrays (theory spectrum, may be at different ells)
    bands : tuple of (ell_min, ell_max) pairs

    Returns
    -------
    dict mapping (ell_min, ell_max) -> mean ratio in that band
    """
    from scipy.interpolate import interp1d
    # Interpolate theory to measured ells
    Cl_th_interp = interp1d(ell_th, Cl_th, fill_value='extrapolate', bounds_error=False)

    results = {}
    for ell_lo, ell_hi in bands:
        mask = (ell_meas >= ell_lo) & (ell_meas <= ell_hi) & (Cl_meas > 0)
        if mask.sum() == 0:
            results[(ell_lo, ell_hi)] = np.nan
            continue
        th_vals = Cl_th_interp(ell_meas[mask])
        valid = th_vals > 0
        if valid.sum() == 0:
            results[(ell_lo, ell_hi)] = np.nan
            continue
        ratio = Cl_meas[mask][valid] / th_vals[valid]
        results[(ell_lo, ell_hi)] = float(np.mean(ratio))

    print("[Cl ratio diagnostics]")
    for (lo, hi), r in results.items():
        print(f"  ell [{lo:4d}, {hi:4d}]: <Cl_meas/Cl_th> = {r:.3f}")
    return results


# ---------------------------------------------------------------------------
# HMF: simulation catalog vs theory comparison
# ---------------------------------------------------------------------------

def compare_sim_vs_theory_hmf(halo_M200c, halo_z, Cls_test, nM_bins=12, nz_bins=5):
    """
    Compare the halo mass/redshift histogram from the simulation input catalog
    against absolute theory expectations from the Tinker HMF.

    Theory counts are computed as:
        N_theory(z-bin, M-bin) = V_comov(z-bin) * ∫_{M-bin} (dn/dlnM) dlnM

    where V_comov is the full-sky comoving volume of the z-bin:
        V = (4π/3) * (χ(z_hi)^3 − χ(z_lo)^3)

    dn/dlnM is in units of [(Mpc/h)^{-3}] (comoving number density per unit
    dlnM), as stored in Cls_test.hmf_Mz_mat.  The z-bin volume is evaluated
    at the bin edges (not integrated over sub-bins).

    Returns dict with:
        sim_counts, theory_counts_abs, theory_counts_norm,
        frac_mismatch_abs, frac_mismatch_norm,
        sim_to_theory_ratio, theory_to_sim_ratio
    """
    logM_min = np.log10(halo_M200c.min())
    logM_max = np.log10(halo_M200c.max())
    z_min, z_max = halo_z.min(), halo_z.max()

    logM_edges = np.linspace(logM_min, logM_max, nM_bins + 1)
    z_edges = np.linspace(z_min, z_max, nz_bins + 1)
    logM_cen = 0.5 * (logM_edges[1:] + logM_edges[:-1])
    z_cen = 0.5 * (z_edges[1:] + z_edges[:-1])

    halo_logM = np.log10(halo_M200c)

    sim_counts = np.zeros((nz_bins, nM_bins))
    theory_counts_abs = np.zeros((nz_bins, nM_bins))

    M_th = np.array(Cls_test.M_array)
    logM_th = np.log10(M_th)
    hmf_Mz = np.array(Cls_test.hmf_Mz_mat)   # shape (nz_th, nM_th) = dn/dlnM  [(Mpc/h)^{-3}]
    z_th = np.array(Cls_test.z_array)

    # Comoving volume per z-bin (full sky)
    # V = (4π/3) * (χ_hi^3 − χ_lo^3)
    cosmo = Cls_test.cosmo_jax
    z_bin_volumes = np.zeros(nz_bins)
    for iz in range(nz_bins):
        z_lo, z_hi = z_edges[iz], z_edges[iz + 1]
        chi_lo = float(radial_comoving_distance(cosmo, 1.0 / (1.0 + z_lo))[0])
        chi_hi = float(radial_comoving_distance(cosmo, 1.0 / (1.0 + z_hi))[0])
        z_bin_volumes[iz] = (4.0 * np.pi / 3.0) * (chi_hi**3 - chi_lo**3)

    for iz in range(nz_bins):
        z_lo, z_hi = z_edges[iz], z_edges[iz + 1]
        sim_counts[iz] = np.histogram(
            halo_logM[(halo_z >= z_lo) & (halo_z < z_hi)],
            bins=logM_edges)[0]

        # Theory: integrate dn/dlnM over each mass bin at closest z, then multiply by volume
        jz_th = np.argmin(np.abs(z_th - z_cen[iz]))
        dndlnM_z = hmf_Mz[jz_th]  # shape (nM_th,)  units: (Mpc/h)^{-3}
        for im in range(nM_bins):
            lm_lo, lm_hi = logM_edges[im], logM_edges[im + 1]
            mask_M = (logM_th >= lm_lo) & (logM_th < lm_hi)
            if mask_M.sum() > 0:
                # ∫ (dn/dlnM) dlnM  →  number density in this mass bin
                ndens_bin = np.trapz(dndlnM_z[mask_M], np.log(M_th[mask_M]))
                theory_counts_abs[iz, im] = ndens_bin * z_bin_volumes[iz]

    # Also provide renormalized theory (old behavior) as secondary diagnostic
    total_sim = sim_counts.sum()
    total_th_abs = theory_counts_abs.sum()
    if total_th_abs > 0:
        theory_counts_norm = theory_counts_abs * (total_sim / total_th_abs)
    else:
        theory_counts_norm = theory_counts_abs.copy()

    # --- Absolute fractional mismatch ---
    def _frac_mismatch(a, b):
        denom = np.maximum(a, b)
        denom = np.where(denom > 0, denom, 1.0)
        return np.where((a > 5) | (b > 5), np.abs(a - b) / denom, np.nan)

    frac_mismatch_abs = _frac_mismatch(sim_counts, theory_counts_abs)
    frac_mismatch_norm = _frac_mismatch(sim_counts, theory_counts_norm)

    # Ratios (finite where both > 0)
    sim_to_theory_ratio = np.where(
        theory_counts_abs > 0, sim_counts / theory_counts_abs, np.nan)
    theory_to_sim_ratio = np.where(
        sim_counts > 0, theory_counts_abs / sim_counts, np.nan)

    valid_abs = np.isfinite(frac_mismatch_abs)
    if valid_abs.any():
        print(f"[HMF comparison — ABSOLUTE] max frac mismatch = {np.nanmax(frac_mismatch_abs):.3f}, "
              f"mean = {np.nanmean(frac_mismatch_abs):.3f} "
              f"(over {valid_abs.sum()} bins with >5 halos)")
        print(f"  Total sim = {total_sim:.0f}, total theory(abs) = {total_th_abs:.0f}, "
              f"ratio = {total_sim / total_th_abs:.4f}" if total_th_abs > 0 else "")
    else:
        print("[HMF comparison] no populated bins")

    valid_norm = np.isfinite(frac_mismatch_norm)
    if valid_norm.any():
        print(f"[HMF comparison — renorm]   max frac mismatch = {np.nanmax(frac_mismatch_norm):.3f}, "
              f"mean = {np.nanmean(frac_mismatch_norm):.3f}")

    return dict(
        logM_cen=logM_cen, z_cen=z_cen,
        logM_edges=logM_edges, z_edges=z_edges,
        sim_counts=sim_counts,
        theory_counts_abs=theory_counts_abs,
        theory_counts_norm=theory_counts_norm,
        frac_mismatch=frac_mismatch_abs,
        frac_mismatch_abs=frac_mismatch_abs,
        frac_mismatch_norm=frac_mismatch_norm,
        sim_to_theory_ratio=sim_to_theory_ratio,
        theory_to_sim_ratio=theory_to_sim_ratio,
        z_bin_volumes=z_bin_volumes,
    )


# ---------------------------------------------------------------------------
# 1-halo vs 2-halo Cl_gg decomposition
# ---------------------------------------------------------------------------

def compute_Cl_gg_1h_2h(Cls_test):
    """
    Compute separate 1-halo and 2-halo Cl_gg theory curves by repeating
    the Limber integral with only the 1h or 2h P(k,z).

    Returns dict with ell, Cl_1h, Cl_2h, Cl_tot arrays.
    """
    ell_array = np.array(Cls_test.ell_array)
    z_arr = np.array(Cls_test.z_array_for_Cls)
    chi_arr = np.array(Cls_test.chi_array_for_Cls)
    dchi_dz = np.array(Cls_test.dchi_dz_array_for_Cls)
    Wg = np.array(Cls_test.Wg_mat[0])  # lens bin 0

    # Prefactor for galaxy probe: Wg / (dchi_dz * chi^2)
    prefac = Wg / (dchi_dz * chi_arr**2)

    # Get 1h and 2h P(k,z) with suppression factor
    Pgg_1h = np.array(Cls_test.Pgg_1h_kz_mat * Cls_test.Pmm_sup_tot_mat)
    Pgg_2h = np.array(Cls_test.Pgg_2h_kz_mat * Cls_test.Pmm_sup_tot_mat)

    kPk = np.array(Cls_test.kPk_array)
    z_grid = np.array(Cls_test.z_array)

    Cl_1h = np.zeros(len(ell_array))
    Cl_2h = np.zeros(len(ell_array))

    for jl, ell in enumerate(ell_array):
        Pkz_1h_at_ell = np.zeros(len(z_arr))
        Pkz_2h_at_ell = np.zeros(len(z_arr))
        for jz_c, z_c in enumerate(z_arr):
            chi_z = chi_arr[jz_c]
            k_ell = (ell + 0.5) / max(chi_z, 1.0)
            # Interpolate P(k,z) to k=ell/chi
            jz_grid = np.argmin(np.abs(z_grid - z_c))
            Pkz_1h_at_ell[jz_c] = np.exp(np.interp(
                np.log(k_ell), np.log(kPk), np.log(np.clip(Pgg_1h[:, jz_grid], 1e-30, None))))
            Pkz_2h_at_ell[jz_c] = np.exp(np.interp(
                np.log(k_ell), np.log(kPk), np.log(np.clip(Pgg_2h[:, jz_grid], 1e-30, None))))

        integrand_1h = prefac**2 * chi_arr**2 * dchi_dz * Pkz_1h_at_ell
        integrand_2h = prefac**2 * chi_arr**2 * dchi_dz * Pkz_2h_at_ell
        Cl_1h[jl] = np.trapz(integrand_1h, z_arr)
        Cl_2h[jl] = np.trapz(integrand_2h, z_arr)

    return dict(ell=ell_array, Cl_1h=Cl_1h, Cl_2h=Cl_2h, Cl_tot=Cl_1h + Cl_2h)


# ---------------------------------------------------------------------------
# HOD second-moment shot noise
# ---------------------------------------------------------------------------

def compute_hod_shot_noise_Cl(Cls_test, jb=0):
    """
    Compute the Poisson shot noise for Cl_gg from the HOD model:
        Cl_shot = ∫ dz  [Wg(z) / (dchi/dz * chi^2)]^2 * chi^2 * dchi/dz / nbar_3d(z)

    where nbar_3d(z) = ∫ dM (dn/dlnM) * (Ncen + Nsat).

    Also returns the simpler 1/nbar_sr estimate for comparison.
    """
    z_arr = np.array(Cls_test.z_array_for_Cls)
    chi_arr = np.array(Cls_test.chi_array_for_Cls)
    dchi_dz = np.array(Cls_test.dchi_dz_array_for_Cls)
    Wg = np.array(Cls_test.Wg_mat[jb])
    prefac = Wg / (dchi_dz * chi_arr**2)

    # nbar_3d(z) from theory
    z_th = np.array(Cls_test.z_array)
    nbarz_th = np.array(Cls_test.nbarz)  # shape (nz,)
    nbar_at_z = np.interp(z_arr, z_th, nbarz_th)
    nbar_at_z = np.where(nbar_at_z > 1e-15, nbar_at_z, 1e-15)

    integrand = prefac**2 * chi_arr**2 * dchi_dz / nbar_at_z
    Cl_shot_hod = float(np.trapz(integrand, z_arr))

    # Also compute the 2nd-moment contribution: <N(N-1)> - <N>^2 = <Nsat> (Poisson) + 0 (Bernoulli Ncen)
    # For Poisson Nsat: <N^2> - <N>^2 = <N> → extra variance = Ntot / nbar^2
    # This is already captured in 1/nbar_sr; the HOD integral gives the same thing
    # when nbar is computed self-consistently.

    print(f"[HOD shot noise] Cl_shot (theory integral) = {Cl_shot_hod:.4e}")
    return Cl_shot_hod


# ---------------------------------------------------------------------------
# Diagnostic summary table
# ---------------------------------------------------------------------------

def print_diagnostic_summary(nz_mismatch, hod_diag, Cl_shot, Cl_shot_hod,
                             band_ratios_gg, band_ratios_gy=None, hmf_diag=None):
    """Print a concise diagnostic summary table."""
    print("\n" + "=" * 65)
    print("  DIAGNOSTIC SUMMARY")
    print("=" * 65)

    # n(z) consistency
    print(f"  n(z) L1 mismatch           : {nz_mismatch:.4f}")

    # HOD Ncen/Nsat
    valid_cen = np.isfinite(hod_diag['frac_mismatch_Ncen'])
    valid_sat = np.isfinite(hod_diag['frac_mismatch_Nsat'])
    if valid_cen.any():
        print(f"  HOD Ncen max/mean mismatch : {np.nanmax(hod_diag['frac_mismatch_Ncen']):.3f} / "
              f"{np.nanmean(hod_diag['frac_mismatch_Ncen']):.3f}")
    if valid_sat.any():
        print(f"  HOD Nsat max/mean mismatch : {np.nanmax(hod_diag['frac_mismatch_Nsat']):.3f} / "
              f"{np.nanmean(hod_diag['frac_mismatch_Nsat']):.3f}")

    # HMF
    if hmf_diag is not None:
        fm = hmf_diag['frac_mismatch']
        valid_hmf = np.isfinite(fm)
        if valid_hmf.any():
            print(f"  HMF max/mean mismatch      : {np.nanmax(fm):.3f} / {np.nanmean(fm):.3f}")

    # Shot noise
    print(f"  Shot noise (map-level)     : {Cl_shot:.4e}")
    print(f"  Shot noise (HOD integral)  : {Cl_shot_hod:.4e}")

    # Bandpower ratios gg
    print("  --- Cl_gg bandpower ratios (meas/th) ---")
    for (lo, hi), r in band_ratios_gg.items():
        status = "OK" if abs(r - 1.0) < 0.15 else ("HIGH" if r > 1 else "LOW")
        print(f"    ell [{lo:4d}, {hi:4d}]: {r:.3f}  [{status}]")

    # Bandpower ratios gy
    if band_ratios_gy is not None:
        print("  --- Cl_gy bandpower ratios (meas/th) ---")
        for (lo, hi), r in band_ratios_gy.items():
            status = "OK" if abs(r - 1.0) < 0.15 else ("HIGH" if r > 1 else "LOW")
            print(f"    ell [{lo:4d}, {hi:4d}]: {r:.3f}  [{status}]")

    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Galaxy map builder
# ---------------------------------------------------------------------------

def make_galaxy_map(mock_gals, nside, zmin=0.1, zmax=0.31):
    """Build a HEALPix galaxy count map from the mock galaxy catalog."""
    ra = mock_gals[:, 0]
    dec = mock_gals[:, 1]
    z = mock_gals[:, 2]
    sel = (z > zmin*0.98) & (z < zmax*1.02)
    pix = hp.ang2pix(nside, ra[sel], dec[sel], lonlat=True)
    m = np.zeros(12 * nside ** 2, dtype=np.float32)
    np.add.at(m, pix, 1.)
    return m


# ---------------------------------------------------------------------------
# Galaxy number density vs redshift
# ---------------------------------------------------------------------------

def compute_ngal_vs_z(gal_z_all, is_cen_all, Prof_test, Cls_test,
                      z_edges=np.linspace(0.0, 0.5, 15)):
    """
    Compute measured and theoretical galaxy number densities in redshift bins.
    Returns dict with keys: zcen, ngal, ncen, nsat, ngal_th, ncen_th, nsat_th.
    """
    zmin_arr = z_edges[:-1]
    zmax_arr = z_edges[1:]
    zcen = 0.5 * (zmin_arr + zmax_arr)
    nbins = len(zcen)

    ngal = np.zeros(nbins)
    ncen = np.zeros(nbins)
    nsat = np.zeros(nbins)
    ngal_th = np.zeros(nbins)
    ncen_th = np.zeros(nbins)
    nsat_th = np.zeros(nbins)

    gal_z_np = np.array(gal_z_all)

    for jz in range(nbins):
        chi_min = bkgrd.radial_comoving_distance(Prof_test.cosmo_jax, 1 / (1 + zmin_arr[jz]))[0]
        chi_max = bkgrd.radial_comoving_distance(Prof_test.cosmo_jax, 1 / (1 + zmax_arr[jz]))[0]
        vol = (4 / 3) * jnp.pi * (chi_max ** 3 - chi_min ** 3)

        sel = (gal_z_np > zmin_arr[jz]) & (gal_z_np < zmax_arr[jz])
        is_cen_sel = is_cen_all[sel]
        ngal[jz] = float(sel.sum() / vol)
        ncen[jz] = float((is_cen_sel > 0).sum() / vol)
        nsat[jz] = float((is_cen_sel < 1).sum() / vol)

        jz_th = np.argmin(np.abs(Cls_test.z_array - zcen[jz]))
        Ncen_th = Cls_test.Ncen_mat[jz_th]
        Nsat_th = Cls_test.Nsat_mat[jz_th]
        dndlogM = Cls_test.hmf_Mz_mat[jz_th]
        logM = jnp.log(Cls_test.M_array)
        ngal_th[jz] = jsi.trapezoid(dndlogM * (Ncen_th + Nsat_th), x=logM, axis=0)
        ncen_th[jz] = jsi.trapezoid(dndlogM * Ncen_th, x=logM, axis=0)
        nsat_th[jz] = jsi.trapezoid(dndlogM * Nsat_th, x=logM, axis=0)

    return dict(zcen=zcen, ngal=ngal, ncen=ncen, nsat=nsat,
                ngal_th=ngal_th, ncen_th=ncen_th, nsat_th=nsat_th)


# ---------------------------------------------------------------------------
# Multi-snapshot map stacking
# ---------------------------------------------------------------------------

def stack_snapshot_maps(snap_range, snap_num_all, zval_all, nside, jdevice, Ndevices,
                        ldir_maps='/mnt/ceph/users/spandey/paste_godmax/GODMAX/results/backlight_pkdgrav/'):
    """Stack maps across multiple snapshots. Returns dict of stacked maps."""
    from tqdm import tqdm
    npix = 12 * nside ** 2
    maps = {k: np.zeros(npix, dtype=np.float32)
            for k in ['rhom', 'ymap', 'ksz', 'tau', 'gal']}

    for snap_num in tqdm(snap_range):
        idx = np.where(snap_num_all == snap_num)[0]
        zval = zval_all[idx][0]
        fname = f'{ldir_maps}/allmaps_sim_B12_nside{nside}_z{zval:.3f}_split{jdevice}of{Ndevices}.pkl'
        saved = pk.load(open(fname, 'rb'))

        gal_ra = saved['mock_gals_all'][0][:, 0]
        gal_dec = saved['mock_gals_all'][0][:, 1]
        pix_gal = hp.ang2pix(nside, gal_ra, gal_dec, lonlat=True)
        gal_map = np.zeros(npix, dtype=np.float32)
        np.add.at(gal_map, pix_gal, 1.)

        maps['rhom'] += saved['map_rhom']
        maps['ymap'] += saved['map_ymap']
        maps['ksz'] += saved['map_ksz']
        maps['tau'] += saved['map_tau']
        maps['gal'] += gal_map

    return maps

# ---------------------------------------------------------------------------
# Calculating lensing convergence (kappa) maps
# ---------------------------------------------------------------------------

def compute_kappa_dmo_and_weight(
    sim_file_path, snap_num_all, zval_all, zmin, zmax,
    Prof_test, nside, part_mass, chi_CMB, cosmo_params_dict,
    smooth_sigma=None,
):
    """
    Compute the DMO kappa map and the effective baryonic lensing weight.

    Both outputs are IDENTICAL for every sample in an SBI suite that shares
    the same N-body lightcone and cosmology. Call this once and cache the
    results; do not recompute per sample.

    The DMO kappa is built by summing per-snapshot contributions:
        delta_kappa_DMO(z) = weight_k(z) * delta_particle(z)

    where weight_k(z) = (3/2)(H0/c)^2 Om0
                        * [(chi_CMB - chi) / chi_CMB] * (1+z) * chi * dchi

    and delta_particle is the differential N-body mass sheet converted to a
    density contrast using the sky-coverage-corrected mean mass per pixel
    (mmpp). The f_sky correction prevents a systematic negative mean kappa
    in lightcones that do not cover the full sky.

    Args:
        sim_file_path:     path containing compressed_massMaps/ subdirectory
        snap_num_all:      all snapshot numbers from zlist.txt
        zval_all:          corresponding redshifts
        zmin, zmax:        redshift shell boundaries
        Prof_test:         Profiles object with .cosmo_jax and .get_Ez(z)
        nside:             HEALPix resolution
        part_mass:         N-body particle mass [Msun/h]
        chi_CMB:           comoving distance to CMB [Mpc/h]
        cosmo_params_dict: dict containing at minimum 'Om0'
        smooth_sigma:      Gaussian smoothing of mass sheets [radians].
                           Defaults to one HEALPix pixel FWHM → sigma.

    Returns:
        map_kappa_dmo : np.ndarray, shape (12*nside**2,), dtype float64
        weight_eff    : float
            Effective lensing weight for the baryonic correction:
                sum_snaps(weight_k / mmpp) / n_snaps
    """
    c_light   = 299792.458
    H0_over_h = 100.0
    Om0       = cosmo_params_dict['Om0']
    npix      = 12 * nside ** 2
    rho_m     = Prof_test.get_rho_m(0.0)

    if smooth_sigma is None:
        fwhm_pix     = hp.nside2resol(nside, arcmin=False)
        smooth_sigma = fwhm_pix / (8.0 * np.log(2.0)) ** 0.5

    in_shell    = (zval_all >= zmin) & (zval_all <= zmax)
    snaps_shell = snap_num_all[in_shell]
    n_snaps     = len(snaps_shell)
    if n_snaps == 0:
        raise ValueError(
            f"No snapshots found in z=[{zmin}, {zmax}]. "
            f"Check zmin/zmax against zlist.txt."
        )

    # Sort all snapshots by redshift for differential shell subtraction
    sorted_snaps = snap_num_all[np.argsort(zval_all)]
    dz_snap      = (zmax - zmin) / n_snaps

    map_kappa_dmo        = np.zeros(npix, dtype=np.float64)
    weight_over_mmpp_sum = 0.0

    print(f"Computing DMO kappa from {n_snaps} snapshots "
          f"in z=[{zmin:.3f}, {zmax:.3f}]...")

    for snap_num in snaps_shell:
        ind  = np.where(snap_num_all == snap_num)[0]
        zval = float(zval_all[ind][0])

        chi_v     = float(bkgrd.radial_comoving_distance(
            Prof_test.cosmo_jax, 1.0 / (1.0 + zval)
        )[0])
        dchi_snap = (c_light / (H0_over_h * float(Prof_test.get_Ez(zval)))) * dz_snap
        r_lo      = chi_v - dchi_snap / 2.0
        r_hi      = chi_v + dchi_snap / 2.0

        weight_k = (
            1.5 * (H0_over_h / c_light)**2 * Om0
            * ((chi_CMB - chi_v) / chi_CMB)
            * (1.0 + zval) * chi_v * dchi_snap
        )

        # --- Load differential N-body mass sheet FIRST ---
        # (mmpp depends on f_sky which depends on map_shell)
        path_tot = (f'{sim_file_path}/compressed_massMaps/'
                    f'massSheet_tot_z_{int(snap_num)}.fits.gz')
        map_tot  = hp.read_map(path_tot, verbose=False)

        curr_idx = int(np.where(sorted_snaps == snap_num)[0][0])
        if curr_idx > 0:
            path_prev = (f'{sim_file_path}/compressed_massMaps/'
                         f'massSheet_tot_z_{int(sorted_snaps[curr_idx - 1])}.fits.gz')
            map_shell = map_tot - hp.read_map(path_prev, verbose=False)
        else:
            map_shell = map_tot

        # f_sky correction: lightcones covering < full sky have empty pixels
        # that would otherwise make delta = -1 everywhere, pulling mean kappa
        # negative. Dividing by f_sky * npix gives the correct mean density
        # over the *filled* volume. For full-sky sims f_sky = 1 and mmpp is
        # unchanged.
        n_filled = float(np.sum(map_shell > 0))
        f_sky    = n_filled / npix
        mmpp     = float(
            rho_m * (4.0 / 3.0) * np.pi * (r_hi**3 - r_lo**3) / (f_sky * npix)
        )

        # Regrade to working nside, convert particle counts → density contrast
        map_shell_ds = hp.ud_grade(
            np.array(map_shell * part_mass, dtype=np.float64), nside, power=-2
        )
        delta_sheet  = map_shell_ds / mmpp - 1.0
        delta_smooth = hp.smoothing(
            np.array(delta_sheet, dtype=np.float64), sigma=smooth_sigma, verbose=False
        )

        map_kappa_dmo        += weight_k * delta_smooth
        weight_over_mmpp_sum += weight_k / mmpp

        print(f"  z={zval:.3f}: chi={chi_v:.1f} Mpc/h, "
              f"weight_k={weight_k:.4e}, f_sky={f_sky:.4f}, mmpp={mmpp:.4e}")

    # weight_eff is the mean per-snapshot lensing weight in units of 1/mmpp.
    # (map_rhom_dmb - map_rhom_dmo) accumulates over all snapshots, so dividing
    # weight_over_mmpp_sum by n_snaps gives the correct effective weight.
    weight_eff = weight_over_mmpp_sum / n_snaps
    return map_kappa_dmo, weight_eff


def apply_baryonic_kappa_correction(map_kappa_dmo, weight_eff,
                                    map_rhom_dmb, map_rhom_dmo):
    """
    Apply the baryonic correction to the precomputed DMO kappa map.

    This is the only per-sample call needed in an SBI suite. The DMO kappa
    and weight_eff are constants across samples and should be loaded from
    cache rather than recomputed.

        kappa = kappa_DMO + weight_eff * (map_rhom_dmb - map_rhom_dmo)

    Args:
        map_kappa_dmo : np.ndarray, shape (12*nside**2,), dtype float64
            From compute_kappa_dmo_and_weight().
        weight_eff    : float
            From compute_kappa_dmo_and_weight().
        map_rhom_dmb  : np.ndarray — baryonified halo matter map
        map_rhom_dmo  : np.ndarray — DMO halo matter map

    Returns:
        map_kappa : np.ndarray, shape (12*nside**2,), dtype float32
    """
    baryonic_delta = (
        np.array(map_rhom_dmb, dtype=np.float64)
        - np.array(map_rhom_dmo, dtype=np.float64)
    )
    map_kappa = map_kappa_dmo + weight_eff * baryonic_delta

    dmo_mean = float(np.abs(map_kappa_dmo).mean())
    bar_mean = float(np.abs(weight_eff * baryonic_delta).mean())
    print(f"\nKappa diagnostics:")
    print(f"  DMO sheet:           mean|kappa| = {dmo_mean:.4e}")
    print(f"  Baryonic correction: mean|kappa| = {bar_mean:.4e}")
    print(f"  Baryonic/DMO ratio:  {bar_mean / (dmo_mean + 1e-30):.3f}")
    print(f"  Total kappa:         mean={map_kappa.mean():.4e}, "
          f"max={np.abs(map_kappa).max():.4e}")

    return map_kappa.astype(np.float32)


def compute_kappa_map(
    sim_file_path, snap_num_all, zval_all, zmin, zmax,
    Prof_test, nside, part_mass, chi_CMB, cosmo_params_dict,
    map_rhom_dmb, map_rhom_dmo, smooth_sigma=None,
    dmo_cache_path=None, weight_cache_path=None,
):
    """
    Compute the full weak lensing convergence map.

    Thin wrapper around compute_kappa_dmo_and_weight and
    apply_baryonic_kappa_correction. Optionally caches the DMO component
    to disk so it is not recomputed for every sample in an SBI suite.

    Cache behaviour:
        If dmo_cache_path and weight_cache_path are provided:
        - On first call: compute DMO kappa and weight, save to those paths.
        - On subsequent calls: load from disk, skip FITS I/O entirely.
        If paths are None: always recompute (original behaviour).

    Args:
        sim_file_path, snap_num_all, zval_all, zmin, zmax,
        Prof_test, nside, part_mass, chi_CMB, cosmo_params_dict,
        map_rhom_dmb, map_rhom_dmo, smooth_sigma:
            See compute_kappa_dmo_and_weight for full documentation.
        dmo_cache_path  : str or None — path to save/load map_kappa_dmo.npy
        weight_cache_path : str or None — path to save/load weight_eff.npy

    Returns:
        map_kappa : np.ndarray, shape (12*nside**2,), dtype float32
    """
    use_cache = (dmo_cache_path is not None) and (weight_cache_path is not None)

    if use_cache and os.path.exists(dmo_cache_path) and os.path.exists(weight_cache_path):
        print("Loading cached DMO kappa from disk...")
        map_kappa_dmo = np.load(dmo_cache_path)
        weight_eff    = float(np.load(weight_cache_path))
    else:
        map_kappa_dmo, weight_eff = compute_kappa_dmo_and_weight(
            sim_file_path, snap_num_all, zval_all, zmin, zmax,
            Prof_test, nside, part_mass, chi_CMB, cosmo_params_dict,
            smooth_sigma=smooth_sigma,
        )
        if use_cache:
            np.save(dmo_cache_path,    map_kappa_dmo)
            np.save(weight_cache_path, np.array(weight_eff))
            print(f"Cached DMO kappa → {dmo_cache_path}")
            print(f"Cached weight_eff → {weight_cache_path}")

    return apply_baryonic_kappa_correction(
        map_kappa_dmo, weight_eff, map_rhom_dmb, map_rhom_dmo)
