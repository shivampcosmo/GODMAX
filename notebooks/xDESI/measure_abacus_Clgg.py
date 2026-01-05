import matplotlib.pyplot as plt
import asdf
import healpy as hp
import numpy as np
import pymaster as nmt
from tqdm import tqdm
import sys, os
import jax_cosmo.background as bkgrd
from jax_cosmo import Cosmology
from jax_cosmo.background import angular_diameter_distance, radial_comoving_distance

cosmo_params = {'Om0': 0.3137, 'Ob0': 0.0493, 'H0': 67.36, 'sigma8': 0.8079, 'ns': 0.9649, 'w0': -1.0}
cosmo_jax = Cosmology(
            Omega_c=cosmo_params['Om0'] - cosmo_params['Ob0'],
            Omega_b=cosmo_params['Ob0'],
            h=cosmo_params['H0'] / 100.,
            sigma8=cosmo_params['sigma8'],
            n_s=cosmo_params['ns'],
            Omega_k=0.,
            w0=cosmo_params['w0'],
            wa=0.
            )

z_array = np.linspace(0, 1.5, 200)
z_centers = 0.5 * (z_array[1:] + z_array[:-1])

nside = 4096
zedges_comoving = np.linspace(0, 1.5, 51)
zcens_comoving = 0.5*(zedges_comoving[1:] + zedges_comoving[:-1])

chi_edges_comoving = radial_comoving_distance(cosmo_jax, 1/(1 + zedges_comoving))
chi_min_comoving = chi_edges_comoving[:-1]
chi_max_comoving = chi_edges_comoving[1:]


ji = int(sys.argv[1])
print(ji)
if ji<10:
    ldir = f'/mnt/ceph/users/spandey/abacus/AbacusSummit_base_c000_ph00{ji}'
else:
    ldir = f'/mnt/ceph/users/spandey/abacus/AbacusSummit_base_c000_ph0{ji}'
zvals = [[0.3, 0.35, 0.4], [0.45, 0.5, 0.575], [0.65, 0.725, 0.8]]

fsky = (1/8.)

comoving_volume_shells = fsky * (4.0/3.0) * np.pi * (chi_max_comoving**3 - chi_min_comoving**3)
nbar_comoving = np.zeros_like(zcens_comoving) + 1e-8


nz_gal_all = {}
Cl_gg_all = {}
nz_gal_all['z_array'] = z_centers

Ngal_all = {}

zmin_all = {}
zmax_all = {}

zedges_comoving = np.linspace(0, 1.5, 51)
zcens_comoving = 0.5*(zedges_comoving[1:] + zedges_comoving[:-1])

chi_edges_comoving = radial_comoving_distance(cosmo_jax, 1/(1 + zedges_comoving))
chi_min_comoving = chi_edges_comoving[:-1]
chi_max_comoving = chi_edges_comoving[1:]
fsky = (1/8.)
comoving_volume_shells = fsky * (4.0/3.0) * np.pi * (chi_max_comoving**3 - chi_min_comoving**3)

nbar_comoving = np.zeros_like(zcens_comoving) + 1e-8

for jz, zval_group in tqdm(enumerate(zvals)):
    gal_ra, gal_dec, gal_z = [], [], []
    rand_ra, rand_dec, rand_z = [], [], []
    for zval in tqdm(zval_group):
        
        fn = f"{ldir}/z{zval:.3f}/catalog_DESI_LRG.asdf"
        f = asdf.open(fn)

        gal_ra.extend(f['data']['RA'])
        gal_dec.extend(f['data']['DEC'])
        gal_z.extend(f['data']['Z_COSMO'])

        rand_ra.extend(f['data']['RAND_RA'])
        rand_dec.extend(f['data']['RAND_DEC'])
        rand_z.extend(f['data']['RAND_Z'])

    hist_z = np.histogram(gal_z, bins=z_array, density=True)[0]
    hist_z_rand = np.histogram(rand_z, bins=z_array, density=True)[0]

    zmin_all[f'z{zval_group[0]:.3f}_{zval_group[-1]:.3f}'] = np.min(gal_z)
    zmax_all[f'z{zval_group[0]:.3f}_{zval_group[-1]:.3f}'] = np.max(gal_z)


    for jzc in range(len(zcens_comoving)):
        indsel_jsz = np.where((gal_z >= zedges_comoving[jzc]) & (gal_z < zedges_comoving[jzc+1]))[0]
        nbar_comoving[jzc] += len(indsel_jsz)/comoving_volume_shells[jzc]

    nz_gal_all[f'z{zval_group[0]:.3f}_{zval_group[-1]:.3f}'] = hist_z
    Ngal_all[f'z{zval_group[0]:.3f}_{zval_group[-1]:.3f}'] = len(gal_z)

    ipix = hp.ang2pix(nside, gal_ra, gal_dec, lonlat=True)
    npix = hp.nside2npix(nside)

    nmap = np.bincount(ipix, minlength=npix)

    ipix_rand = hp.ang2pix(nside, rand_ra, rand_dec, lonlat=True)
    npix_rand = hp.nside2npix(nside)

    nmap_rand = np.bincount(ipix_rand, minlength=npix_rand)

    print('Running NaMaster for Cl_gg...')
    lmax = nside
    b = nmt.NmtBin.from_lmax_linear(lmax, nlb=100)
    leff = b.get_effective_ells()
    pos_data = np.array([f['data']['RA'], f['data']['DEC']])
    w_data = np.ones_like(f['data']['RA'])
    pos_ran = np.array([f['data']['RAND_RA'], f['data']['RAND_DEC']])
    w_ran = np.ones_like(f['data']['RAND_RA'])
    f_cat = nmt.NmtFieldCatalogClustering(pos_data, w_data, pos_ran, w_ran, lmax=lmax, lonlat=True)
    w_cat = nmt.NmtWorkspace.from_fields(f_cat, f_cat, b)
    pcl = nmt.compute_coupled_cell(f_cat, f_cat)
    cl_cat = w_cat.decouple_cell(pcl)

    Cl_gg_all['l_array'] = leff
    Cl_gg_all[f'z{zval_group[0]:.3f}_{zval_group[-1]:.3f}'] = cl_cat[0]

for jz, zval_group in tqdm(enumerate(zvals)):
    gal_ra, gal_dec, gal_z = [], [], []
    for zval in tqdm(zval_group):
        
        fn = f"{ldir}/z{zval:.3f}/catalog_DESI_LRG.asdf"
        f = asdf.open(fn)

        gal_ra.extend(f['data']['RA'])
        gal_dec.extend(f['data']['DEC'])
        gal_z.extend(f['data']['Z_COSMO'])

    for jzc in range(len(zcens_comoving)):
        indsel_jsz = np.where((gal_z >= zedges_comoving[jzc]) & (gal_z < zedges_comoving[jzc+1]))[0]
        nbar_comoving[jzc] += len(indsel_jsz)/comoving_volume_shells[jzc]


import pickle as pk
with open(f'/mnt/ceph/users/spandey/abacus/abacus_LRG_nz_Clgg_v4_deltaell_100_AbacusSummit_base_c000_ph00{ji}.pkl', 'wb') as f:
    pk.dump({'nz_gal_all': nz_gal_all, 'Cl_gg_all': Cl_gg_all,'Ngal_all':Ngal_all,  
            'nbar_comoving': nbar_comoving, 'zcens_comoving': zcens_comoving,
            'zmin_all': zmin_all, 'zmax_all': zmax_all, 'zvals': zvals}, f)




