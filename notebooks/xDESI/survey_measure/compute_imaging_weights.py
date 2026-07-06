# Example script that computes the imaging weights for the Main/Extended LRG sample

from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
from astropy.table import Table, vstack, hstack
import fitsio

import yaml

include_ebv = True
sample = 'main'
# sample = 'extended'

if sample=='main':
    if include_ebv:
        weights_path = 'catalogs/imaging_weights/main_lrg_linear_coeffs_pz.yaml'
        output_fn = 'catalogs/more/dr9_lrg_pzbins-weights.fits'
    else:
        weights_path = 'catalogs/imaging_weights/main_lrg_linear_coeffs_pz_no_ebv.yaml'
        output_fn = 'catalogs/more/dr9_lrg_pzbins-weights_no_ebv.fits'
    # Load LRG catalog
    cat = Table(fitsio.read('catalogs/dr9_lrg_pzbins.fits'))
    more_2 = Table(fitsio.read('catalogs/more/dr9_lrg_more_2.fits',
                   columns=['GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']))
    print(len(cat)==len(more_2))
    cat = hstack([cat, more_2], join_type='exact')

elif sample=='extended':
    if include_ebv:
        weights_path = 'catalogs/imaging_weights/extended_lrg_linear_coeffs_pz.yaml'
        output_fn = 'catalogs/more/dr9_extended_lrg_pzbins-weights.fits'
    else:
        weights_path = 'catalogs/imaging_weights/extended_lrg_linear_coeffs_pz_no_ebv.yaml'
        output_fn = 'catalogs/more/dr9_extended_lrg_pzbins-weights_no_ebv.fits'
    # Load LRG catalog
    cat = Table(fitsio.read('catalogs/dr9_extended_lrg_pzbins.fits'))
    more_2 = Table(fitsio.read('catalogs/more/dr9_extended_lrg_more_2.fits',
                   columns=['GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']))
    print(len(cat)==len(more_2))
    cat = hstack([cat, more_2], join_type='exact')

# Convert depths to units of magnitude
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cat['galdepth_gmag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['GALDEPTH_G'])))-9) - 3.214*cat['EBV']
    cat['galdepth_rmag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['GALDEPTH_R'])))-9) - 2.165*cat['EBV']
    cat['galdepth_zmag_ebv'] = -2.5*(np.log10((5/np.sqrt(cat['GALDEPTH_Z'])))-9) - 1.211*cat['EBV']

cat['weight'] = 0.

for field in ['north', 'south']:

    if field=='south':
        photsys = 'S'
    elif field=='north':
        photsys = 'N'

    # Load weights
    with open(weights_path, "r") as f:
        linear_coeffs = yaml.safe_load(f)

    for bin_index in range(1, 5):  # 4 bins

        mask_bin = cat['PHOTSYS']==photsys
        mask_bin &= cat['pz_bin']==bin_index
        cat1 = cat[mask_bin].copy()

        xnames_fit = list(linear_coeffs['south_bin_1'].keys())
        xnames_fit.remove('intercept')
        # Assign zero weights to objects with invalid imaging properties
        # (their fraction should be negligibly small)
        mask_bad = np.full(len(cat1), False)
        for col in xnames_fit:
            mask_bad |= ~np.isfinite(cat1[col])
        if np.sum(mask_bad)!=0:
            print('{} invalid objects'.format(np.sum(mask_bad)))

        weights = np.zeros(len(cat1))
        bin_str = '{}_bin_{}'.format(field, bin_index)

        # create array of coefficients, with the first coefficient being the intercept
        coeffs = np.array([linear_coeffs[bin_str]['intercept']]+[linear_coeffs[bin_str][xname] for xname in xnames_fit])

        data = np.column_stack([cat1[~mask_bad][xname] for xname in xnames_fit])
        # create 2-D array of imaging properties, with the first columns being unity
        data1 = np.insert(data, 0, 1., axis=1)
        # wt = coeff0 + coeff1 * rand['EBV'] + coeff2 * rand['PSFSIZE_G'] + ...
        weights[~mask_bad] = 1/np.dot(coeffs, data1.T)  # 1/predicted_density as weights for objects

        cat['weight'][mask_bin] = weights

cat = cat[['weight']]
cat.write(output_fn)

