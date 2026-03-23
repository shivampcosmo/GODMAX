import os
import numpy as np
import h5py as h5
import healpy as hp
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================
SIM_DIR = '/work/hdd/bdne/spandey3/backlight/fiducial/100'
OUT_DIR = '/work/hdd/bdne/aacharya2/GODMAX/data/backlight'
os.makedirs(OUT_DIR, exist_ok=True)

Z_MIN = 0.3
Z_MAX = 0.5
M_CUT = 10**12.75
OUT_FILE = os.path.join(OUT_DIR, 'halo_catalog_Mlim_1e12.75_zlim_0.3_0.5.h5')

# =============================================================================
# YOUR EXACT HELPER FUNCTIONS
# =============================================================================
def open_data(file, Mlim=1e12):
    # Relies on global ldir being set in the loop
    df = h5.File(ldir+file, 'r')
    M200c = df['M200c'][()]
    X, Y, Z = df['X'][()], df['Y'][()], df['Z'][()]
    VX, VY, VZ = df['VX'][()], df['VY'][()], df['VZ'][()]
    
    # Check for non-empty dataset
    if (M200c.shape) is not None:
        indsel = np.where(M200c > Mlim)[0]
        X_val = X[indsel]
        Y_val = Y[indsel]
        Z_val = Z[indsel]
        M200c_val = M200c[indsel]
        VX_val = VX[indsel]
        VY_val = VY[indsel]
        VZ_val = VZ[indsel]
        Vlos = (VX_val*X_val + VY_val*Y_val + VZ_val*Z_val)/np.sqrt(X_val**2 + Y_val**2 + Z_val**2)
    else:
        X_val = np.array([]); Y_val = np.array([]); Z_val = np.array([]); M200c_val = np.array([])
        VX_val = np.array([]); VY_val = np.array([]); VZ_val = np.array([]); Vlos = np.array([])
    df.close()
    return (X_val, Y_val, Z_val, Vlos, M200c_val)

def concatenate_data(results):
    lengths = np.array([len(result[0]) for result in results])
    total_length = lengths.sum()
    X_all = np.empty(total_length); Y_all = np.empty(total_length); Z_all = np.empty(total_length)
    Vlos_all = np.empty(total_length); M200_all = np.empty(total_length)
    end_ind_all = np.cumsum(lengths)
    start_ind_all = np.roll(end_ind_all, 1); start_ind_all[0] = 0
    for i, (start, end, result) in enumerate(zip(start_ind_all, end_ind_all, results)):
        X_all[start:end] = result[0]; Y_all[start:end] = result[1]; Z_all[start:end] = result[2]
        Vlos_all[start:end] = result[3]; M200_all[start:end] = result[4]
    return X_all, Y_all, Z_all, Vlos_all, M200_all


# =============================================================================
# MULTIPROCESSING INITIALIZER
# =============================================================================
def init_worker(shared_ldir):
    """Injects the current snapshot directory into the worker's global namespace"""
    global ldir
    ldir = shared_ldir

# =============================================================================
# MAIN AGGREGATION LOOP
# =============================================================================
def main():
    print("1. Loading Redshift List...")
    zlist = np.loadtxt(f'{SIM_DIR}/zlist.txt')
    snap_num_all, zval_all = zlist[:, 0].astype(int), zlist[:, 1]
    
    # Filter by redshift bounds
    mask = (zval_all >= Z_MIN) & (zval_all <= Z_MAX)
    snaps_in_shell = snap_num_all[mask]
    zvals_in_shell = zval_all[mask]
    
    ra_master, dec_master, z_master, M200c_master, vlos_master = [], [], [], [], []
    total_halos = 0
    
    open_data_partial = partial(open_data, Mlim=M_CUT)
    
    for snap_num, zval in zip(snaps_in_shell, zvals_in_shell):
        current_ldir = os.path.join(SIM_DIR, 'halos', str(snap_num)) + '/'
        if not os.path.exists(current_ldir):
            continue
            
        # Because open_data relies on ldir+file, we just pass the filenames, not absolute paths
        h5_files = [f for f in os.listdir(current_ldir) if f.endswith('.h5')]
        
        # Pass current_ldir to workers so your global `ldir` logic works seamlessly
        with Pool(cpu_count(), initializer=init_worker, initargs=(current_ldir,)) as pool:
            results = pool.map(open_data_partial, h5_files)
            
        X, Y, Z, vlos, M200c = concatenate_data(results)
        
        if len(M200c) > 0:
            # Convert Cartesian to RA/Dec
            ra, dec = hp.vec2ang(np.array([X, Y, Z]).T, lonlat=True)
            ra = np.clip(ra, 0, 360)
            dec = np.clip(dec, -90, 90)
            
            # Create z-array for this slice
            z_array = np.full_like(ra, zval)
            
            ra_master.append(ra)
            dec_master.append(dec)
            z_master.append(z_array)
            M200c_master.append(M200c)
            vlos_master.append(vlos)
            
            total_halos += len(M200c)
            print(f"   -> z={zval:.3f} | Added {len(M200c)} halos")

    print("\n3. Stitching final arrays...")
    if total_halos > 0:
        ra_final = np.concatenate(ra_master)
        dec_final = np.concatenate(dec_master)
        z_final = np.concatenate(z_master)
        M200c_final = np.concatenate(M200c_master)
        vlos_final = np.concatenate(vlos_master)
        
        print(f"\n4. Saving {total_halos} total halos to HDF5...")
        with h5.File(OUT_FILE, 'w') as f:
            f.create_dataset('ra', data=ra_final)
            f.create_dataset('dec', data=dec_final)
            f.create_dataset('z', data=z_final)
            f.create_dataset('M200c', data=M200c_final)
            f.create_dataset('vlos', data=vlos_final)
            
        print(f"Done! Saved to: {OUT_FILE}")
    else:
        print("No halos found in the specified ranges.")

if __name__ == '__main__':
    main()
