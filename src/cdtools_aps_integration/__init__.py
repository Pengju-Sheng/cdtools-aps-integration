from ._version import __version__

# Major dependencies
import os
import numpy as np
import torch as t
import h5py
from tqdm import tqdm

# Custom dependencies
import cdtools


def load_dataset(
        scan_id,
        base_path='/net/micdata/data2/12IDC/2026_Data/2026_2/02_levitan/preproc',
        distance=None,
        pixel_size=172e-6,
        projection_num=None,
        asize=None,
        original_asize=1024,
        center=None,
        swap_flipi_flipj=[0,0,0],
        flipx_flipy=[1,1],
        verbose=False,
):
    swap, flipi, flipj = swap_flipi_flipj
    flipx, flipy = flipx_flipy

    base_folder = f'{base_path}/S{scan_id:04d}/'
    pattern_filename = f'{base_folder}/data_roi0_Ndp{original_asize}_dp.hdf5'
    para_filename = f'{base_folder}/data_roi0_Ndp{original_asize}_para.hdf5'

    if verbose:
        print('Loading data from file', pattern_filename)

    # load the patterns and mask
    with h5py.File(pattern_filename, 'r') as f:
        patterns = f["dp"][()]
        mask = f["det_pixel_mask"][()]

    # load the translations, energy, and distance
    with h5py.File(para_filename, 'r') as f:
        x = f['ppX'][()] # units in metres
        y = f['ppY'][()] # units in metres
        energy = f['energy'][()] # units in keV
        distance = f['detector_distance'][()] # units in metres

        if flipx:
            x *= -1
        if flipy:
            y *= -1

    wavelength = 1.2398419843320026e-09 / energy # units in metres
        
    translations = np.stack([x,y,0*x], axis=1)
    
    if swap:
        mask = mask.transpose(0,1)
        patterns = patterns.transpose(1,2)
    if flipi:
        mask = mask.flip(0)
        patterns = patterns.flip(1)
    if flipj:
        mask = mask.flip(1)
        patterns = patterns.flip(2)

    pattern_shape = np.asarray(patterns[0].shape)
    
    if center is None:
        center = pattern_shape // 2
        if verbose:
            print(f'Center was automatically set to [{center[0]},{center[1]}]')
    center = np.asarray(center)

    if np.any(center < 0) or np.any(center >= pattern_shape):
        raise ValueError(f'Center [{center[0]},{center[1]}] is not within the detector shape of [{final_pattern_shape[0]},{final_pattern_shape[1]}].')
    
    if asize is None:
        # This finds the largest crop which sets `center` to the zero
        # frequency pixel of an fftshifted array.
        left = center
        right = pattern_shape - center
        left = np.minimum(left, right)
        right = np.minimum(left + 1, right)
        asize = left + right
        if verbose:
            print(f'Asize was automatically set to [{asize[0]},{asize[1]}]')

    asize = np.asarray(asize)

    start = center - asize // 2
    end = start + asize

    if verbose:
        print(f'Using detector crop [{start[0]}:{end[0]},{start[1]}:{end[1]}]')
    
    if np.any(start < 0) or np.any((end - 1) > pattern_shape):
        raise ValueError(f'Crop region with center [{center[0]},{center[1]}] and asize [{asize[0]},{asize[1]}] resolves to [{start[0]}:{end[0]},{start[1]}:{end[1]}] and exceeds detector size of [{pattern_shape[0]},{pattern_shape[1]}].')

    patterns = patterns[...,start[0]:end[0],start[1]:end[1]]
    mask = mask[start[0]:end[0],start[1]:end[1]]

    if projection_num is not None:
        if projection_num < 0 or projection_num >= patterns.shape[0]:
            raise ValueError(f'Projection number {projection_num} is out of bounds for {patterns.shape[0]} projections.')
        patterns = patterns[projection_num:projection_num+1]
        translations = translations[projection_num:projection_num+1]
    
    if distance is None:
        raise NotImplementedError('Auto-loading distance from saved metadata is not implemented')
    
    detector_geometry = {
        'basis' : t.as_tensor(np.array([[0,-pixel_size],
                                        [-pixel_size,0],
                                        [0,0]], dtype=np.float32)),
        'distance' : distance
    }

    mask = t.as_tensor(mask)
    patterns = t.as_tensor(patterns)
    translations = t.as_tensor(translations)
    wavelength = t.as_tensor(wavelength)

    dataset = cdtools.datasets.Ptycho2DDataset(
        translations,
        patterns,
        wavelength=wavelength,
        detector_geometry=detector_geometry,
        mask=mask,
    )

    return dataset

    
# wavelength is in instrument/positioners/energy_dcm/
# 
