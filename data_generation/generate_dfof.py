import numpy as np
import h5py

# list of all data files
dat_files = ['/media/storage/DATA/lfads_export/f1_221027.h5',
             '/media/storage/DATA/lfads_export/f1_221103.h5',
             '/media/storage/DATA/lfads_export/f2_221103.h5',
             '/media/storage/DATA/lfads_export/f3_221103.h5']

filepath = dat_files[0]
data_folder = '/home/tomh/Documents/projects/czebra/test_notebooks/data/'
filename = filepath.split('/')[-1][:-3] # fish and date only
filename_dfof = f'{filename[-9:]}_dfof.npz'



with h5py.File(filepath, 'r') as f:
    neural = f['rois']['dfof']
    
    # save
    np.savez(f"{data_folder}{filename_dfof}", neural=neural)