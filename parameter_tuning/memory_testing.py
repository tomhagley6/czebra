### Incrementally load more data and fit models ###

import cebra
import numpy as np
import h5py
import sys

## definitions ##

# list of all data files
dat_files = ['/media/storage/DATA/lfads_export/f1_221027.h5',
             '/media/storage/DATA/lfads_export/f1_221103.h5',
             '/media/storage/DATA/lfads_export/f2_221103.h5',
             '/media/storage/DATA/lfads_export/f3_221103.h5']

# global params
global FILENAME
global ITERS

FILENAME = dat_files[0]
ITERS = 1000                                   # kept low for speed

# paths
filename = FILENAME
filename_trunc = filename.split('/')[-1][:-3]  # fish and date only
data_folder = 'data/'
filename_dfof = f'{filename[-12:-3]}_dfof.npz'
filename_output = 'stdout_VRAM_test.txt'

# params
timesteps = 15000                              # initial timesteps value
rois = 10000                                   # initial rois value
increment_var = 'timesteps'                    # control which variable is incremented
step = 1000                                    # how much to increment the variable

# flags
train_model = False

# model 
cebra_time_model = cebra.CEBRA(
    model_architecture='offset10-model',
    device='cuda_if_available',
    conditional='time',
    temperature_mode='auto',
    min_temperature=0.1,
    time_offsets=10,
    max_iterations=ITERS,                      
    max_adapt_iterations=500,
    batch_size=None,
    learning_rate=1e-4,
    output_dimension=3,
    verbose=True,
    num_hidden_units=32,
    hybrid=False
    )

# redirect print output to text file
orig_stdout = sys.stdout
output = open(filename_output, 'w')
sys.stdout = output

## main loop ##

# Incremented timesteps, constant ROIs
while True:
    try:
        # choose where in dataset to sample
        start, stop = 0, 0+timesteps

        # extract and neural data
        # do not attempt to load the entire file 
        print("Accessing data...")
        with h5py.File(filename, 'r') as f:
            
            # neural 
            neural = f['rois']['dfof']
            
            # select first TIMESTEPS timesteps and random ROIS rois
            # neural
            neural_indexes = np.sort(
                                np.random.choice(
                                            np.arange(neural.shape[1]), size=rois, replace=False
                                            )
                                )
            neural = np.array(neural[start:stop, neural_indexes])
            print(f"Loaded neural dataset of shape: {neural.shape}")

            assert(neural.shape == (timesteps, rois))

            # save dataset
            np.savez(f'{data_folder}{filename_dfof}', neural=neural)
            print(f"Saved neural dataset of shape: {neural.shape}")


            # load dataset
            neural = cebra.load_data(f'{data_folder}{filename_dfof}', key="neural")
            print(f"Loaded neural dataset of shape {neural.shape}")

            
            # train and save the model
            if train_model:
                model_name = f"{filename_trunc}_time_{timesteps}points_{rois}rois_{ITERS}iters.pt"
                model_path = f'models/{model_name}'

                cebra_time_model.fit(neural)
                cebra_time_model.save(model_path)
                print(f"Model fit and saved for neural dataset of shape {neural.shape}")

            print("")

            # increment the relevant variable
            if increment_var == 'timesteps':
                timesteps += step
            elif increment_var == 'rois':
                rois += step
    
    # Expect a memory error exception
    except Exception as e:
        print(e)
        # close file before exiting
        f.close()
        break
    
    exit() # end program
