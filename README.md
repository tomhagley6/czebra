# Czebra
Project folder for CEBRA modelling

## Set up
Package dependencies are contained in the environment.yml file  
Use prefix: /Users/username/anaconda3/envs/env-name  
Manually installed packages were:
- cebra
- pytorch
- cuda

## Contents
Notebooks used over time are found in the dated 'notebooks' folders. These can replicate models 
created in the specified week.  
'Archived models' contains saved models from each week, along with saved figures of model outputs

## Usage
Notebooks expect a path to an h5 data file, and a directory to store/load data from (local 'data' folder).  
To save a model, notebooks expect a directory to save the model and figures to (project 'archived_models' folder).
Models can be loaded from this archived model folder. You will need to input the speicific path of the model into cebra.load(), 
or run the notebook with global LOAD_MODEL = True, using the correct labels and parameters in the notebook. 

The most recent leaky integrator model is found in: test_notebooks > 231030 > decode_stimulus_laterality_tectal_leaky-integrator_NEWEST
