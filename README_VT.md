# README for VT2 

this readme guides through the added code for the VT from BULE

Folder structure which is added:

Additional_CODE_BULE -  depreceated           # code which is not used anymore
                     -  loe_utils             # contains the helperfunctions for the saving of pickle files in the original LOE implementation
                     -  plots                 # plots of the results
                     -  training_scripts      # contains bash script to run multiple runs of the LOE blind etc of ntl
                     -  utils                 # contains helperfunctions and model for the Added code from the VT
                     -
                     ...                      
                     evaluating_models.ipynb  # compares different models from VT and BOSCH LOE
                     inspecting_dataset.ipynb # inspecting the features with dimensionalty reduction method and rescaling of the features
                     ...
                     workNB...                # these  notebooks are made for working on differnet toppics,




Code which is altered in the original Code base from BOSCH is marked by the comment #BULE 
basically it just added the   for the NTL. trainer file the returns of the groundtruth / confidence scores from the model 
and added a few lines to save the results in a pickle file per run in the RESULTS folder

als there is and argument added to the argparse --assumed_contamination which can be used as the prior "knowledge" or assumption of
the contamination of the dataset



# How to use:


## for LOE etc 

run :
python Launch_Exps.py --config-file config_fmnist.yml   --contamination 0.0 --assumed-contamination 0.0 --dataset-name fmnist  --trainset_fraction

if you want different - loss functions etc change the  config file

- it saves contamination ration in a diffenrent folder, to concatenate them run the function:

concatenate_allresults(MODEL_RESULT_PATH:str,modelname:str='loe_hard',assumed_contamination:float=0.0,n_runs:int=5)
from
LatentOE-AD/Additional_Code_BULE/loe_utils/helperfunctions.py


for the multirun bash scripts run 

 bash Additional_Code_BULE/training_scripts/run_multiple.sh



# Data

- downsample dataset with: Extract_img_features import downsample_dataset on the extracetd features


## for VT autencoder etc. 
run training_scripts/ training_main_*.py

- it automatically saves the pickle files of the results in a folder under RESULTS/

each run gets an allresults_run_{}.pkl file which contains all the results for the different contamination ratios



## inspect results

- evaluating_models.ipynb 