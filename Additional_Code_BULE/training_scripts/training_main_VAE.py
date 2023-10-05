
"""
this script trains and test a model for a given set of contamination ratios 

saves for each contamination ratio a pickle file containing the specified distance metrics for each example

 """

#activate conda env AutencoderTF env tf '2.10.0' , python 3.9.16
import tensorflow as tf
from sklearn.preprocessing import normalize, MinMaxScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alive_progress import  alive_bar
import torch
import sys
from pathlib import Path
import os



# https://avandekleut.github.io/vae/

import torch; torch.manual_seed(0)
import torch.nn.functional as F
import torch.utils
import torch.distributions
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
device = 'cuda' if torch.cuda.is_available() else 'cpu'


print(sys.path)
sys.path.append('/root/LatentOE-AD')
from loader.LoadData import CIFAR10_feat , FMNIST_feat
from Additional_Code_BULE.utils.helper_functions  import *
from Additional_Code_BULE.utils.VAE_models_Torch  import *


print(f'tensorflow_version: {tf.__version__}')
print(f'torch_version: {torch.__version__}')
print("Tensorflow: Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(f'GPU for pytorch: {torch.cuda.is_available()}')


DATA_PATH="/root/LatentOE-AD/DATA/fmnist_features/"
modelname='Variational_Autoencoder_features'
MODEL_SHORT= 'VAE'

runs=[0,1,2,3,4]    #list of how many runs 
all_metrics=False # if True, all metrics are calculated, if False only mse 
testrun=False

labels=[0,1,2,3,4,5,6,7,8,9]
contam_list=np.round(np.arange(0,0.7,0.05),2)
epochs=10

if testrun:
    labels=[0]
    contam_list=[0.0,0.1]
    epochs=1

MODEL_RESULT_PATH = Path(f"/root/LatentOE-AD/RESULTS/fmnist/{MODEL_SHORT}")

def main():
    with alive_bar(len(runs)*len(labels)*len(contam_list)*10000,force_tty=True) as bar:
        #each run creates new samples
            for run in runs:
            # for every contamination ratio
                for contam in contam_list:

                    nmi,mse,nrmse,csim,msle,normal_label,anomaly,contam_ratio=[],[],[],[],[],[],[],[]

                    for i in labels:
                        # load data for each class
                        x_train, _ , x_test, y_test = FMNIST_feat(i,root=DATA_PATH,contamination_rate=contam)

                        #x_train= x_train.numpy()
                        #x_test= x_test.numpy()

                        ##transformation of features
                        scaler = MinMaxScaler()
                        scaler.fit(x_train)
                        x_train=torch.tensor(np.sqrt(scaler.transform(x_train)),dtype=torch.float32)

                        scaler = MinMaxScaler()
                        scaler.fit(x_test)
                        x_test=torch.tensor(np.sqrt(scaler.transform(x_test)),dtype=torch.float32)
                        
                        # instantiate model & fit to data
                        latent_dims=64
                        vae = VariationalAutoencoder(latent_dims).to(device) # GPU
                        vae = train(vae, x_train.to(device), epochs=epochs)

                        #reconstruction metrics for vectors
                        for j in range(0,len(x_test)):

                            #different metrics 
                            predicted_sample_j = (vae(x_test[j].to(device)).reshape(2048)).detach()
                            original_sample_j = x_test[j].to(device)
                            if all_metrics:
                                    #different metrics 
                                mse.append(torch.norm(predicted_sample_j - original_sample_j, 2).cpu().numpy())
                                csim.append(F.cosine_similarity(predicted_sample_j.unsqueeze(0), original_sample_j.unsqueeze(0)).cpu().numpy())
                            else:
                                mse.append(torch.norm(predicted_sample_j - original_sample_j, 2).cpu().numpy())

                            # labels ratio and anomaly ratio
                            normal_label.append(i)
                            anomaly.append(y_test[j])
                            contam_ratio.append(contam)
                            bar()
                            
                    # creating a pandas dataset

                    if all_metrics:
                        csim=[value.item() for value in csim]
                        df_per_contam =pd.DataFrame({'mse_': mse,'csim_':csim,'anomaly_':anomaly,'normal_label':normal_label,'contam_ratio':contam_ratio})#, 
                        df_per_contam['mse_']=df_per_contam['mse_'].astype(str).astype(float)
                        df_per_contam['csim_']=df_per_contam['csim_'].astype(str).astype(float)
                    else:
                        df_per_contam =pd.DataFrame({'mse_': mse,'anomaly_':anomaly,'normal_label':normal_label,'contam_ratio':contam_ratio})#, 
                        df_per_contam['mse_']=df_per_contam['mse_'].astype(str).astype(float)

                    SAVE_PATH=os.path.join(MODEL_RESULT_PATH,f"{modelname}_contam_{int(contam*100)}%_run_{run}.pkl")
                    df_per_contam.to_pickle(SAVE_PATH)


                save_allresults_pickle(MODEL_RESULT_PATH,run=run,name=modelname)



if __name__ == "__main__":
    main()