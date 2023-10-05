
"""
this script trains and test a model for a given set of contamination ratios 

saves for each contamination ratio a pickle file containing the specified distance metrics for each example

 """


#activate conda env AutencoderTF env tf '2.10.0' , python 3.9.16
import tensorflow as tf
from skimage.metrics import mean_squared_error, normalized_root_mse,normalized_mutual_information
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.svm import OneClassSVM

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from alive_progress import  alive_bar
import torch
import sys
from pathlib import Path
import os

print(sys.path)
sys.path.append('/root/LatentOE-AD')
from loader.LoadData import CIFAR10_feat , FMNIST_feat
from Additional_Code_BULE.utils.helper_functions  import *



print(f'tensorflow_version: {tf.__version__}')
print(f'torch_version: {torch.__version__}')
print("Tensorflow: Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(f'GPU for pytorch: {torch.cuda.is_available()}')


DATA_PATH="/root/LatentOE-AD/DATA/fmnist_features/"
modelname='OneClassSVM'
MODEL_SHORT= 'OneClassSVM'



runs=[0,1,2,3,4]    #list of how many runs 
testrun=False



labels=[0,1,2,3,4,5,6,7,8,9]
contam_list=np.round(np.arange(0,0.5,0.05),2)
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

                scores_list,y_pred_list,normal_label,anomaly,contam_ratio=[],[],[],[],[]

                for i in labels:
                    # load data for each class
                    x_train, y_train, x_test, y_test = FMNIST_feat(i,root=DATA_PATH,contamination_rate=contam)
                    x_train=x_train.numpy()
                    x_test=x_test.numpy()

                    # # transformation of features train
                    scaler = MinMaxScaler()
                    scaler.fit(x_train)
                    x_train=np.sqrt(scaler.transform(x_train))

                    # # transformation of features train
                    scaler = MinMaxScaler()
                    scaler.fit(x_test)
                    x_test=np.sqrt(scaler.transform(x_test))
                    
                    # OneclassSVM
                    clf = OneClassSVM(gamma='auto').fit(x_train)
                    y_pred = clf.predict(x_test)
                    scores_= clf.score_samples(x_test)

                    y_pred[y_pred==-1]=0
                    scores_list.extend(scores_) 
                    y_pred_list.extend(y_pred)

                    #reconstruction metrics for vectors
                    for j in range(0,len(y_pred)):
                    # labels ratio and anomaly ratio
                        normal_label.append(i)
                        anomaly.append(y_test[j])
                        contam_ratio.append(contam)
                        
                        bar()

                df_per_contam =pd.DataFrame({'scores_':scores_list,'y_pred_': y_pred_list,'normal_label':normal_label,'anomaly_':anomaly,'contam_ratio':contam_ratio})#,

                SAVE_PATH=os.path.join(MODEL_RESULT_PATH,f"{modelname}_contam_{int(contam*100)}%_run_{run}.pkl")
                df_per_contam.to_pickle(SAVE_PATH)

            save_allresults_pickle(MODEL_RESULT_PATH,run=run,name=modelname)


if __name__ == "__main__":
    main()