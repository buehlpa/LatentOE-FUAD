# helperfuctions 

import pandas as pd
import seaborn as sns
import os 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import time
from alive_progress import  alive_bar

from pathlib import Path
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from sklearn.metrics import precision_recall_curve

from tensorflow.keras.datasets import fashion_mnist
from PIL import Image
import matplotlib.colors as mcolors

import torch

############################ Data handling IO functions #########################################

def merge_pickle_to_df(MODEL_RESULT_PATH:str)->pd.DataFrame:
    """
    reads pickle files to pandas dataframe
    params: MODEL_RESULT_PATH: path to pickle files
    returns: pandas dataframe
    """
    filename_list=[f for f in os.listdir(MODEL_RESULT_PATH) if (f.endswith(".pkl") and not ("allresults" in f))]
    if len(filename_list)==0:
        raise ValueError("there are no pickle files")
    for idx , pickle_filename in enumerate(filename_list):
        if idx==0:
            results_df = pd.read_pickle(os.path.join(MODEL_RESULT_PATH,pickle_filename) )
        else:
            results_df=pd.concat([results_df,pd.read_pickle(os.path.join(MODEL_RESULT_PATH,pickle_filename))])

    return results_df


def remove_all_intermediate_results(MODEL_RESULT_PATH:str)->None:
    """
    removes all intermediate results
    params: MODEL_RESULT_PATH: path to pickle files
    returns: None
    """
    filename_list=[f for f in os.listdir(MODEL_RESULT_PATH) if (f.endswith(".pkl") and not ("allresults" in f))]
    if len(filename_list)==0:
        raise ValueError("there are no pickle files")
    for pickle_filename in filename_list:
        os.remove(os.path.join(MODEL_RESULT_PATH,pickle_filename))


def save_allresults_pickle(MODEL_RESULT_PATH:str,run:int,name:str='MODEL')->pd.DataFrame:
        """
        saves all results to pickle file and removes all intermediate results
        params: MODEL_RESULT_PATH: path to pickle files
                name: name of the model
        returns: pandas dataframe 
        """
        results_df=merge_pickle_to_df(MODEL_RESULT_PATH)
        SAVE_PATH=os.path.join(MODEL_RESULT_PATH,f"{name}_allresults_run_{run}.pkl")
        results_df.to_pickle(SAVE_PATH)
        # remove all intermediate results
        time.sleep(0.1)
        remove_all_intermediate_results(MODEL_RESULT_PATH)
        # return results_df
            
############################ Plotting / Evaluation #########################################

def plot_density_per_class(pd_dataframe:pd.DataFrame)->None: 
    """
    plots density per class
    """
    plt.rcParams["figure.figsize"] = (30,10)
    for metric in [name for name in list(pd_dataframe) if name not in ['anomaly_', 'normal_label'] ]:
        g = sns.displot(data=pd_dataframe, y=metric, hue="anomaly_", col="normal_label",kind="kde", height=4, aspect=.7,)
        g.set_axis_labels("Density", metric)
        g.set_titles("class: {col_name}")
        g.fig.suptitle(f"Distribution of {metric} by normal_label and anomaly level ", y=1.05)



def  precision_recall(distance_metric:list,groundtruth:list,quantiles:float=0.01,thresh_larger:bool=False)->tuple:
        """
        calculates precision and recall for a given distance metric
        params: distance_metric: list of distance metric values
                groundtruth: list of groundtruth labels
                quantiles: number of quantiles to calculate precision and recall
                thresh_larger: if True, then the larger the distance_metric the more likely its a normal sample

        retuns: tuple of precision and recall
        """

        quantile_list = np.arange(0, 1, quantiles)
        precision, recall = [], []
        for quantile in quantile_list:
            thresh = np.quantile(distance_metric, quantile)
            predict = np.zeros_like(distance_metric)  # predictions are all normals
            if thresh_larger:
                 idxs = np.where(np.array(distance_metric) <= thresh)[0]  # all anomalies           
            else:
                idxs = np.where(np.array(distance_metric) >= thresh)[0]  # all anomalies

            predict[idxs] = 1
            tn, fp, fn, tp = confusion_matrix(groundtruth, predict).ravel()

            precision.append(tp / (tp + fp))
            recall.append(tp / (tp + fn))
        return precision, recall



def calculate_auc(precision:list, recall:list)->float:
    """
    Calculates the area under the curve (AUC) for a precision-recall curve
    params: precision: list of precision values
            recall: list of recall values
    """
    n = len(precision)
    auc = 0.0

    for i in range(1, n):
        # Calculate the width of the trapezoid
        delta_recall = recall[i] - recall[i - 1]

        # Calculate the average height of the trapezoid
        avg_precision = (precision[i] + precision[i - 1]) / 2

        # Calculate the area of the trapezoid
        trapezoid_area = delta_recall * avg_precision

        # Accumulate the area under the curve
        auc += trapezoid_area

    return abs(auc)




def data_precision_recall_dataset(dataframe:pd.DataFrame,quantiles:float=0.01)->pd.DataFrame:


    #TODO output is strange with 0.6 in the end.....
    """
    calculates precision recall for each class and each contam ratio
    params: dataframe with columns: normal_label, contam_ratio, anomaly_, and arbitrary number of metrics
            quantiles: quantiles for the precision recall curve
    returns: dataframe with columns: normal_label, contam_ratio, anomaly_, and arbitrary number of metrics
    """
    # create dataframe with the contam ratio and the normal label

    
    # sets of unique values for normal label and contam ratio
    class_labels = list(set(dataframe['normal_label']))
    contam_list = list(set(dataframe['contam_ratio']))
    
    # create labels for each class and each contam ratio for granularity of  quantiles
    pr_labels=[x for x in class_labels for _ in range(len(np.arange(0, 1, quantiles)))]*len(contam_list)
    pr_contam=[x for x in contam_list for _ in range(len(class_labels)*len(np.arange(0, 1, quantiles)))]
    
    pr_results = pd.DataFrame({'normal_label': pr_labels,'contam_ratio': pr_contam})

    # for each metric calculate precision recall and add to dataframe
    for metric in [name for name in list(dataframe) if name not in ['anomaly_', 'normal_label','contam_ratio']]:
                print(metric)
                
                precision_all,recall_all = [],[]             
                for ratio in contam_list:

                    for class_label in class_labels:

                        subset = dataframe[(dataframe["normal_label"] == class_label) & (dataframe["contam_ratio"] == ratio)]
                        subset_mse = subset[metric]
                        subset_anomaly = subset["anomaly_"]
                        if metric == "nmi_":
                            precision, recall = precision_recall(subset_mse, subset_anomaly,quantiles=quantiles,thresh_larger=True)
                        else:
                            precision, recall = precision_recall(subset_mse, subset_anomaly,quantiles=quantiles)

                        precision_all.extend(precision)
                        recall_all.extend(recall)
                
                # add to dataframe
                pr_add = pd.DataFrame({f'precision_{metric}': precision_all,f'recall_{metric}': recall_all})
                pr_results = pd.concat([pr_results, pr_add], axis=1)
    return pr_results



def plot_precision_recall_multi(dataframe_list:list, plot:bool=True, show_img:bool=False,contam_list:list=None,sklearn_pr:bool=None,title:str="plot",pred_y:str="mse_",min_max:bool=True )->None:
    """
    Calculates precision-recall for all the classes and contamination ratios for multiple datasets and different runs.
    Adds confidence bands to the precision-recall plot.
    plots the precision recall curves for all classes and contamination ratios
    # only for fmnist data

    Args:
        dataframe_list (list): List of dataframes containing results for different runs/datasets.
        plot (bool): Whether to plot the precision-recall curves. Defaults to True. 
        show_img (bool): Whether to show sample images. Defaults to False.
        contam_list (list): List of contamination ratios. Defaults to None. If None, the contamination ratios are extracted from the dataframe.
        sklearn_pr (bool): Whether to use sklearn precision_recall_curve. Defaults to None. If None, the precision_recall function is used.
        title (str): Title of the plot. Defaults to "plot".
        pred_y (str): which metric to use . Defaults to "mse_". If pred_y = "pred_y" then pr goes to estimate the precision recall for the predicted y values with sklearn.
        min_max (bool): Whether to use min and max values for the confidence bands. Defaults to True. If False, the standard deviation is used.
    Returns:

    """
    if show_img:
        scaled_images=get_sample_img_fmnist()

    if contam_list is None:
        contam_list = list(set(dataframe_list[0]['contam_ratio']))

    label_names= ["t_shirt","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle_boot"]

    class_labels = list(set(dataframe_list[0]['normal_label'].astype('int32')))
    

    fig, axes = plt.subplots(2, 5)  # 2 rows and 5 columns for 10 subplots
    fig.suptitle(title)
    handles = []  # List to store plot handles for legend
    with alive_bar(len(class_labels),force_tty=True) as bar:
        for i, class_label in enumerate(class_labels):
            row = i // 5
            col = i % 5
            ax = axes[row, col]
            ax.set_title(f"{label_names[class_label]}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_ylim([0.88, 1.02])
            ax.set_yticks(np.arange(0.88, 1.01,0.02))
            ax.grid(True,alpha =0.5) 

            auc_values = []  # List to store AUC values
            handles = []  # List to store plot handles for legend

            for j, ratio in enumerate(contam_list):
                precision_runs = []  # List to store precision values for each run
                recall_runs = []  # List to store recall values for each run

                for dataframe in dataframe_list:
                    subset = dataframe[(dataframe["normal_label"] == class_label) & (dataframe["contam_ratio"] == ratio)]
                    if pred_y == "pred_y":
                        subset_pred_y = subset['y_pred_']
                        subset_anomaly = subset["anomaly_"]

                        if sklearn_pr:
                            precision, recall, _ = precision_recall_curve(subset_anomaly, subset_pred_y,pos_label=1)
                        else:# does the same as sklearn_ not implementes for pred_y
                            precision, recall, _ = precision_recall_curve(subset_anomaly, subset_pred_y,pos_label=1) 
                    else:
                        subset_score = subset[pred_y]
                        subset_anomaly = subset["anomaly_"]
                    
                        if sklearn_pr:
                            precision, recall, _ = precision_recall_curve(subset_anomaly, subset_score)
                        else:
                            precision, recall = precision_recall(subset_score, subset_anomaly, thresh_larger=False)
                    


                    precision_runs.append(precision)
                    recall_runs.append(recall)

                # Calculate mean and confidence intervals for precision and recall
                precision_mean = np.mean(precision_runs, axis=0)
                precision_std = np.std(precision_runs, axis=0)
                recall_mean = np.mean(recall_runs, axis=0)
                recall_std = np.std(recall_runs, axis=0)
                
                if min_max:
                    precision_max = np.max(precision_runs, axis=0)
                    precision_min = np.min(precision_runs, axis=0)

                # Calculate AUC for mean precision-recall curve
                auc = calculate_auc(precision_mean, recall_mean)
                auc_values.append(auc)


                if plot:
                    # Add confidence bands to the plot
                    if min_max:
                        ax.fill_between(recall_mean, precision_min, precision_max,
                                        alpha=0.3)
                    else:
                        ax.fill_between(recall_mean, precision_mean - 2 * precision_std, precision_mean + 2 * precision_std,
                                        alpha=0.3)

                    if show_img:
                        imagebox = OffsetImage(scaled_images[class_label], zoom=1)
                        ab = AnnotationBbox(imagebox, (1, 1), xycoords='axes fraction', frameon=False)
                        ax.add_artist(ab)
                    handle = ax.plot(recall_mean, precision_mean)  # Store the plot handle
                    handles.append(handle[0])

            bar()
            # Add AUC values to the legend
            legend_labels = [r"$\alpha_{0}$=" + f"{ratio} (AUC: {auc:.2f})" for ratio, auc in zip(contam_list, auc_values)]
            ax.legend(handles, legend_labels, loc='best', title=r"AUC of $\mu$ PR",fontsize='small')

    if plot:
        plt.tight_layout()
        plt.show()


def get_sample_img_fmnist()->list:
    """
    Returns a list of sample images per class from the FMNIST dataset.
    """
    # Load the FMNIST dataset
    (x_train, y_train), _ = fashion_mnist.load_data()
    # Create a list to store the scaled down images
    scaled_images = []
    # Select one sample from each class
    for label in range(10):
        class_indices = np.where(y_train == label)[0]
        sample_index = class_indices[0]  # Get the first index of the class
        sample_image = x_train[sample_index]
        sample_image = Image.fromarray(sample_image)
        sample_image = sample_image.resize((28, 28))
        scaled_images.append(sample_image)
    return scaled_images


def plot_precision_recall_multi_comparison(dataframe_list_1:list, dataframe_list_2:list, plot:bool=True, show_img:bool=False,
                                            contam_list:list=None, sklearn_pr:bool=None, title:str="plot", pred_y:str="mse_", pred_y2:str="mse_", min_max:bool=True )->None:
    """
    #TODO finepolish graphics blue and red are not very good to distinguish
    
    Calculates precision-recall for all the classes and contamination ratios for multiple datasets and different runs.
    draws it for two different models for comparison blue and red 
    Adds confidence bands to the precision-recall plot.
    plots the precision recall curves for all classes and contamination ratios
    # only for fmnist data

    Args:
        dataframe_list_1 (list): List of dataframes containing results for the first set of runs/datasets.
        dataframe_list_2 (list): List of dataframes containing results for the second set of runs/datasets.
        plot (bool): Whether to plot the precision-recall curves. Defaults to True. 
        show_img (bool): Whether to show sample images. Defaults to False.
        contam_list (list): List of contamination ratios. Defaults to None. If None, the contamination ratios are extracted from the dataframe.
        sklearn_pr (bool): Whether to use sklearn precision_recall_curve. Defaults to None. If None, the precision_recall function is used.
        title (str): Title of the plot. Defaults to "plot".
        pred_y (str): which metric to use. Defaults to "mse_". If pred_y = "pred_y" then pr goes to estimate the precision recall for the predicted y values with sklearn.
        min_max (bool): Whether to use min and max values for the confidence bands. Defaults to True. If False, the standard deviation is used.
    Returns:

    """
    if show_img:
        scaled_images = get_sample_img_fmnist()

    if contam_list is None:
        contam_list = list(set(dataframe_list_1[0]['contam_ratio']))

    label_names = ["t_shirt","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle_boot"]

    class_labels = list(set(dataframe_list_1[0]['normal_label'].astype('int32')))

    fig, axes = plt.subplots(2, 5)  # 2 rows and 5 columns for 10 subplots
    fig.suptitle(title)
    handles = []  # List to store plot handles for legend
    with alive_bar(len(class_labels),force_tty=True) as bar:
        for i, class_label in enumerate(class_labels):
            row = i // 5
            col = i % 5
            ax = axes[row, col]
            ax.set_title(f"{label_names[class_label]}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_ylim([0.88, 1.02])
            ax.set_yticks(np.arange(0.88, 1.01,0.02))
            ax.grid(True,alpha =0.5) 

            auc_values = []  # List to store AUC values
            handles = []  # List to store plot handles for legend

            for j, ratio in enumerate(contam_list):
                precision_runs = []  # List to store precision values for each run
                recall_runs = []  # List to store recall values for each run

                for dataframe in dataframe_list_1:
                    subset = dataframe[(dataframe["normal_label"] == class_label) & (dataframe["contam_ratio"] == ratio)]
                    if pred_y == "pred_y":
                        subset_pred_y = subset['y_pred_']
                        subset_anomaly = subset["anomaly_"]

                        if sklearn_pr:
                            precision, recall, _ = precision_recall_curve(subset_anomaly, subset_pred_y,pos_label=1)
                        else:  # does the same as sklearn_ not implemented for pred_y
                            precision, recall, _ = precision_recall_curve(subset_anomaly, subset_pred_y,pos_label=1) 
                    else:
                        subset_mse = subset[pred_y]
                        subset_anomaly = subset["anomaly_"]
                    
                        if sklearn_pr:
                            precision, recall, _ = precision_recall_curve(subset_anomaly, subset_mse)
                        else:
                            precision, recall = precision_recall(subset_mse, subset_anomaly, thresh_larger=False)
                    
                    precision_runs.append(precision)
                    recall_runs.append(recall)

                for dataframe in dataframe_list_2:
                    subset = dataframe[(dataframe["normal_label"] == class_label) & (dataframe["contam_ratio"] == ratio)]
                    if pred_y2 == "pred_y":
                        subset_pred_y = subset['y_pred_']
                        subset_anomaly = subset["anomaly_"]

                        if sklearn_pr:
                            precision, recall, _ = precision_recall_curve(subset_anomaly, subset_pred_y,pos_label=1)
                        else:  # does the same as sklearn_ not implemented for pred_y
                            precision, recall, _ = precision_recall_curve(subset_anomaly, subset_pred_y,pos_label=1) 
                    else:
                        subset_mse = subset[pred_y2]
                        subset_anomaly = subset["anomaly_"]
                    
                        if sklearn_pr:
                            precision, recall, _ = precision_recall_curve(subset_anomaly, subset_mse)
                        else:
                            precision, recall = precision_recall(subset_mse, subset_anomaly, thresh_larger=False)
                    
                    precision_runs.append(precision)
                    recall_runs.append(recall)

                # Calculate mean and confidence intervals for precision and recall
                precision_mean = np.mean(precision_runs, axis=0)
                precision_std = np.std(precision_runs, axis=0)
                recall_mean = np.mean(recall_runs, axis=0)
                recall_std = np.std(recall_runs, axis=0)
                
                if min_max:
                    precision_max = np.max(precision_runs, axis=0)
                    precision_min = np.min(precision_runs, axis=0)

                # Calculate AUC for mean precision-recall curve
                auc = calculate_auc(precision_mean, recall_mean)
                auc_values.append(auc)

                if plot:
                    # Add confidence bands to the plot
                    if min_max:
                        ax.fill_between(recall_mean, precision_min, precision_max, alpha=0.3, color='blue')
                    else:
                        ax.fill_between(recall_mean, precision_mean - 2 * precision_std, precision_mean + 2 * precision_std, alpha=0.3, color='blue')

                    if show_img:
                        imagebox = OffsetImage(scaled_images[class_label], zoom=1)
                        ab = AnnotationBbox(imagebox, (1, 1), xycoords='axes fraction', frameon=False)
                        ax.add_artist(ab)
                    handle = ax.plot(recall_mean, precision_mean, color='blue')  # Store the plot handle
                    handles.append(handle[0])

                for j, ratio in enumerate(contam_list):
                    precision_runs = []  # List to store precision values for each run
                    recall_runs = []  # List to store recall values for each run

                    for dataframe in dataframe_list_2:
                        subset = dataframe[(dataframe["normal_label"] == class_label) & (dataframe["contam_ratio"] == ratio)]
                        if pred_y2 == "pred_y":
                            subset_pred_y = subset['y_pred_']
                            subset_anomaly = subset["anomaly_"]

                            if sklearn_pr:
                                precision, recall, _ = precision_recall_curve(subset_anomaly, subset_pred_y,pos_label=1)
                            else:  # does the same as sklearn_ not implemented for pred_y
                                precision, recall, _ = precision_recall_curve(subset_anomaly, subset_pred_y,pos_label=1) 
                        else:
                            subset_mse = subset[pred_y2]
                            subset_anomaly = subset["anomaly_"]
                        
                            if sklearn_pr:
                                precision, recall, _ = precision_recall_curve(subset_anomaly, subset_mse)
                            else:
                                precision, recall = precision_recall(subset_mse, subset_anomaly, thresh_larger=False)
                        
                        precision_runs.append(precision)
                        recall_runs.append(recall)

                    # Calculate mean and confidence intervals for precision and recall
                    precision_mean = np.mean(precision_runs, axis=0)
                    precision_std = np.std(precision_runs, axis=0)
                    recall_mean = np.mean(recall_runs, axis=0)
                    recall_std = np.std(recall_runs, axis=0)
                    
                    if min_max:
                        precision_max = np.max(precision_runs, axis=0)
                        precision_min = np.min(precision_runs, axis=0)

                    # Calculate AUC for mean precision-recall curve
                    auc = calculate_auc(precision_mean, recall_mean)
                    auc_values.append(auc)

                    if plot:
                        # Add confidence bands to the plot
                        if min_max:
                            ax.fill_between(recall_mean, precision_min, precision_max, alpha=0.3, color='red')
                        else:
                            ax.fill_between(recall_mean, precision_mean - 2 * precision_std, precision_mean + 2 * precision_std, alpha=0.3, color='red')

                        if show_img:
                            imagebox = OffsetImage(scaled_images[class_label], zoom=1)
                            ab = AnnotationBbox(imagebox, (1, 1), xycoords='axes fraction', frameon=False)
                            ax.add_artist(ab)
                        handle = ax.plot(recall_mean, precision_mean, color='red')  # Store the plot handle
                        handles.append(handle[0])

            bar()
            # Add AUC values to the legend
            legend_labels = [f"Ratio: {ratio} (AUC: {auc:.2f})" for ratio, auc in zip(contam_list, auc_values)]
            ax.legend(handles, legend_labels, loc='best', title="Ratios", fontsize='small')

    if plot:
        plt.tight_layout()
        plt.show()


def plot_pr_ntl(assumed_contamination,contam_list:list = [0.0, 0.1, 0.2,0.3, 0.4],model:str="loe_hard_assumed_contam",RESULTS_PATH:str=r"/root/LatentOE-AD/RESULTS/fmnist/")->None:
    """
    calls the plot_precision_recall_multi function for differnet NTL models 
    params: assumed_contamination: assumed contamination ratio
            contam_list: list of contamination ratios
            model: name of the model
            RESULTS_PATH: path to results
    returns: None
    
    """


    plt.rcParams["figure.figsize"] = (25, 10)  # Adjust the figure size as per your preference
    
    


    if "assumed" not in model:
        title=f'NTL {model}' + r' Precision-Recall Curves for Fashion MNIST different contamination ratios, $\alpha_{0}$ = $\alpha$'
        MODEL_RESULT_PATH = os.path.join(Path(RESULTS_PATH),f'{model}_allresults')
        # load the diffent runs in a list
        nameslist,dataframe_list=[name for name in os.listdir(MODEL_RESULT_PATH) if "allresults_run_"in name],[]
        for name in nameslist:
            dataframe_list.append(pd.read_pickle(os.path.join(MODEL_RESULT_PATH,name)))


    else:
        title=f'NTL {model[:-15]}' + r' Precision-Recall Curves for Fashion MNIST different contamination ratios, $\alpha$ = '+str(assumed_contamination)
        MODEL_RESULT_PATH = os.path.join(Path(RESULTS_PATH),f'{model}_{assumed_contamination}_allresults')
        # load the diffent runs in a list
        nameslist,dataframe_list=[name for name in os.listdir(MODEL_RESULT_PATH) if "allresults_run_"in name],[]
        for name in nameslist:
            dataframe_list.append(pd.read_pickle(os.path.join(MODEL_RESULT_PATH,name)))




    plot_precision_recall_multi(dataframe_list,plot=True,contam_list=contam_list,show_img=True,sklearn_pr=False,title=title,pred_y='score_')


################  helperfuncitons for cifar , fmnsit plots #####################

def show_imgs(X:np.array,y:np.array)->None:
    """
    Show 100 random images from the dataset
    Parameters
    X : np.array
    y : np.array
    Returns None
    """

    plt.figure(1, figsize=(12,12))
    k = 0
    for i in range(0,10):
        for j in range(0,10):
            while y[k] != i: k += 1
            plt.subplot2grid((10,10),(i,j))
            plt.imshow(X[k])
            plt.axis('off')
            k += 1
    plt.show()

def show_imgs_fmnist(X:np.array, y:np.array)->None:
    """
    Function to show the first 5 images of each class in the Fashion MNIST dataset.
    :param X: The images of the dataset
    :param y: The labels of the dataset
    :return: None
    """
    
    label_names = ["t-shirt","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle boot"]

    plt.figure(1, figsize=(12, 10))  # Adjust the figsize as needed to fit the labels on the side
    plt.suptitle("Fashion MNIST Samples ", fontsize=16, y=0.88)  # Add the title at the top

    for i in range(10):  # Loop through each column
        images_of_class_i = [X[j] for j in range(len(X)) if y[j] == i]

        plt.subplot(11, 10, i + 1)  # Position for the label on top of the column
        plt.text(0.5, 0.2, f'{label_names[i]}', ha='center', va='center', fontsize=14)
        plt.axis('off')  # Turn off the axis for the subplot containing the label

        for j in range(5):  # Loop through each row
            plt.subplot(11, 10, (j+1)*10 + i + 1)  # Position for the image in the column
            plt.imshow(images_of_class_i[j])
            plt.axis('off')
    plt.show()

################  helperfuncitons for tsne plots              #####################


from loader.LoadData import  FMNIST_feat

def index_dictionary(trainset:torch.Tensor, x_train:torch.Tensor,y_train:torch.Tensor, batch_size:int)->dict:
        
        """ 
        #BULE used for the tsne plots
        computes the indices of the matching elements in x_train for each element in trainset
        returns a dictionary with 
        Args:
            trainset: torch.Tensor of shape (total_samples, 2048)
            x_train: torch.Tensor of shape (total_samples, 2048)
            y_train: torch.Tensor of shape (total_samples, 1)
            batch_size: int

        Returns:
            indexdict: dict

        """

        # Assuming trainset[0] is a tensor of shape (total_samples, 2048)
        total_samples = trainset.shape[0]
        # Compute element-wise equality between trainset and x_train in batches
        equal_elements_list = []
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            equal_elements = torch.all(trainset[i:end_idx, None, :] == x_train, dim=-1)
            equal_elements_list.append(equal_elements)

        # Concatenate results from all batches
        equal_elements = torch.cat(equal_elements_list)

        # Find the indices of the matching elements in x_train for each element in trainset
        matching_indices = torch.nonzero(equal_elements, as_tuple=False)

        # Create a dictionary with default values of 2 for all indices in trainset
        indexdict = {i: 2 for i in range(total_samples)}

        # Update the indexdict with matching indices from x_train and y_train
        for i, j in matching_indices:
            indexdict[i.item()] = int(y_train[j.item()])

        # Now indexdict contains the required mapping of indices in trainset to y_train values
        return indexdict


def all_indexdicts(dataset:torch.Tensor,classlist:list,fraction:float,trainset_name:str,contamination_rate:float=0.1,root:str="/root/LatentOE-AD/DATA/fmnist_features/",use_test_set:bool=False)->dict:
    """
    returns a dictionary with the index of the trainset for each class and the fraction of the trainset

    params:     
                trainset: torch.Tensor of shape (total_samples, 2048)
                classlist: list of classes form: [0,1,2,3,4,5,6,7,8,9]
                fraction: fraction of the trainset 
                trainset_name: name of the trainset form: trainset_2048_fraction_0.01.pt or trainset_2048.pt for full set
                contamination_rate: contamination rate of the trainset
                root: root path of the trainset
                use_test_set: bool if true uses the testset instead of the trainset
    """

    if 'fraction' not in trainset_name:
        fraction=1

    indexdicts={}
    for class_ in classlist:
        x_train, y_train, x_test, y_test = FMNIST_feat(class_,root=root  ,contamination_rate=contamination_rate,trainset_name=trainset_name)
        # Set batch size according to your memory limitations
        batch_size = 3000
        if use_test_set:
            indexdicts[class_] = index_dictionary(dataset, x_test,y_test, batch_size)
        else:
            indexdicts[class_] = index_dictionary(dataset, x_train,y_train, batch_size)


    indexdicts["fraction"]=fraction
    indexdicts["contamination_rate"]=contamination_rate    

    return indexdicts



def subplot_tsne_per_class(tsne_result:np.array, indexdicts:dict, class_:int)->None:

    """
    helper function to plot tsne for each class

    Args:   tsne_result: np.array of shape (n_samples,2)
            indexdicts: dict of indexdicts for each class
            class_: int class to plot

    """

    indexdict = indexdicts[class_]

    label_names = ["t-shirt","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle-boot"]
    colors = {0.0: sns.color_palette("Paired")[class_], 1.0: 'black', 2.0:'lightgrey'}
    
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': list(indexdict.values())})
    tsne_result_df = tsne_result_df.sort_values(by='label', ascending=False)

    # Create a new subplot for each class
    ax = plt.subplot(2, 5, class_ + 1)  # 2 rows, 5 columns, class_ + 1 (since indexing starts from 1)
    sns.scatterplot(x='tsne_1', y='tsne_2', data=tsne_result_df, hue='label', palette=colors, s=15, ax=ax)

    for num in colors.keys():
        alpha = 0.01 if num == 2 else 1.0 if num == 0 else 0.8
        facecolor = colors[num]
        scatter = ax.findobj(match=lambda x: x.get_label() == num)
        for s in scatter:
            s.set_alpha(alpha)
            s.set_facecolor(facecolor)

    ax.set_title(f'{label_names[class_]}', fontsize=16)
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_ylabel('')  # Remove y-axis label

    legend_labels = {0.0: 'normal', 1.0: 'anomaly', 2.0: 'unused'}
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, [legend_labels[float(label.get_label())] for label in handles],loc="lower right", fontsize=13)





def plot_all_tsne_per_class(tsne_result:np.array, indexdicts:dict,show_testset:bool=False)->None:
    """
    plots all the trainingset on a given tsne results for each class

    Args:   tsne_result: np.array of shape (n_samples,2)   
            indexdicts: dict of indexdicts for each class
            show_testset: bool if true plots a differtent title
    """
    
    if show_testset:
        title = f'FMNIST TSNE Testset'
    else:

        fraction = indexdicts["fraction"]
        contamination_rate = indexdicts["contamination_rate"]
        title = r'TSNE Trainingset $\lambda$ = '+  f'{int(fraction*100)}%,'+ r' contamination rate:  $\alpha_{0}= $' +f'{int(contamination_rate*100)}%'

    # Create the main figure to hold all the subplots
    plt.figure(figsize=(18, 8))
    # Iterate over all classes and plot the t-SNE for each one
    for class_ in range(10):
        subplot_tsne_per_class(tsne_result, indexdicts, class_)

    # Add the overall title above all the subplots
    plt.suptitle(title, fontsize=18, y=1)

    # Adjust the layout and spacing of the subplots
    plt.tight_layout()

    # Show the combined plot
    plt.show()



def apply_color_to_image(image:np.array, color:tuple)->np.array:
    """"
    Applies a color to a grayscale image helperfunciton for plot_tsne_with_imgs
    :param image: grayscale image
    :param color: color to apply
    :return: colored image
    """

    # Normalize the grayscale image to the range [0, 1]
    normalized_image = image / 255.0
    # Create a 3-channel RGB image by replicating the grayscale values across channels
    rgb_image = np.stack([normalized_image] * 3, axis=-1)
    # Replace the RGB values with the desired color from the sns palette
    colored_image = rgb_image * np.array(color)
    # Clip values to ensure they are within the valid range [0, 1]
    colored_image = np.clip(colored_image, 0, 1)
    # Convert back to the range [0, 255] for plotting
    colored_image = (colored_image * 255).astype(np.uint8)
    colored_image[colored_image.sum(axis=-1) == 1] = [0, 0, 0]  # set background to white
    return colored_image


def plot_tsne_with_imgs(tsne_result_df, x_train, sample_size=10000, imsize=2, title=None, title_fontsize=20, label_fontsize=12):
    """
    Plots the tsne result with the images from the training set.

    Args:
        tsne_result_df (pd.DataFrame): The tsne result dataframe. from sklearn.manifold TSNE 
        x_train (np.array): The training set. with the original images.
        sample_size (int, optional): The number of samples to plot. Defaults to 10000.
        imsize (int, optional): The size of the image. Defaults to 2.
        title (str, optional): The title of the plot. Defaults to None.
        title_fontsize (int, optional): The font size for the title. Defaults to 20.
        label_fontsize (int, optional): The font size for the legend labels. Defaults to 12.

    """
    plt.figure(figsize=(30, 30))
    plt.style.use('dark_background')

    sample_indexes = np.random.choice(len(x_train), sample_size, replace=False)
    tsne_sample = tsne_result_df.iloc[sample_indexes]

    sns_palette = sns.color_palette('Paired', n_colors=12)
    labels = tsne_sample['label'].astype(int)
    colors = np.array(sns_palette)[labels]

    x_vals = tsne_sample['tsne_1']
    y_vals = tsne_sample['tsne_2']

    for i, (x, y, label) in enumerate(zip(x_vals, y_vals, labels)):
        img = x_train[sample_indexes[i]]
        plt.imshow(apply_color_to_image(img, colors[i]), extent=(x - imsize, x + imsize, y - imsize, y + imsize), cmap="gray", alpha=0.8)

    # Scatter plot all points at once
    plt.scatter(x_vals, y_vals, c=labels, cmap='Paired', s=0.01)

    # Add title
    if title is None:
        plt.title(f'TSNE on  ({sample_size} Samples)', fontsize=title_fontsize)
    else:
        plt.title(title, fontsize=title_fontsize)

    label_names = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle-boot"]
    label_dict = {i: label_names[i] for i in range(len(label_names))}
    # Change the legend labels
    legend_labels = label_dict
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markersize=10, label=l)
                        for c, l in zip(np.array(sns_palette), legend_labels.values())], fontsize=label_fontsize)

    plt.axis('off')
    plt.show()




def plot_tsne_scatter(tsne_result_df:pd.DataFrame,label_dict:dict,title=None)->None:
    """
    Plot a scatter plot of the tsne results
    :param tsne_result_df: the tsne results dataframe
    :param label_dict: a dictionary mapping the labels to the names
    :param title: the title of the plot
    :return: None
    """
    
    plt.rcParams["figure.figsize"] = (10,10)
    ax=sns.scatterplot(x='tsne_1', y='tsne_2',data=tsne_result_df,hue='label',palette="Paired",s=5)
    # Add title
    if title != None:
        ax.set_title(title)
    else:
        ax.set_title(f'TSNE on images FMNIST set')

    # Change the legend labels
    legend_labels = label_dict
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, [legend_labels[float(label.get_label())] for label in handles])