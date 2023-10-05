import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# this file contains the funcitons to crate the lineplots of the resulst AUC and the heatmap for the sensitivity analysis

def get_dataframe_from_results(contam_list:list,folderpath:str,loss_list:list,assumed_contamination:float)->pd.DataFrame:
    """
    This function reads the results from the folderpath and returns a dataframe with the results
    :param contam_list: list of real contamination ratios for which to read the results
    :param folderpath: path to the folder where the results are stored
    :param loss_list: list of losses
    :param assumed_contamination: assumed contamination ratio for the loe models
    :return: dataframe with the results
    """

    for loss in loss_list:
        modelname_list=[f'{cont}_{loss}_{assumed_contamination}_' for cont in contam_list]
        modelname_list.extend([f'{cont}_{loss}' for cont in contam_list])
        abspath_list_results=sorted([folderpath +file+ '/assessment_results.json' for file in os.listdir(folderpath)if file in modelname_list ])
        if loss_list.index(loss)==0:
            df=read_json_from_results(abspath_list_results,loss,assumed_contamination)
        else:
            df_=read_json_from_results(abspath_list_results,loss,assumed_contamination)
            df=pd.concat([df,df_],axis=0)
    df['contam']=df['contam'].astype(float)
    return df

def regex_extract(filepath:str,loss:str,assumed_contamination:float)->float:
    """
    Extracts the number from a string for a distinc regex pattern
    param filepath: string example: "/root/LatentOE-AD/RESULTS/fmnist/0.3_loe_hard_0.1_/assessment_results.json"
    return: number
    """
    # distictive pattern of the file name
    pattern = f"/(\d+(?:\.\d+)?)(?=_{loss})"#_0\{str(assumed_contamination)[1:]}

    match = re.search(pattern, filepath)
    if match:
        return match.group(1)
    else:
        return None


def read_json_from_results(abspath_list_results:list,loss:str,assumed_contamination:float)->pd.DataFrame:
    """
    extract the auc and contamination from the results file and returns a dataframe
    param abspath_list_results: list of absolute paths to the results files
    param loss: string of the loss function
    param assumed_contamination: float of the assumed contamination
    return: pd.DataFrame
    """

    auc_list,contam_list,modelnames=[],[],[]
    for filepath in abspath_list_results:
        with open(filepath) as f:
            results = f.read()
            auc = json.loads(results)['avg_TS_auc_all']
            contamination=regex_extract(filepath,loss,assumed_contamination)
            contam_list.append(contamination)
            auc_list.append(auc)
            modelnames.append(loss)
    df=pd.DataFrame({'auc':auc_list,'contam':contam_list,'model':modelnames})
    return df



def plot_lineplots(df:pd.DataFrame)->None:
    """
    This function plots the results of the dataframe
    param df: dataframe with the strucutre of pd.DataFrame({'auc':auc_list,'contam':contam_list,'model':modelnames})
    """
    contam_list= list(set(df['contam']))
    sns.lineplot(data=df, x="contam", y="auc", hue="model")
    sns.scatterplot(data=df, x="contam", y="auc", hue="model", marker="o", legend=False)
    plt.legend(loc='lower left')
    plt.yticks(np.arange(0.78, 1, step=0.04))
    plt.ylim([0.776, 0.99])
    plt.xlabel(r'contamination ratio , $\alpha_{0}$')
    plt.title(f'FMNIST, AUROC for different contamination ratios ')
    plt.xticks(contam_list)
    plt.show()



def get_heatmap_df(folderpath:str,model:str,contlist:list,mode:str='all')->pd.DataFrame:
    """
    gets for each of the values  j and k in contlist the model with contlist[i]_model_contlist[k] and returns a dataframe with the results
    param folderpath: string of the folderpath where the results are stored
    param model: string of the model
    param contlist: list of contamination ratios
    param mode: string of the mode, can be 'all','min' or 'max'    
    return: pd.DataFrame with the results oreder by alpha0
    
    """

    alpha0 = sorted(np.round(list(np.round(contlist, 2)) * len(contlist), 2))
    alpha = np.round(list(np.round(contlist, 2)) * len(contlist), 2)
    loss_list=[model]*len(alpha0)

    modelname_list=[f'{a0}_{loss}_{a}_' for a0,a,loss in zip(alpha0,alpha,loss_list)]
    abspath_list_results=[folderpath +file+ '/assessment_results.json' for file in os.listdir(folderpath)if file in modelname_list ]

    auc_list=[]
    for a0,a in zip(alpha0,alpha):
        filepath=get_path_from_abspathlist(model,a0,a,abspath_list_results)
        with open(filepath) as f:
            results = f.read()
            if mode=='all':
                auc = json.loads(results)['avg_TS_auc_all']
            if mode=='min':
                json_results = json.loads(results)
                auc= json_get_min(json_results)
            if mode=='max':
                json_results = json.loads(results)
                auc= json_get_max(json_results)
            auc_list.append(auc)

    df = pd.DataFrame({'alpha0': alpha0, 'alpha': alpha, 'auc': auc_list})
    return df



def get_path_from_abspathlist(model:str,a0:float,a:float,abspath_list_results:list)->str:
    """
    returns the path of the results file for a given model and contamination ratio
    param model: string of the model
    param a0: float of the assumed contamination
    param a: float of the real contamination
    param abspath_list_results: list of absolute paths to the results files
    
    return: string of the path
    """
    
    for path in abspath_list_results:
        if f'{a0}_{model}_{a}_' in path:
            return path
    else:
        return None


def json_get_min(json_results):
    """
    returns the minimum of the auc values in the json file
    param json_results: json file
    return: float of the minimum
    """
    classlist_auc=[f'avg_TS_auc_{nr}' for nr in np.arange(0,10,1)]
    auc_list=[]
    for class_ in classlist_auc:
        auc_list.append(json_results[class_])
    return min(auc_list)

def json_get_max(json_results):
    """
    returns the maximum of the auc values in the json file
    param json_results: json file
    return: float of the maximum
    """
    classlist_auc=[f'avg_TS_auc_{nr}' for nr in np.arange(0,10,1)]
    auc_list=[]
    for class_ in classlist_auc:
        auc_list.append(json_results[class_])
    return max(auc_list)    



def plot_heatmap(df:pd.DataFrame,title:str='AUROC',vmin_max:tuple=None)->None:
        """
        This function plots the results as a heatmap to reproduce the sensistivity analysis
        param df: dataframe with the results format : pd.DataFrame({'alpha0': alpha0, 'alpha': alpha, 'auc': auc_list})
        param title: string of the title
        param vmin_max: tuple of the min and max values for the colorbar

        """
        plt.figure(figsize=(12, 9))

        # Pivot dataframe
        df_pivot = df.pivot('alpha0','alpha',  'auc')
        # Create heatmap
        if vmin_max is not None:
            sns.heatmap(df_pivot, cmap='viridis', annot=True, fmt=".3f", cbar=True,vmin=vmin_max[0],vmax=vmin_max[1])
        else:
            sns.heatmap(df_pivot, cmap='viridis', annot=True, fmt=".3f", cbar=True)

        plt.title(title)
        plt.xlabel(r'assumed contamination: $\alpha$')
        plt.ylabel(r'real contamination: $\alpha_{0}$')
        plt.show()


def plot_heatmap_multiple(dfs: list, titles: list = None,overall_title:str=None, vmin_max: tuple = None) -> None:
    """
    This function plots the heatmaps for multiple dataframes side by side.
    
    param dfs: list of dataframes with the results. Each dataframe should have columns: 'alpha0', 'alpha', 'auc'.
    param titles: list of strings for the titles of each heatmap.
    param vmin_max: tuple of the min and max values for the colorbar.

    """
    if not isinstance(dfs, list) or len(dfs) == 0:
        raise ValueError("The 'dfs' parameter must be a non-empty list of dataframes.")

    num_plots = len(dfs)
    if titles is None:
        titles = [f"Heatmap {i + 1}" for i in range(num_plots)]
    elif len(titles) != num_plots:
        raise ValueError("The number of titles should match the number of dataframes.")

    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharey=True)#figsize=(10 * num_plots, 12)#(6 * num_plots, 9)

    # Calculate the overall vmin and vmax for the colorbar
    if vmin_max is None:
        all_data = pd.concat(dfs)
        overall_vmin = all_data['auc'].min()
        overall_vmax = all_data['auc'].max()
    else:
        overall_vmin, overall_vmax = vmin_max

    for i, (df, ax) in enumerate(zip(dfs, axes), start=1):
        df_pivot = df.pivot('alpha0', 'alpha', 'auc')

        sns.heatmap(df_pivot, cmap='viridis', annot=True, fmt=".3f", cbar=False,
                    vmin=overall_vmin, vmax=overall_vmax, ax=ax)

        ax.set_title(titles[i - 1])
        ax.set_xlabel('')  # Remove x-axis label for individual heatmaps
        ax.set_ylabel('')  # Remove y-axis label for individual heatmaps
        #ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)  # Remove ticks and tick labels

    # Create a separate colorbar for all heatmaps and position it outside the subplots
    cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=overall_vmin, vmax=overall_vmax)),
                        cax=cbar_ax)
    #cbar.set_label('', fontsize=14)  # Set the colorbar label here
    cbar.ax.tick_params(labelsize=12)  # Set the tick label size for the colorbar

    # Add overall title to the whole plot
    if overall_title is None:
        overall_title = "Overall Title of the Plot"
    else:
        overall_title = str(overall_title)

    plt.suptitle(overall_title, fontsize=16, y=0.95)

    # Set y-label on the left side
    y_label_text = r'real contamination: $\alpha_{0}$'
    fig.text(-0.02, 0.5, y_label_text, va='center', rotation='vertical', fontsize=14)

    # Set x-label in the middle of the plot
    x_label_text = r'assumed contamination: $\alpha$'
    fig.text(0.5, -0.02, x_label_text, ha='center', fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.show()