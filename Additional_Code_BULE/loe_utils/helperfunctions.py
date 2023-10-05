import os
import pandas as pd
import time

# this script contains helper functions to save the results from NTL Bosch and merge them to one dataframe so that it can be compared to the resulots from
# the VT


############################ Data handling IO functions #########################################


def merge_pickle_to_df(MODEL_RESULT_PATH:str,filename_list:list)->pd.DataFrame:
    """
    reads pickle files to pandas dataframe
    params: MODEL_RESULT_PATH: path to pickle files
            filename_list: for which files to choose
                   
    returns: pandas dataframe
    """

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


def save_allresults_pickle(MODEL_RESULT_PATH:str,name:str='NTL',n_runs:int=5)->pd.DataFrame:
        """
        saves all results to pickle file and removes all intermediate results
        params: MODEL_RESULT_PATH: path to pickle files
                name: name of the model
                n_runs: number of runs

        returns: pandas dataframe 
        """
        for run in range(n_runs):
            filename_list=[f for f in os.listdir(MODEL_RESULT_PATH) if (f.endswith(".pkl") and not ("allresults" in f) and (f'run_{run}' in f))]
            results_df=merge_pickle_to_df(MODEL_RESULT_PATH,filename_list)
            SAVE_PATH=os.path.join(MODEL_RESULT_PATH,f"{name}_allresults_run_{run}.pkl")
            results_df.to_pickle(SAVE_PATH)


        # remove all intermediate results
        time.sleep(0.1)
        remove_all_intermediate_results(MODEL_RESULT_PATH)
        # return results_df
            
def concatenate_allresults(MODEL_RESULT_PATH:str,modelname:str='loe_hard',assumed_contamination:float=0.0,n_runs:int=5):

    """
    searches in each subfolder which contains the name of the model an concatenate all results

    params: MODEL_RESULT_PATH: path to pickle files
            name: name of the model
            n_runs: number of runs

    returns: 
    """

    ALLRES_PATH=os.path.join(MODEL_RESULT_PATH,modelname+f'_assumed_contam_{assumed_contamination}_allresults')
    if not os.path.exists(ALLRES_PATH):
            os.makedirs(ALLRES_PATH)

    pathlist=search_files_allresults(MODEL_RESULT_PATH,modelname+f'_{assumed_contamination}_')

    for run in range(n_runs):
        filename_list=[f for f in pathlist if (f.endswith(".pkl") and (f'run_{run}' in f))]
        results_df=merge_pickle_to_df_abspath(filename_list)
        SAVE_PATH=os.path.join(ALLRES_PATH,f"{modelname}_allresults_run_{run}.pkl")
        results_df.to_pickle(SAVE_PATH)



def merge_pickle_to_df_abspath(path_filename_list:list)->pd.DataFrame:
    """
    reads pickle files to pandas dataframe
    params: MODEL_RESULT_PATH: path to pickle files
            filename_list: for which files to choose
                   
    returns: pandas dataframe
    """
    for idx , pickle_filename in enumerate(path_filename_list):
        if idx==0:
            results_df = pd.read_pickle(pickle_filename )
        else:
            results_df=pd.concat([results_df,pd.read_pickle(pickle_filename)])

    return results_df


def search_files_allresults(folder_path:str,modelname:str="loe_hard_0.1_")->list:
    """
    Search for files containing "allresults" in the folder

    params: folder_path: path to folder
            modelname: name of the model default="loe_hard_0.1_"  first is modelname second is assumed contamination
    returns: list of paths to files
    """
    results = []
    # Iterate over all items (folders and files) in the given folder
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        
        # Check if the item is a folder and contains "hard_loe" in its name
        if os.path.isdir(item_path) and modelname  in item and not "allresults" in item:
            
            # Search for files containing "allresults" in the folder
            for root, dirs, files in os.walk(item_path):
                for file in files:
                    file_path = os.path.join(root, file)
                   
                    # Check if the file contains "allresults"
                    if "allresults" in file:
                        results.append(os.path.abspath(file_path))
    
    return results



