import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import numpy as np
import scipy.stats as stats
import pandas as pd
import multiprocessing as mp
import tempfile
import os


def smoothing(df1, wins=3, max_iterations=50):
    """Smooth a list using a sliding window."""
    lst = df1['predicted_weights1'].tolist()
    half_window = wins // 2
    lst_copy = lst.copy()
    for _ in range(max_iterations):
        new_lst = lst_copy.copy()
        for i in range(half_window, len(lst) - half_window):
            # Compute the mean of all elements in the window except the center element
            window = lst_copy[i - half_window : i + half_window + 1]
            mean = (sum(window) - lst_copy[i]) / (wins - 1)
            new_lst[i] = round(mean)
        if new_lst == lst_copy:
            break
        lst_copy = new_lst
    return lst_copy

def worker(region, gmm_results_df, bins, temp_file_path):
    region_li = region.rstrip().split('\t')
    tmpdf = gmm_results_df[gmm_results_df['chr'] == region_li[0]]
    tmpdf = tmpdf[tmpdf['start'] >= int(region_li[1])]
    tmpdf = tmpdf[tmpdf['end'] <= int(region_li[2])]
    if tmpdf.shape[0] < bins:
        smoothed_li = tmpdf['predicted_weights1']              
    else:
        smoothed_li = smoothing(tmpdf, bins)
    tmpdf['smoothed_weights1'] = smoothed_li
    tmpdf_file = temp_file_path
    tmpdf.to_csv(tmpdf_file, sep='\t', index=False)
    return tmpdf


def intervalsMerging(smoothed_results_df):
    mask_region = '/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/GMM/maskRegions.tsv'
    smoothed_results_df = smoothed_results_df[smoothed_results_df['label'] == 1]
    smoothed_results_df['group'] = (smoothed_results_df['start'] - smoothed_results_df['end'].shift() > 1000).cumsum()
    a = smoothed_results_df[smoothed_results_df['start'] > 11930041]
    print(a[a['end'] < 12005286])
    df_grouped = smoothed_results_df.groupby('group').agg({
        'chr': 'first',
        'start': 'first',
        'end': 'last',
        'avg_mapq': 'mean',
        'r0_rate': 'mean',
        'rdv': 'mean',
        'rstd': 'mean',
        'depth_mean': 'mean',
        'r0r_depth_mean': 'mean',
        'depth_cv': 'mean',
        'predicted_cluster': 'mean',
        'predicted_weights1': 'mean',
        'smoothed_weights1': 'mean',
        'label': 'mean'
    })

    
    df_grouped['mask'] = df_grouped['label'].round().astype(int)
    print(df_grouped[df_grouped['start'] > 11930041])
    mask_region_df = df_grouped[df_grouped['mask'] == 1]
    column_to_delete = ['avg_mapq', 'r0_rate', 'rdv', 'rstd', 'depth_mean', 'r0r_depth_mean', 'depth_cv', 'predicted_cluster', 'predicted_weights1', 'smoothed_weights1', 'label', 'mask']
    mask_region_df_copy = mask_region_df.copy()
    mask_region_df_copy.drop(columns=column_to_delete, inplace=True)
    mask_region_df_copy.to_csv(mask_region, sep='\t', index=False)


def parallel_processing(gmm_results_df, file2, n_jobs, bins):
    pool = mp.Pool(n_jobs)
    smoothed_region_li = []
    temp_dir = '/czlab/xuwen/DFCI/project2/refs/tmmmp'
    # Create a list to store the temporary file names
    temp_files = []
    final_output_file = '/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/GMM/gmmResults_smoothed_output.txt'
    if os.path.exists(final_output_file):
        smoothed_results_df = pd.read_csv(final_output_file, sep='\t')
        smoothed_results_df['label'] = smoothed_results_df['smoothed_weights1'].round().astype(int)
        intervalsMerging(smoothed_results_df)
    else:
        with open(file2, 'r') as file:
            regions = file.readlines()
            for region in regions:
                temp_file = tempfile.NamedTemporaryFile(suffix='.txt', dir=temp_dir, delete=False)
                temp_file_path = temp_file.name
                temp_files.append(temp_file_path)
                # Start a new worker process to process this chunk
                result = pool.apply_async(worker, args=(region, gmm_results_df,bins, temp_file_path))
                smoothed_region_li.append(result)   
            # Wait for all worker processes to finish
            for result in smoothed_region_li:
                result.get()
        with open(final_output_file, 'w') as outfile:
            for i, fname in enumerate(temp_files):
                with open(fname) as infile:
                    if i > 0:
                        next(infile)
                    outfile.write(infile.read())
        # Delete all the temporary files
        for temp_file in temp_files:
            os.remove(temp_file)
        
        smoothed_results_df = pd.read_csv(final_output_file, sep='\t')
        smoothed_results_df['label'] = smoothed_results_df['smoothed_weights1'].round().astype(int)
        intervalsMerging(smoothed_results_df)


def main(file1, file2, bins):
    gmm_results_df = pd.read_csv(file1, sep='\t')
    n_jobs = 16
    parallel_processing(gmm_results_df, file2, n_jobs, bins)
    
 
if __name__ == '__main__':
    
    features_file = '/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/GMM/gmm_results_file.tsv'
    q0_regions_file = '/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/NA12878.recal.merged.new.sorted.q0.b.chrX.q0.b.merged.bed'
    bins = 3
    main(features_file, q0_regions_file, bins)
