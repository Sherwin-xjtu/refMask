import pandas as pd
import multiprocessing as mp
import os
import tempfile
import time
import numpy as np

# Create a sample DataFrame
file1 = '/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/GMM/gmmFeatures.tsv'
df = pd.read_csv(file1, sep='\t')

def worker(df_chunk, output_file):
    """
    Worker function writes DataFrame chunk to a temporary file.
    """
    df_chunk.to_csv(output_file, index=False)
    return output_file

def parallel_dataframe_processing(df, n_jobs=4):
    """
    Function splits a DataFrame into chunks and processes them in parallel.
    """

    # Create a pool of workers
    pool = mp.Pool(n_jobs)

    # Split DataFrame into chunks
    df_split = np.array_split(df, n_jobs)

    # Create a list to store the temporary file names
    temp_files = []

    # Process each chunk in parallel
    results = []
    for df_chunk in df_split:
        # Create a temporary file for this chunk
        temp_file = tempfile.mktemp(suffix='.txt')
        temp_files.append(temp_file)

        # Start a new worker process to process this chunk
        result = pool.apply_async(worker, args=(df_chunk, temp_file))
        results.append(result)

    # Wait for all worker processes to finish
    for result in results:
        result.get()

    # Combine all the temporary files into a final output file
    final_output_file = 'final_output.txt'
    with open(final_output_file, 'w') as outfile:
        for fname in temp_files:
            with open(fname) as infile:
                outfile.write(infile.read())

    # Delete all the temporary files
    for temp_file in temp_files:
        os.remove(temp_file)

    print("Processing complete. Final output saved as 'final_output.txt'.")

# Test the function
parallel_dataframe_processing(df, n_jobs=4)
