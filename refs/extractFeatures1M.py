import pysam
import numpy as np
import pandas as pd
import os
import multiprocessing as mp
import tempfile
import time
import numpy as np

# depth
def calculateDepth(chrom, start_pos, end_pos, bam):
    depths = []
    for pileupcolumn in bam.pileup(chrom, start_pos, end_pos, stepper="nofilter"):
        # depths.append(pileupcolumn.n)
        if pileupcolumn.pos >= start_pos and pileupcolumn.pos < end_pos:
            depths.append(pileupcolumn.n)
    return depths


def extractFeatures(chrom, start_pos, end_pos, bam, q0q0_count_file):
    # average MAPQ and the rate of reads with 0 MAPQ
    # features = []
    total_mapq = 0
    count = 0
    zero_mapq_count = 0
    for read in bam.fetch(chrom, start_pos, end_pos):
        total_mapq += read.mapping_quality
        count += 1
        if read.mapping_quality == 0:
            zero_mapq_count += 1
    if count == 0:
        average_mapq = 0
        zero_mapq_percentage = 0
    else:
        average_mapq = total_mapq / count
        zero_mapq_percentage = (zero_mapq_count / count)
    depths = calculateDepth(chrom, start_pos, end_pos, bam)
    r0r_depths = countR0r(chrom, start_pos, end_pos, q0q0_count_file)
    if len(depths) == 0:
        depth_variance = 0
        depth_std = 0
        r0r_rate = 0
        depth_mean = 0
        r0r_depth_mean = 0
        depth_cv = 0
    else:
        depth_variance = np.var(depths)
        depth_std = np.std(depths)
        depth_mean = np.mean(depths)
        r0r_depth_mean = np.mean(r0r_depths)
        r0r_rate = np.sum(r0r_depths) / np.sum(depths)
        depth_cv = depth_std / depth_mean
    
    
    features = [average_mapq, zero_mapq_percentage, depth_variance, depth_std, r0r_rate, depth_mean, r0r_depth_mean, depth_cv]
    return features


def countR0r(chrom, start, end, cdf):
    r0r_depths = []
    tmp = 0
    rcdf = cdf[cdf['chr'] == chrom]
    rcdf = rcdf[rcdf['pos'] >= start]
    rcdf = rcdf[rcdf['pos'] < end]
    r0r_depths = rcdf['num']
    return r0r_depths


def split_intervals(start, end, interval_length):
    intervals = []
    # if end -start < 1000:
    #     if start > 1000:
    #         intervals.append([start - 1000, start -1])
    #         intervals.append([start, end])
    #         intervals.append([end + 1, 999])
    current_start = start
    current_end = start + interval_length - 1
    while current_end <= end:
        intervals.append([current_start, current_end])
        current_start = current_end + 1
        current_end = current_start + interval_length - 1
    if current_start <= end:
        intervals.append([current_start, end])
    return intervals


def worker(df_chunk, bam_file, cdf, interval_length, output_file):
    """
    Worker function writes DataFrame chunk to a temporary file.
    """
    bam = pysam.AlignmentFile(bam_file, "rb")
    filename = output_file

    gmmFeatures_file = open(filename, 'w')
    header = ['chr', 'start', 'end', 'avg_mapq', 'r0_rate', 'rdv', 'rstd', 'r0r_rate', 'depth_mean', 'r0r_depth_mean', 'depth_cv']
    header = '\t'.join(header) + '\n'
    gmmFeatures_file.write(header)

    for idx, row in df_chunk.iterrows():

        chromosome = row['chr']
        start_pos = row['start']
        end_pos = row['end']

        if row['end'] - row['start'] < 1000:
            features = extractFeatures(chromosome, start_pos, end_pos, bam, cdf)
            gmm_feature = [chromosome, start_pos, end_pos] + features
            gmm_feature = [str(item) for item in gmm_feature]
            gmm_feature = '\t'.join(gmm_feature) + '\n'
            gmmFeatures_file.write(gmm_feature)      
        else:
            start = start_pos
            end = end_pos
            intervals = split_intervals(start, end, interval_length)
            for interval in intervals:
                start_pos = interval[0]
                end_pos = interval[1]
                features = extractFeatures(chromosome, start_pos, end_pos, bam, cdf)
                gmm_feature = [chromosome, start_pos, end_pos] + features
                gmm_feature = [str(item) for item in gmm_feature]
                gmm_feature = '\t'.join(gmm_feature) + '\n'
                gmmFeatures_file.write(gmm_feature) 
    bam.close()
    return output_file


def parallel_dataframe_processing(df, cdf, interval_length, bam_file, n_jobs):
    """
    Function splits a DataFrame into chunks and processes them in parallel.
    """

    # Create a pool of workers
    pool = mp.Pool(n_jobs)

    # Split DataFrame into chunks
    df_split = np.array_split(df, n_jobs)
    temp_dir = '/czlab/xuwen/DFCI/project2/refs/tmp'
    # Create a list to store the temporary file names
    temp_files = []

    # Process each chunk in parallel
    results = []
    for df_chunk in df_split:
        # Create a temporary file for this chunk
        # Create a temporary file for this chunk
        temp_file = tempfile.NamedTemporaryFile(suffix='.txt', dir=temp_dir, delete=False)
        temp_file_path = temp_file.name
        temp_files.append(temp_file_path)

        # Start a new worker process to process this chunk
        result = pool.apply_async(worker, args=(df_chunk, bam_file, cdf, interval_length, temp_file_path))
        results.append(result)

    # Wait for all worker processes to finish
    for result in results:
        result.get()

    # Combine all the temporary files into a final output file
    final_output_file = '/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/GMM/gmmFeatures_final_output.txt'
    with open(final_output_file, 'w') as outfile:
        for i, fname in enumerate(temp_files):
            with open(fname) as infile:
                if i > 0:
                    next(infile)
                outfile.write(infile.read())
    # Delete all the temporary files
    for temp_file in temp_files:
        os.remove(temp_file)




def main(bam_file, q0_regions_file, q0q0_count_file):
    
    header_columns = ['chr', 'start', 'end']
    q0_regions = pd.read_csv(q0_regions_file, sep='\t', names=header_columns)
    
    header_columns = ['chr', 'pos', 'num']
    cdf = pd.read_csv(q0q0_count_file, sep='\t', names=header_columns)
    interval_length = 1001
    n_jobs = 16
    parallel_dataframe_processing(q0_regions, cdf, interval_length, bam_file, n_jobs)




if __name__ == '__main__':
    bam_file = "/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/NA12878.recal.merged.new.sorted.chrX.bam"
    q0_regions_file = '/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/NA12878.recal.merged.new.sorted.q0.b.chrX.q0.b.merged.bed'
    q0q0_count_file = '/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/unique_q0q0region_counts.txt'

    main(bam_file, q0_regions_file, q0q0_count_file)