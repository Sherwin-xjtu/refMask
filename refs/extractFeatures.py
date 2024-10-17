import pysam
import numpy as np
import pandas as pd
import os


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
    else:
        depth_variance = np.var(depths)
        depth_std = np.std(depths)
        r0r_rate = np.sum(r0r_depths) / np.sum(depths)
    
    
    features = [average_mapq, zero_mapq_percentage, depth_variance, depth_std, r0r_rate]
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



def main(bam_file, q0_regions_file, q0q0_count_file):
    
    header_columns = ['chr', 'start', 'end']
    q0_regions = pd.read_csv(q0_regions_file, sep='\t', names=header_columns)
    bam = pysam.AlignmentFile(bam_file, "rb")
    header_columns = ['chr', 'pos', 'num']
    cdf = pd.read_csv(q0q0_count_file, sep='\t', names=header_columns)
    filename = '/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/GMM/gmmFeatures.tsv'
    if not os.path.exists(filename):
        open(filename, 'w')

    gmmFeatures_file = open(filename, 'w')
    header = ['chr', 'start', 'end', 'avg_mapq', 'r0_rate', 'rdv', 'rstd', 'r0r_rate']
    header = '\t'.join(header) + '\n'
    gmmFeatures_file.write(header)
    interval_length = 1001
    merge_size = 3
    for idx, row in q0_regions.iterrows():
        feature_chr = []
        feature_start = []
        feature_end = []
        feature_average_mapq = []
        feature_zero_mapq_percentage = []
        feature_depth_variance = []
        feature_depth_std = []
        feature_ror_num = []
        feature_ror_rate = []

        chromosome = row['chr']
        start_pos = row['start']
        end_pos = row['end']

        # feature_chr.append(chromosome)
        if row['end'] - row['start'] < 1000:
            # intervals = split_intervals(start, end, interval_length)
            # feature_start.append(start_pos)
            # feature_end.append(end_pos)
            features = extractFeatures(chromosome, start_pos, end_pos, bam, cdf)
            # feature_average_mapq.append(features[0])
            # feature_zero_mapq_percentage.append(features[1])
            # feature_depth_variance.append(features[2])
            # feature_depth_std.append(features[3])
            # feature_ror_rate.append(features[4])
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
                # feature_start.append(start_pos)
                # feature_end.append(end_pos)
                features = extractFeatures(chromosome, start_pos, end_pos, bam, cdf)
                # feature_average_mapq.append(features[0])
                # feature_zero_mapq_percentage.append(features[1])
                # feature_depth_variance.append(features[2])
                # feature_depth_std.append(features[3])
                # feature_ror_rate.append(features[4])
                gmm_feature = [chromosome, start_pos, end_pos] + features
                gmm_feature = [str(item) for item in gmm_feature]
                gmm_feature = '\t'.join(gmm_feature) + '\n'
                gmmFeatures_file.write(gmm_feature) 


    # gmmFeatures_df = pd.DataFrame({'chr': feature_chr, 
    #                                'start': feature_start, 
    #                                'end': feature_end,
    #                                'avg_mapq': feature_average_mapq,
    #                                'r0_rate': feature_zero_mapq_percentage,
    #                                'rdv': feature_depth_variance,
    #                                'rstd': feature_depth_std,
    #                                'r0r_rate': feature_ror_rate
    #                                })
                
            
    # gmmFeatures_df.to_csv('/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/GMM/gmmFeatures.tsv', sep='\t', index=False)
    bam.close()
    gmmFeatures_file.close()



if __name__ == '__main__':
    bam_file = "/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/NA12878.recal.merged.new.sorted.chrX.bam"
    q0_regions_file = '/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/NA12878.recal.merged.new.sorted.q0.b.chrX.q0.b.merged.bed'
    q0q0_count_file = '/czlab/xuwen/DFCI/project2/NA12878/analysis/hg38/unique_q0q0region_counts.txt'

    main(bam_file, q0_regions_file, q0q0_count_file)