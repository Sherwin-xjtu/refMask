#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module: localFind
Author: Sherwin
Date: 4/3/23 4:14 PM
Description:
"""

import pandas as pd


def main(sampfile):
    paf_df = pd.read_csv(sampfile, sep="\t")
    paf_df.columns = ['Qchrom', 'Qlen', 'Qstart', 'Qend', 'Strand', 'Tchrom', 'Tlen', 'Tstart', 'Tend', 'NRM', 'ABL', 'MQ', 'AN1', 'AN2', 'AN3', 'AN4', 'AN5', 'AN6']
    paf_df_new = paf_df.sort_values(by=['Qchrom', 'Qstart'])
    # print(paf_df_new)
    for idx, row in paf_df_new.iterrows():
        if row['Qchrom'] != row['Tchrom']:
            print(row)
            exit()



if __name__ == '__main__':

    paf = '/Volumes/xuwen/DFCI/project2/minimap2results/Brapa_Bnapus.minimap2new.paf'
    main(paf)
