import json
import pickle
import os
import numpy as np
import pandas as pd
import torch
from glob import glob
from graph import NeighborFinder

def preprocess(data_name):
    u_list, i_list, ts_list = [], [], []
    idx_list = []

    with open(data_name) as f:
        for idx, line in enumerate(f):
            e = line.strip().split('\t')
            # values = [v.split(' ') for v in e]
            u = int(e[0])
            i = int(e[1])
            ts_float = float(e[2])
            ts = int(ts_float)

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            idx_list.append(idx)

    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'idx': idx_list})


def run(data_name):

    PATH = './data/train/Real/processed/{}.txt'.format(data_name)
    OUT_DF = './data/train/Real/processed/ml_{}.csv'.format(data_name)
    OUT_NODE_FEAT = './data/test/Real/processed/ml_{}_node.npy'.format(data_name)

    df = preprocess(PATH)

    num_total_unique_nodes = max(df.u.values.max(), df.i.values.max())
    feat = 129
    rand_feat = np.zeros((num_total_unique_nodes + 1, feat))

    df.to_csv(OUT_DF)
    np.save(OUT_NODE_FEAT, rand_feat)

def run_edited(data_name):
       directory = os.path.dirname(data_name)  # Get the directory path
       base_name = os.path.splitext(os.path.basename(data_name))[0]  # Get the file name without extension

       # Construct the output file path (save in the same directory)
       OUT_NODE_FEAT = os.path.join(directory, f"{base_name}_node.npy")

       df = pd.read_csv(data_name, delimiter=',', skipinitialspace=True, dtype={'u': int, 'i': int})

       num_total_unique_nodes = len(set(df['u']).union(df['i']))

       # NUMBER OF FEATURES
       feat = 128

       rand_feat = np.zeros((num_total_unique_nodes + 1, feat))

       np.save(OUT_NODE_FEAT, rand_feat)

def run_all_csvs_in_directory(directory):
    # Find all .csv files in the directory
    csv_files = glob(os.path.join(directory, "*.csv"))

    # Process each .csv file
    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        run_edited(csv_file)


#TO PROCESS ALL FILES IN A DIRECTORY
#directory_path = "./data/train/Real/processed/seq/"
#print(directory_path)
#run_all_csvs_in_directory(directory_path)

#TO PROCESS ONE FILE
run_edited('./data/test/Real/processed/seq/ml_edit-ltwiktionary.csv')
