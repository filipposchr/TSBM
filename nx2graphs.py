import pandas as pd
import numpy as np
import networkx as nx
import pickle
import os
import torch
from graph import NeighborFinder
from utils import temporal_adjacency_list, pass_through_degree, compute_earliest_arrival

def load_real_data(dataName):
    g_df = pd.read_csv('./data/test/Real/processed/seq/ml_{}.csv'.format(dataName))
    print(f"Testing dataset: {dataName}")
    src, dst, ts = g_df['u'].values, g_df['i'].values, g_df['ts'].values

    num_nodes = len(set(np.unique(np.hstack([src, dst]))))

    src_list = g_df.u.values
    dst_list = g_df.i.values
    ts_list = g_df.ts.values

    max_idx = max(g_df.u.values.max(), g_df.i.values.max())
    node_count = len(set(np.unique(np.hstack([g_df.u.values, g_df.i.values]))))
    node_list = np.unique(np.hstack([src_list, dst_list]))
    maxTime_list = max(g_df.ts.values)
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_list, dst_list, g_df.idx.values, ts_list):
        adj_list[dst].append((src, eidx, ts))

    ngh_finder = NeighborFinder(adj_list, uniform=False)

    temporal_edges = list(zip(src_list, dst_list, ts_list))

    pass_through_d = pass_through_degree(temporal_edges, num_nodes)
    pass_through_d = torch.tensor(pass_through_d, dtype=torch.float32)

    earl_arrival = compute_earliest_arrival(num_nodes, src_list, dst_list, ts_list)
    earl_arrival = torch.tensor(earl_arrival, dtype=torch.float32)

    return src_list, dst_list, ts_list, node_count, node_list, maxTime_list, ngh_finder, pass_through_d, earl_arrival

def load_train_real_data(UNIFORM, save_dir="graph_features"):
    src_list, dst_list, ts_list, node_count, node_list, maxTime_list, ngh_finder = [], [], [], [], [], [], []
    pass_through_d_list = []
    earl_arrival_list = []

    train_real_datasets = ['edit-mrwiktionary', 'edit-siwiktionary', 'edit-stwiktionary', 'edit-wowiktionary',
                           'edit-tkwiktionary', 'edit-aywiktionary', 'edit-anwiktionary', 'edit-pawiktionary',
                           'edit-iawiktionary', 'edit-sowiktionary', 'edit-tiwiktionary', 'edit-sswiktionary',
                           'edit-gnwiktionary', 'edit-iewiktionary', 'edit-pnbwiktionary', 'edit-gdwiktionary',
                           'edit-srwikiquote', 'edit-nowikiquote', 'edit-etwikiquote',
                           'edit-jawikiquote', 'edit-mtwiktionary', 'edit-dvwiktionary', 'edit-iuwiktionary',
                           'edit-kuwikiquote', 'edit-suwiktionary', 'edit-nawiktionary', 'edit-miwiktionary',
                           'edit-roa_rupwiktionary', 'edit-tpiwiktionary', 'edit-gdwiktionary',
                           'edit-lnwiktionary', 'edit-omwiktionary', 'edit-sgwiktionary', 'edit-quwiktionary',
                           'edit-rwwiktionary', 'edit-stwikipedia', 'edit-olowikipedia', 'edit-tnwikipedia',
                           'edit-ffwikipedia', 'edit-dzwikipedia', 'edit-tyvwikipedia',
                           'edit-xhwikipedia',  'edit-tswikipedia', 'edit-bgwikiquote',
                           'edit-idwikiquote', 'edit-aswikiquote', 'edit-yiwikiquote', 'edit-sawikiquote']

    print("Total training graphs : ", len(train_real_datasets))

    #Save the node features (first time)
    save_all_graph_features(train_real_datasets, save_dir="graph_features")

    for dataset_name in train_real_datasets:
        # Load the dataset
        g_df = pd.read_csv(f'./data/train/Real/processed/seq/ml_{dataset_name}.csv')
        src_list.append(g_df.u.values)
        dst_list.append(g_df.i.values)
        ts_list.append(g_df.ts.values)
        max_idx = max(g_df.u.values.max(), g_df.i.values.max())

        # Load precomputed features from pickle
        file_path = os.path.join(save_dir, f"{dataset_name}_features.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                graph_features = pickle.load(f)
            #print(f"Loaded features for {dataset_name} from {file_path}")
        else:
            raise FileNotFoundError(
                f"Features for {dataset_name} not found in {save_dir}. Run save_all_graph_features first.")

        pass_through_d = graph_features.get("pass_through_d")
        pass_through_d_list.append(pass_through_d)

        earl_arrival = graph_features.get("earl_arrival")
        earl_arrival_list.append(earl_arrival)

        # Populate adjacency list for NeighborFinder
        adj_list = [[] for _ in range(max_idx + 1)]
        for src, dst, eidx, ts in zip(src_list[-1], dst_list[-1], g_df.idx.values, ts_list[-1]):
            adj_list[dst].append((src, eidx, ts))

        # Add graph-specific details
        node_count.append(len(set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))))

        node_list.append(np.unique(np.hstack([src_list[-1], dst_list[-1]])))
        maxTime_list.append(max(g_df.ts.values))
        ngh_finder.append(NeighborFinder(adj_list, uniform=UNIFORM))

    return (src_list, dst_list, ts_list, node_count, node_list, maxTime_list, ngh_finder, pass_through_d_list, earl_arrival_list)


def load_real_true_TKC(dataName, bet_mode='sh'):
    if bet_mode == 'sh':
        print("  Temporal Shortest Betweenness...")
        path = f'./data/test/Real/scores/graph_{dataName}_bet.txt'
    elif bet_mode == 'sfm':
        print("  Temporal Shortest-Foremost Betweenness...")
        path = f'./data/test/Real/shf-bc_scores/graph_{dataName}_shf_bet.txt'
    else:
        raise ValueError(f"Unknown betweenness mode: {bet_mode}")

    g_df = pd.read_csv(path, names=['node_id', 'score'], sep=' ')
    test_nodeList = g_df['node_id'].astype(int).tolist()
    test_tkcList = g_df['score'].tolist()

    return test_nodeList, test_tkcList


def load_real_train_true_TKC(bet_mode='sh'):
    train_nodeList, train_true_tkc = [], []

    train_real_datasets = ['edit-mrwiktionary', 'edit-siwiktionary', 'edit-stwiktionary', 'edit-wowiktionary',
                           'edit-tkwiktionary', 'edit-aywiktionary', 'edit-anwiktionary', 'edit-pawiktionary',
                           'edit-iawiktionary', 'edit-sowiktionary', 'edit-tiwiktionary', 'edit-sswiktionary',
                           'edit-gnwiktionary', 'edit-iewiktionary', 'edit-pnbwiktionary', 'edit-gdwiktionary',
                           'edit-srwikiquote', 'edit-nowikiquote', 'edit-etwikiquote',
                           'edit-jawikiquote', 'edit-mtwiktionary', 'edit-dvwiktionary', 'edit-iuwiktionary',
                           'edit-kuwikiquote', 'edit-suwiktionary', 'edit-nawiktionary', 'edit-miwiktionary',
                           'edit-roa_rupwiktionary', 'edit-tpiwiktionary', 'edit-gdwiktionary',
                           'edit-lnwiktionary', 'edit-omwiktionary', 'edit-sgwiktionary', 'edit-quwiktionary',
                           'edit-rwwiktionary', 'edit-stwikipedia', 'edit-olowikipedia', 'edit-tnwikipedia',
                           'edit-ffwikipedia', 'edit-dzwikipedia', 'edit-tyvwikipedia',
                           'edit-xhwikipedia',  'edit-tswikipedia', 'edit-bgwikiquote',
                           'edit-idwikiquote', 'edit-aswikiquote', 'edit-yiwikiquote', 'edit-sawikiquote']

    for index in range(len(train_real_datasets)):
        dataset_name = train_real_datasets[index]

        if bet_mode == 'sh':
            path = f'./data/train/Real/scores/bc_scores/{dataset_name}_bc.txt'
        elif bet_mode == 'sfm':
            path = f'./data/train/Real/scores/shf_scores/{dataset_name}_bc.txt'
        else:
            raise ValueError(f"Unknown betweenness mode: {bet_mode}")

        g_df = pd.read_csv(path, names=['node_id', 'score'], sep=' ')

        nodeList = g_df['node_id'].astype(int).tolist()
        tkcList = g_df['score'].tolist()

        train_nodeList.append(nodeList)
        train_true_tkc.append(tkcList)

    return train_nodeList, train_true_tkc


def preprocess_data(csv_file):
    """
    Reads a temporal graph from a CSV file and returns edge_index and edge_time tensors.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        edge_index (torch.Tensor): Shape [2, num_edges], source and destination nodes.
        edge_time (torch.Tensor): Shape [num_edges], timestamps for each edge.
    """
    # Read the CSV file, skip the first row (header), and use only necessary columns
    df = pd.read_csv(csv_file, skiprows=1, header=None, usecols=[1, 2, 3])

    # Extract the source, destination, and time columns
    source_nodes = df.iloc[:, 0].tolist()
    destination_nodes = df.iloc[:, 1].tolist()
    timestamps = df.iloc[:, 2].tolist()

    # Convert to PyTorch tensors
    edge_index = torch.tensor([source_nodes, destination_nodes], dtype=torch.long)
    edge_time = torch.tensor(timestamps, dtype=torch.float)

    return edge_index, edge_time

def save_all_graph_features(train_real_datasets, save_dir="graph_features"):
    os.makedirs(save_dir, exist_ok=True)
    for dataset_name in train_real_datasets:
        #print(f"      Processing dataset: {dataset_name}")
        g_df = pd.read_csv(f'./data/train/Real/processed/seq/ml_{dataset_name}.csv')
        src, dst, ts = g_df['u'].values, g_df['i'].values, g_df['ts'].values
        num_nodes = len(set(np.unique(np.hstack([src, dst]))))

        temporal_edges = list(zip(src, dst, ts))

        pass_through_d = pass_through_degree(temporal_edges,num_nodes)
        pass_through_d = torch.tensor(pass_through_d, dtype=torch.float32)

        earl_arrival = compute_earliest_arrival(num_nodes, src, dst, ts)
        earl_arrival = torch.tensor(earl_arrival, dtype=torch.float32)

        graph_features = {
            "pass_through_d" : pass_through_d,
            "earl_arrival" : earl_arrival
        }

        # Save graph features to pickle
        file_path = os.path.join(save_dir, f"{dataset_name}_features.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(graph_features, f)
