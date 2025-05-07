import numpy as np
import torch
import torch.nn.functional as F
from typing import List
from scipy.stats import weightedtau

def compute_topk_accuracy(preds, labels, k_list=[1, 10, 20, 30]):
    """
    preds, labels: torch tensors of shape [N]
    Returns dict: {"Top@10%": ..., "Top@20%": ...}
    """
    stats = {}
    N = len(preds)
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()

    for k in k_list:
        topk = int(N * (k / 100))
        if topk == 0:
            stats[f"Top@{k}%"] = 0.0
            continue

        pred_topk_idx = torch.topk(preds, topk).indices
        true_topk_idx = torch.topk(labels, topk).indices

        overlap = len(set(pred_topk_idx.tolist()) & set(true_topk_idx.tolist()))
        stats[f"Top@{k}%"] = overlap / topk

    return stats

def temporal_adjacency_list(src_list, dst_list, ts_list, num_nodes):
    tal = [[] for _ in range(num_nodes + 1)]  # +1 for 1-based indexing
    # Populate the temporal adjacency list
    for src, dst, ts in zip(src_list, dst_list, ts_list):
        for u, v, t in zip(src, dst, ts):
            tal[u].append((v, t))

    return tal

def edge_time_range(temporal_edges):
    """
    Replicates the Julia function edge_time_range.

    Args:
        temporal_edges: List of tuples (src, dst, timestamp)

    Returns:
        sorted_min: List of ((src, dst), min_ts) sorted by min_ts
        sorted_max: List of ((src, dst), max_ts) sorted by max_ts
    """
    temporal_edge_min = {}
    temporal_edge_max = {}

    for src, dst, ts in temporal_edges:
        key = (src, dst)
        if key not in temporal_edge_min or ts < temporal_edge_min[key]:
            temporal_edge_min[key] = ts
        if key not in temporal_edge_max or ts > temporal_edge_max[key]:
            temporal_edge_max[key] = ts

    sorted_min = sorted(temporal_edge_min.items(), key=lambda x: x[1])
    sorted_max = sorted(temporal_edge_max.items(), key=lambda x: x[1])

    return sorted_min, sorted_max


def count_less_than(arr: List[int], t: int) -> int:
    left, right = 0, len(arr) - 1
    pos = 0
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] < t:
            pos = mid + 1
            left = mid + 1
        else:
            right = mid - 1
    return pos

def pass_through_degree(temporal_edges, num_nodes):
    sorted_min, sorted_max = edge_time_range(temporal_edges)

    min_arrival_times = [[] for _ in range(num_nodes)]
    for (src, dst), t in sorted_min:
        if dst < num_nodes:
            min_arrival_times[dst].append(t)

    ptd = np.zeros(num_nodes, dtype=int)
    for (src, dst), t in sorted_max:
        if src < num_nodes:
            ptd[src-1] += count_less_than(min_arrival_times[src], t)

    return ptd

def loss_cal_simple(y_out, true_val, num_nodes, device):
    _, order_y_true = torch.sort(-true_val[:num_nodes])

    sample_num = num_nodes * 80
    ind_1 = torch.randint(0, num_nodes, (sample_num,)).long().to(device)
    ind_2 = torch.randint(0, num_nodes, (sample_num,)).long().to(device)

    rank_measure = torch.sign(-1 * (ind_1 - ind_2)).float()

    input_arr1 = y_out[:num_nodes][order_y_true[ind_1]].to(device)
    input_arr2 = y_out[:num_nodes][order_y_true[ind_2]].to(device)

    loss_rank = torch.nn.MarginRankingLoss(margin=1.0).forward(input_arr1, input_arr2, rank_measure)

    return loss_rank

def loss_cal_topk_hybrid(y_out, true_val, num_nodes, device, topk_ratio=0.25, sample_per_node=40):
    true_val = true_val[:num_nodes]
    y_out = y_out[:num_nodes]

    # Sort ground truth BC to get top-k node indices
    sorted_idx = torch.argsort(true_val, descending=True)
    k = max(1, int(topk_ratio * num_nodes))
    topk_idx = sorted_idx[:k]
    rest_idx = sorted_idx[k:]

    # Sample more pairs involving top-k nodes
    num_samples = min(num_nodes * sample_per_node, 20000)
    topk_sample_size = int(num_samples * 0.4)  # 70% from topk vs rest
    uniform_sample_size = num_samples - topk_sample_size

    # Top-k vs rest sampling
    ind_1_topk = topk_idx[torch.randint(0, len(topk_idx), (topk_sample_size,))].to(device)
    ind_2_rest = rest_idx[torch.randint(0, len(rest_idx), (topk_sample_size,))].to(device)

    # Uniform sampling
    ind_1_uniform = torch.randint(0, num_nodes, (uniform_sample_size,), device=device)
    ind_2_uniform = torch.randint(0, num_nodes, (uniform_sample_size,), device=device)

    # Merge all
    ind_1 = torch.cat([ind_1_topk, ind_1_uniform])
    ind_2 = torch.cat([ind_2_rest, ind_2_uniform])

    # Compute rank measure
    rank_measure = torch.sign(true_val[ind_1] - true_val[ind_2]).float()
    valid_mask = rank_measure != 0

    if valid_mask.sum() < 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    input_arr1 = y_out[ind_1[valid_mask]]
    input_arr2 = y_out[ind_2[valid_mask]]
    rank_measure = rank_measure[valid_mask]

    loss_rank = torch.nn.MarginRankingLoss(margin=0.8).forward(input_arr1, input_arr2, rank_measure)
    return loss_rank

def safe_kendall_tau(pred, true):
    pred = np.asarray(pred)
    true = np.asarray(true)

    # Mask to filter out invalid entries
    mask = np.isfinite(pred) & np.isfinite(true)
    pred = pred[mask]
    true = true[mask]

    if len(np.unique(pred)) <= 1 or len(np.unique(true)) <= 1:
        return 0.0  # No meaningful ranking
    kt, _ = weightedtau(pred, true)
    return kt if np.isfinite(kt) else 0.0

