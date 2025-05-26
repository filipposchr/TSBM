import numpy as np
from scipy.stats import weightedtau
from utils import compute_kendall_tau
import torch
import torch.nn.functional as F
from scipy.stats import weightedtau, rankdata
import math

#Evaluating TBM over MANTRA results

def compute_topk_metrics(preds, labels, k_list=[1, 5, 10, 20], jac=True):
    """
    Computes overlap Top@k% = |pred ∩ true| / k and full-set Jaccard index.
    """
    # Convert to tensors if needed
    if isinstance(preds, np.ndarray):
        preds = torch.from_numpy(preds)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    N = len(preds)
    true_nonzero_set = set(torch.where(labels > 0)[0].tolist())
    stats = {}
    if len(true_nonzero_set) == 0:
        for k in k_list:
            stats[f"Top@{k}%"] = 0.0
        if jac:
            stats["Jaccard"] = 0.0
        return stats

    result_line = []

    for k in k_list:
        topk = max(1, math.ceil(N * (k / 100)))
        pred_topk_idx = set(torch.topk(preds, topk).indices.tolist())
        true_topk_idx = set(torch.topk(labels, topk).indices.tolist())

        intersection = pred_topk_idx & true_topk_idx
        score = len(intersection) / topk
        stats[f"Top@{k}%"] = score
        result_line.append(f"Top@{k}%: {score:.4f}")

    if jac:
        pred_full_set = set(torch.topk(preds, len(true_nonzero_set)).indices.tolist())
        union = pred_full_set | true_nonzero_set
        inter = pred_full_set & true_nonzero_set
        jaccard = len(inter) / len(union)
        stats["Jaccard"] = jaccard
        result_line.append(f"Jaccard: {jaccard:.4f}")

    print(" | ".join(result_line))
    return stats


#real values (TBM)
with open('data/test/Real/shf-bc_scores/graph_edit-facebook_wall_shf_bet.txt', 'r') as f2:
#with open('data/test/Real/scores/graph_edit-slashdot_reply_bet.txt', 'r') as f2:
    values2 = [float(line.strip().split()[1]) for line in f2 if line.strip()]

#apx values (MANTRA)
with open('data/apx/facebook_wall_sfm_apx.txt', 'r') as f1:
    values1 = [float(line.strip()) for line in f1 if line.strip()]

# Convert to NumPy arrays and align length
values1 = np.array(values1[:len(values2)])
values2 = np.array(values2)
N = len(values1)

# --- Weighted Kendall Tau ---
tau, _ = weightedtau(values1, values2)
#print(f"Weighted Kendall Tau default: {tau:.4f}")

kt_all, kt_nonzero = compute_kendall_tau(values2, values1)
compute_topk_metrics(values2, values1)

print(f"Kendall Tau (all nodes):      {kt_all:.4f}")
print(f"Kendall Tau (non-zero only): {kt_nonzero:.4f}")
