import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import random
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from module_bet import TATKC_TGAT
from scipy.stats import weightedtau
from nx2graphs import load_real_data, load_real_true_TKC, load_train_real_data, load_real_train_true_TKC
from utils import loss_cal_with_soft_topk, loss_cal, compute_kendall_tau, compute_topk_metrics
from torch.optim.lr_scheduler import MultiStepLR

# Argument and global variables
parser = argparse.ArgumentParser('Interface for TATKC experiments')
parser.add_argument('-d', '--data', type=str, help='data sources to use', default='edit-tgwiktioanry')
parser.add_argument('--bs', type=int, default=1500, help='batch_size')
parser.add_argument('--prefix', type=str, default='hello_world', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=1, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=3, help='idx for the gpu to use')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method',
                    default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod',
                    help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['sintime', 'pos_time_aware', 'time', 'hierarchical', 'pos', 'empty'], help='how to use time information',
                    default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument("--local_rank", type=int)


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(1)


BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr

MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}.pth'
LR_MODEL_SAVE_PATH = f'./saved_models/{args.agg_method}-{args.attn_mode}-{args.data}_mlp.pth'
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.data}-{epoch}.pth'

# set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

# Load data
n_feat = np.load('./data/test/Real/processed/seq/ml_{}_node.npy'.format(DATA), allow_pickle=True)
test_real_feat = np.zeros((1400000, 128))


def setSeeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

setSeeds(89)


train_real_src_l, train_real_dst_l, train_real_ts_l, train_real_node_count, train_real_node, train_real_time, \
    train_real_ngh_finder, pass_through_d_list = load_train_real_data(UNIFORM)

nodeList_train_real, train_label_l_real = load_real_train_true_TKC()

test_real_src_l, test_real_dst_l, test_real_ts_l, test_real_node_count, test_real_node, test_real_time, \
    test_real_ngh_finder, test_pass_through_d = load_real_data(dataName=DATA)

nodeList_test_real, test_label_l_real = load_real_true_TKC('{}'.format(DATA))
train_ts_list, test_ts_list, train_real_ts_list = [], [], []


for idx in range(len(nodeList_train_real)):
    train_real_ts_list.append(np.array([train_real_time[idx]] * len(nodeList_train_real[idx])))

test_real_ts_list = np.array([test_real_time] * len(nodeList_test_real))
TEST_BATCH_SIZE = BATCH_SIZE

num_test_instance = len(nodeList_test_real)
num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

for k in range(num_test_batch):
    s_idx = k * TEST_BATCH_SIZE
    e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
    test_src_l_cut = np.array(nodeList_test_real[s_idx:e_idx])
    test_ts_l_cut = np.array(test_real_ts_list[s_idx:e_idx])
    test_real_ngh_finder.preprocess(tuple(test_src_l_cut), tuple(test_ts_l_cut), NUM_LAYER, NUM_NEIGHBORS)

device = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')
ngh_finder = train_real_ngh_finder[0]
tatkc_tgat_model = TATKC_TGAT(
    train_real_ngh_finder[0],
    test_real_feat,
    attn_mode=ATTN_MODE,
    use_time=USE_TIME,
    agg_method=AGG_METHOD,
    num_layers=NUM_LAYER,
    n_head=NUM_HEADS,
    drop_out=DROP_OUT
)

class MLPFilm(nn.Module):
    #Using FiLM and CONCAT
    def __init__(self, node_dim=128, ptd_dim=1, final_dim=128, drop=0.1):
        super().__init__()

        # Project scalar PTD to scale and bias
        self.ptd_scale_proj = nn.Sequential(
            nn.Linear(ptd_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.Sigmoid(),  # keep scale in [0, 1]
        )

        self.ptd_bias_proj = nn.Sequential(
            nn.Linear(ptd_dim, node_dim),
            nn.LayerNorm(node_dim),
        )

        self.ptd_proj = nn.Sequential(
            nn.Linear(ptd_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )


        # Project modulated src_feat to final dimension
        self.input_proj = nn.Sequential(
            nn.Linear(2 * node_dim, final_dim),  # e.g., 256 → 128
            nn.ReLU(),
            nn.Dropout(drop)
        )

        # Deep MLP (no residuals here, but can be added later)
        self.mlp = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.ReLU(),
            nn.Dropout(drop),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(drop),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(drop),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(drop),

            nn.Linear(64, 1)
        )

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, src_feat, ptd_feat):
        if ptd_feat.dim() == 1:
            ptd_feat = ptd_feat.unsqueeze(-1)  # shape [B, 1]

        scale = self.ptd_scale_proj(ptd_feat)  # shape [B, 128]
        bias  = self.ptd_bias_proj(ptd_feat)   # shape [B, 128]

        modulated_feat = src_feat * scale + bias  # FiLM-style modulation

        with torch.no_grad():
            src_norm = torch.norm(src_feat, dim=1).mean().item()
            mod_norm = torch.norm(modulated_feat, dim=1).mean().item()
            rel_weight = mod_norm / (src_norm + 1e-6)
            with open("./saved_models/ptd_vs_src_log_film.txt", "a") as f:
                f.write(f"{src_norm:.6f},{mod_norm:.6f},{rel_weight:.6f}\n")

        ptd_proj = self.ptd_proj(ptd_feat)  # [B, 128]
        x = torch.cat([modulated_feat, ptd_proj], dim=1)
        x = self.input_proj(x)
        return self.mlp(x).squeeze(1)

class MLPFilmDef(nn.Module):
    #Using only FiLM (no CONCAT)
    def __init__(self, node_dim=128, ptd_dim=1, final_dim=128, drop=0.1):
        super().__init__()

        # Project scalar PTD to scale and bias
        self.ptd_scale_proj = nn.Sequential(
            nn.Linear(ptd_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.Sigmoid(),  # keep scale in [0, 1]
        )

        self.ptd_bias_proj = nn.Sequential(
            nn.Linear(ptd_dim, node_dim),
            nn.LayerNorm(node_dim),
        )

        # Project modulated src_feat to final dimension
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, final_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        # Deep MLP (no residuals here, but can be added later)
        self.mlp = nn.Sequential(
            nn.Linear(final_dim, 256),
            nn.ReLU(),
            nn.Dropout(drop),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(drop),

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(drop),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(drop),

            nn.Linear(64, 1)
        )

        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, src_feat, ptd_feat):
        if ptd_feat.dim() == 1:
            ptd_feat = ptd_feat.unsqueeze(-1)  # shape [B, 1]

        scale = self.ptd_scale_proj(ptd_feat)  # shape [B, 128]
        bias  = self.ptd_bias_proj(ptd_feat)   # shape [B, 128]

        modulated_feat = src_feat * scale + bias  # FiLM-style modulation

        with torch.no_grad():
            src_norm = torch.norm(src_feat, dim=1).mean().item()
            mod_norm = torch.norm(modulated_feat, dim=1).mean().item()
            rel_weight = mod_norm / (src_norm + 1e-6)
            with open("./saved_models/ptd_vs_src_log_film.txt", "a") as f:
                f.write(f"{src_norm:.6f},{mod_norm:.6f},{rel_weight:.6f}\n")

        x = self.input_proj(modulated_feat)
        return self.mlp(x).squeeze(1)


MLP_model = MLPFilm().to(device) #FiLM + CONCAT
#MLP_model = MLPFilmDef().to(device) #FiLM

optimizer = torch.optim.Adam(list(tatkc_tgat_model.parameters()) + list(MLP_model.parameters()),lr=LEARNING_RATE)
tatkc_tgat_model.to(device)

print("Epochs: ", NUM_EPOCH)

testing = False

#LOAD MODEL
if testing:
    tatkc_tgat_model.load_state_dict(torch.load('./saved_models/model_TGAT_2.pth'))
    MLP_model.load_state_dict(torch.load('./saved_models/model_MLP_2.pth'))

def normalize_preds(pred, method="minmax"):
    if method == "minmax":
        min_val, max_val = pred.min(), pred.max()
        if max_val - min_val > 1e-6:
            return (pred - min_val) / (max_val - min_val)
        else:
            return torch.zeros_like(pred)
    elif method == "zscore":
        mean, std = pred.mean(), pred.std()
        if std > 1e-6:
            return (pred - mean) / std
        else:
            return torch.zeros_like(pred)
    else:
        raise ValueError("Unsupported normalization method.")


def eval_real_data(hint, tgan, lr_model, sampler, src, ts, label):
    start_time = time.time()
    test_pred_tbc_list = []
    tgan.ngh_finder = sampler
    with torch.no_grad():
        lr_model = lr_model.eval()
        tgan = tgan.eval()
        TEST_BATCH_SIZE = BATCH_SIZE
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in tqdm(range(num_test_batch), desc="Evaluating batches"):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            test_src_l_cut = np.array(src[s_idx:e_idx])
            test_ts_l_cut = np.array(ts[s_idx:e_idx])

            src_embed = tgan.tem_conv(
                src_idx_l=test_src_l_cut,
                cut_time_l=test_ts_l_cut,
                curr_layers=NUM_LAYER,
                num_neighbors=NUM_NEIGHBORS
            )

            ptd = test_pass_through_d[test_src_l_cut - 1].float()
            test_pass_through_degree_batch = ptd.unsqueeze(-1)

            test_pred_tbc = lr_model(src_embed, test_pass_through_degree_batch)
            test_pred_tbc_list.extend(test_pred_tbc.cpu().detach().numpy().tolist())

        #unique_vals = np.unique(test_pred_tbc_list)
        #print(f"Unique prediction values: {len(unique_vals)}")
        #print("Some values:", unique_vals[:10])
        #print(f"Pred min: {test_pred_tbc.min()}, max: {test_pred_tbc.max()}")
        #print(f"Pred mean: {test_pred_tbc.mean()}, std: {test_pred_tbc.std()}")

        with open("test_kendaltau/predicted_bet_cat_mathoverflow.txt", "w") as pred_file:
            for value in test_pred_tbc_list:
                pred_file.write(f"{value}\n")

        label = np.clip(label, a_min=0.0, a_max=None)
        wkt, _ = weightedtau(test_pred_tbc_list, label)

        print("Kendall Tau (default) : ", wkt)
        kt_all, kt_nonzero = compute_kendall_tau(test_pred_tbc_list, label)
        print(f"Kendall Tau (all nodes):      {kt_all:.4f}")
        print(f"Kendall Tau (non-zero only): {kt_nonzero:.4f}")

        if not torch.is_tensor(test_pred_tbc_list):
            test_pred_tbc_list = torch.tensor(test_pred_tbc_list, dtype=torch.float32)
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.float32)

        k_list = [1, 5, 10, 20]
        test_pred_tbc_list = normalize_preds(test_pred_tbc_list, method="zscore")

        compute_topk_metrics(test_pred_tbc_list, label, k_list=k_list)

    end_time = time.time()
    e_time = (end_time - start_time) / 60.0

    return  e_time

def training_tatkc_tgat():
    for epoch in range(NUM_EPOCH):
        epoch_topk_10 = []
        epoch_topk_20 = []
        epoch_topk_1 = []

        tatkc_tgat_model.train()
        MLP_model.train()
        m_loss = []

        graph_indices = list(range(len(train_real_ts_l)))

        for j in tqdm(graph_indices):
            tatkc_tgat_model.ngh_finder = train_real_ngh_finder[j]

            node_list = nodeList_train_real[j]
            label_list = train_label_l_real[j]
            ts_list = train_real_ts_list[j]

            num_train_instance = len(node_list)
            num_train_batch = math.ceil(num_train_instance / BATCH_SIZE)

            pass_through_degree = pass_through_d_list[j]

            for batch_i in range(num_train_batch):
                s_idx = batch_i * BATCH_SIZE
                e_idx = min(num_train_instance, s_idx + BATCH_SIZE)

                src_l_cut = np.array(node_list[s_idx:e_idx])
                ts_l_cut = ts_list[s_idx:e_idx]
                label_l_cut = label_list[s_idx:e_idx]

                optimizer.zero_grad()
                scheduler = MultiStepLR(optimizer, milestones=[10], gamma=0.01)

                src_embed = tatkc_tgat_model.tem_conv(
                    src_idx_l=src_l_cut,
                    cut_time_l=ts_l_cut,
                    curr_layers=NUM_LAYER,
                    num_neighbors=NUM_NEIGHBORS
                )

                true_label = torch.tensor(label_l_cut, dtype=torch.float32).to(device)

                ptd = pass_through_degree[src_l_cut - 1].float()
                pass_through_degree_batch = ptd.unsqueeze(-1)

                pred_bc = MLP_model(src_embed, pass_through_degree_batch)

                topk_stats = compute_topk_metrics(pred_bc, true_label, k_list=[1 ,5, 10, 20, 30], jac=False)

                if topk_stats['Top@1%'] == 0.0 and topk_stats['Top@20%'] == 0.0 and topk_stats['Top@30%'] == 0.0:
                    continue

                #loss = loss_cal(pred_bc, true_label, len(pred_bc), device)
                loss = loss_cal_with_soft_topk(pred_bc, true_label, len(pred_bc), device,
                                               topk_ratio=0.02,  # supervise top-k%
                                               decay=0.9,  # decaying soft weights
                                               alpha=0.3  # balance between ranking and soft loss
                                               )

                epoch_topk_1.append(topk_stats['Top@1%'])
                epoch_topk_10.append(topk_stats['Top@10%'])
                epoch_topk_20.append(topk_stats['Top@20%'])

                loss.backward()

                torch.nn.utils.clip_grad_norm_(list(tatkc_tgat_model.parameters()) + list(MLP_model.parameters()), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                m_loss.append(loss.item())

        avg_topk_1 = np.mean(epoch_topk_1)
        avg_topk_10 = np.mean(epoch_topk_10)
        avg_topk_20 = np.mean(epoch_topk_20)

        print(
            f" Epoch {epoch:02d} Summary : Avg Top@1%: {avg_topk_1:.4f} | Top@10%: {avg_topk_10:.4f} | Top@20%: {avg_topk_20:.4f} ")

        epoch_loss = np.mean(m_loss)
        logger.info(f"Epoch {epoch}: Avg Loss {epoch_loss:.5f}")

if not testing:
    training_tatkc_tgat()

e_time = eval_real_data('test for real data', tatkc_tgat_model, MLP_model, test_real_ngh_finder,
                                              nodeList_test_real, test_real_ts_list, test_label_l_real)

print("Evaluation Time: ", e_time)

#SAVE MODEL
if not testing:
    torch.save(MLP_model.state_dict(), './saved_models/model_MLP_2.pth')
    torch.save(tatkc_tgat_model.state_dict(), './saved_models/model_TGAT_2.pth')
