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
from module_bet import TATKC_TGAT
from scipy.stats import weightedtau
from nx2graphs import load_real_data, load_real_true_TKC, load_train_real_data, load_real_train_true_TKC
from utils import loss_cal, compute_kendall_tau, compute_topk_metrics, normalized_supremum_deviation, normalized_mae
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

# Argument and global variables
parser = argparse.ArgumentParser('Interface for TATKC experiments')
parser.add_argument('-d', '--data', type=str, help='data sources to use', default='edit-tgwiktioanry')
parser.add_argument('--bs', type=int, default=1500, help='batch_size')
parser.add_argument('--prefix', type=str, default='hello_world', help='prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')
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
parser.add_argument('--test', action='store_true', help='Run in test mode')
parser.add_argument('--bet', choices=['sh', 'sfm'], default='sh', help='Betweenness mode: sh (shortest) or sfm (shortest-foremost)')

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
testing = args.test
bet_mode = args.bet

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
    train_real_ngh_finder, pass_through_d_list, earl_arrival_list = load_train_real_data(UNIFORM)

nodeList_train_real, train_label_l_real = load_real_train_true_TKC(bet_mode)

test_real_src_l, test_real_dst_l, test_real_ts_l, test_real_node_count, test_real_node, test_real_time, \
    test_real_ngh_finder, test_pass_through_d,  test_earl_arrival = load_real_data(dataName=DATA)

nodeList_test_real, test_label_l_real = load_real_true_TKC('{}'.format(DATA), bet_mode)
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

class MLPWithPTD(nn.Module):
    def __init__(self, node_dim=128, ptd_dim=128, drop=0.1):
        super().__init__()

        self.ptd_proj = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        self.fc_1 = nn.Linear(node_dim + ptd_dim, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop)

        # Weight initialization
        for layer in [self.fc_1, self.fc_2, self.fc_3]:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, src_feat, ptd):
        # ptd: [B] or [B, 1]
        if ptd.dim() == 1:
            ptd = ptd.unsqueeze(-1)  # [B, 1]

        ptd_embed = self.ptd_proj(ptd)  # [B, ptd_dim]
        x = torch.cat([src_feat, ptd_embed], dim=1)  # [B, node_dim + ptd_dim]

        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        out = self.fc_3(x).squeeze(1)
        return out


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

        ptd_proj = self.ptd_proj(ptd_feat)  # [B, 128]
        x = torch.cat([modulated_feat, ptd_proj], dim=1)
        x = self.input_proj(x)
        return self.mlp(x).squeeze(1)


class MLPFilmThreeFeatCF(nn.Module):
    def __init__(self, node_dim=128, scalar_dim=1, final_dim=128, drop=0.1):
        super().__init__()

        # Project scalar PTD to FiLM scale/bias
        self.ptd_scale_proj = nn.Sequential(
            nn.Linear(scalar_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.Sigmoid()
        )
        self.ptd_bias_proj = nn.Sequential(
            nn.Linear(scalar_dim, node_dim),
            nn.LayerNorm(node_dim)
        )

        # Project PTD for concat
        self.ptd_proj = nn.Sequential(
            nn.Linear(scalar_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        # Project arrival time scalar to match dim
        self.arrival_proj = nn.Sequential(
            nn.Linear(scalar_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        # Combined projection: modulated src_feat + ptd_proj + arrival_proj = 3 * node_dim
        self.input_proj = nn.Sequential(
            nn.Linear(3 * node_dim, final_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )

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

    def forward(self, src_feat, ptd_feat, arrival_feat):
        if ptd_feat.dim() == 1:
            ptd_feat = ptd_feat.unsqueeze(-1)
        if arrival_feat.dim() == 1:
            arrival_feat = arrival_feat.unsqueeze(-1)

        # FiLM modulation with PTD
        scale = self.ptd_scale_proj(ptd_feat)
        bias = self.ptd_bias_proj(ptd_feat)
        modulated_feat = src_feat * scale + bias

        # Project both scalar features
        ptd_emb = self.ptd_proj(ptd_feat)
        arrival_emb = self.arrival_proj(arrival_feat)

        # Concatenate all three representations
        x = torch.cat([modulated_feat, ptd_emb, arrival_emb], dim=1)  # [B, 384]
        x = self.input_proj(x)
        return self.mlp(x).squeeze(1)


class MLPWithThreeFeatC(nn.Module):
    def __init__(self, node_dim=128, scalar_dim=1, aux_dim=128, drop=0.1):
        super().__init__()

        # Project PTD feature (scalar) to 128-dim
        self.ptd_proj = nn.Sequential(
            nn.Linear(scalar_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        # Project arrival feature (scalar) to 128-dim
        self.arrival_proj = nn.Sequential(
            nn.Linear(scalar_dim, node_dim),
            nn.LayerNorm(node_dim),
            nn.ReLU(),
            nn.Dropout(drop)
        )

        # Final input dim = node_dim + ptd_dim + arrival_dim = 128 + 128 + 128
        self.fc_1 = nn.Linear(node_dim + 2 * aux_dim, 128)
        self.fc_2 = nn.Linear(128, 64)
        self.fc_3 = nn.Linear(64, 1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(drop)

        for layer in [self.fc_1, self.fc_2, self.fc_3]:
            nn.init.kaiming_normal_(layer.weight)

    def forward(self, src_feat, ptd, arrival):
        # ptd and arrival are expected to be [B] or [B, 1]
        if ptd.dim() == 1:
            ptd = ptd.unsqueeze(-1)
        if arrival.dim() == 1:
            arrival = arrival.unsqueeze(-1)

        ptd_embed = self.ptd_proj(ptd)            # [B, 128]

        arrival_feat = (arrival - arrival.mean()) / (arrival.std() + 1e-6)
        arrival_embed = self.arrival_proj(arrival_feat)  # [B, 128]

        x = torch.cat([src_feat, ptd_embed, arrival_embed], dim=1)  # [B, 384]
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        out = self.fc_3(x).squeeze(1)  # [B]
        return out

#MLP MODELS
#MLP_model = MLPFilm().to(device) #FiLM + CONCAT
MLP_model = MLPWithPTD().to(device) #CONCAT
#MLP_model = MLPWithThreeFeatC().to(device) #three features CONCAT
#MLP_model = MLPFilmThreeFeatCF().to(device) #three features FiLM + CONCAT

optimizer = torch.optim.Adam(list(tatkc_tgat_model.parameters()) + list(MLP_model.parameters()),lr=LEARNING_RATE)
tatkc_tgat_model.to(device)

print("Epochs: ", NUM_EPOCH)

#LOAD MODEL
if testing:
    print("Running in test mode...")
    tatkc_tgat_model.load_state_dict(torch.load('./saved_models/model_TGAT_1.pth', weights_only=True))
    MLP_model.load_state_dict(torch.load('./saved_models/model_MLP_1.pth', weights_only=True))

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

            earl_arrival = test_earl_arrival[test_src_l_cut - 1].float()
            earl_arrival_batch = earl_arrival.unsqueeze(-1)

            if isinstance(MLP_model, (MLPFilm, MLPWithPTD)): #two features
                test_pred_tbc = lr_model(src_embed, test_pass_through_degree_batch)
            else: #three features
                test_pred_tbc = lr_model(src_embed, test_pass_through_degree_batch, earl_arrival_batch)

            test_pred_tbc_list.extend(test_pred_tbc.cpu().detach().numpy().tolist())

        with open("test_kendaltau/predicted_values.txt", "w") as pred_file:
            for value in test_pred_tbc_list:
                pred_file.write(f"{value}\n")

        label = np.clip(label, a_min=0.0, a_max=None)
        wkt, _ = weightedtau(test_pred_tbc_list, label)

        sd_value = normalized_supremum_deviation(test_pred_tbc_list, label)
        norm_mae = normalized_mae(test_pred_tbc_list, label)

        print("Formal Supremum Deviation Norm:", sd_value)
        print("MAE Norm:", norm_mae)

        print("Kendall Tau (default) : ", wkt)
        kt_all, kt_nonzero = compute_kendall_tau(test_pred_tbc_list, label)
        print(f"Kendall Tau (all nodes):      {kt_all:.4f}")
        print(f"Kendall Tau (non-zero only): {kt_nonzero:.4f}")

        if not torch.is_tensor(test_pred_tbc_list):
            test_pred_tbc_list = torch.tensor(test_pred_tbc_list, dtype=torch.float32)
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.float32)

        k_list = [1, 5, 10, 20]
        #test_pred_tbc_list = normalize_preds(test_pred_tbc_list, method="zscore")

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
            earl_arrival = earl_arrival_list[j]

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

                earl_arr = earl_arrival[src_l_cut - 1].float()
                earl_arr_batch = earl_arr.unsqueeze(-1)

                if isinstance(MLP_model, (MLPFilm, MLPWithPTD)):  # two features
                    pred_bc = MLP_model(src_embed, pass_through_degree_batch)
                else:  # three features
                    pred_bc = MLP_model(src_embed, pass_through_degree_batch, earl_arr_batch)

                topk_stats = compute_topk_metrics(pred_bc, true_label, k_list=[1 ,5, 10, 20, 30], jac=False)

                if topk_stats['Top@1%'] == 0.0 and topk_stats['Top@20%'] == 0.0 and topk_stats['Top@30%'] == 0.0:
                    continue

                loss = loss_cal(pred_bc, true_label, len(pred_bc), device)

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


#SAVE MODEL
if not testing:
    print("Running in training mode...")
    torch.save(MLP_model.state_dict(), './saved_models/model_MLP_2.pth')
    torch.save(tatkc_tgat_model.state_dict(), './saved_models/model_TGAT_2.pth')
