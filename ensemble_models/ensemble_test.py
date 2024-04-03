import math
import timeit
import os
import os.path as osp
from pathlib import Path
import numpy as np
from datetime import datetime
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear
from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TransformerConv

# internal imports
from tgb.utils.utils import get_args, set_random_seed, save_results, _logger
from tgb.linkproppred.evaluate import Evaluator
from ensemble_models.decoder_test import LinkPredictor_Test
from modules.emb_module import GraphAttentionEmbedding
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator
from modules.neighbor_loader import LastNeighborLoader
from modules.memory_module import TGNMemory
from modules.early_stopping import EarlyStopMonitor
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tqdm import tqdm
import sys
import argparse
from modules.edgebank_predictor import EdgeBankPredictor
from tgb.linkproppred.dataset import LinkPropPredDataset

from edgebank import test as edgebank_test

# ========================================================================================================
# ========================================================================================================
# ========================================================================================================
# Get TGN "logits" for ensemble testing

def train(subset:str='True'): # CHANGED to return logits for ensemble methods
    r"""
    Training procedure for TGN model
    This function uses some objects that are globally defined in the current scrips 

    Parameters:
        None
    Returns:
        None
            
    """

    model['memory'].train()
    model['gnn'].train()
    model['link_pred'].train()

    model['memory'].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    pos_logits = []
    neg_logits = []
    for batch in train_loader:

        batch = batch.to(device)
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_dst = torch.randint(
            min_dst_idx,
            max_dst_idx + 1,
            (src.size(0),),
            dtype=torch.long,
            device=device,
        )

        n_id = torch.cat([src, pos_dst, neg_dst]).unique() # Torch of all nodes in batch (including negative)
        n_id, edge_index, e_id = neighbor_loader(n_id) # 
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = model['memory'](n_id)
        z = model['gnn'](
            z,
            last_update,
            edge_index,
            data.t[e_id].to(device),
            data.msg[e_id].to(device),
        )

        pos_out = model['link_pred'](z[assoc[src]], z[assoc[pos_dst]])
        neg_out = model['link_pred'](z[assoc[src]], z[assoc[neg_dst]])

        pos_logits.append(pos_out.detach())
        neg_logits.append(neg_out.detach())

        pos_out_sigmoid = torch.sigmoid(pos_out)
        neg_out_sigmoid = torch.sigmoid(neg_out)
        
        loss = criterion(pos_out_sigmoid, torch.ones_like(pos_out_sigmoid))
        loss += criterion(neg_out_sigmoid, torch.zeros_like(neg_out_sigmoid))

        # Update memory and neighbor loader with ground-truth state.
        model['memory'].update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        model['memory'].detach()
        total_loss += float(loss) * batch.num_events

        if subset=='True':
            break
    
    pos_logits = torch.cat(pos_logits)
    neg_logits = torch.cat(neg_logits)

    return total_loss / train_data.num_events, pos_logits, neg_logits

# Start...
start_overall = timeit.default_timer()

# ========== set parameters...
args, _ = get_args()
print("INFO: Arguments:", args)

DATA = args.data
LR = args.lr
BATCH_SIZE = args.bs
K_VALUE = args.k_value  
NUM_EPOCH = args.num_epoch
SEED = args.seed
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
TOLERANCE = args.tolerance
PATIENCE = args.patience
NUM_RUNS = args.num_run
NUM_NEIGHBORS = 10
SUBSET = args.subset
RUN = args.run

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

MODEL_NAME = 'TGN'
# ==========

experiment_log_dir = os.path.join(logs_save_dir, MODEL_NAME, RUN, DATA + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug(f"Run description: {RUN}")
logger.debug(f"We are using {device}")
logger.debug("=" * 45)
logger.debug(f'Pre-training Dataset: {DATA} (subset={SUBSET})')
logger.debug(f'Method:  {MODEL_NAME}')
logger.debug("=" * 45)


# data loading
dataset = PyGLinkPropPredDataset(name=DATA, root="datasets")
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask
data = dataset.get_TemporalData()
data = data.to(device)
metric = dataset.eval_metric

train_data = data[train_mask]
val_data = data[val_mask]
test_data = data[test_mask]

train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)

logger.debug("Data loaded...")

# Ensure to only sample actual destination nodes as negatives.
min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

# neighborhood sampler
neighbor_loader = LastNeighborLoader(data.num_nodes, size=NUM_NEIGHBORS, device=device)

# define the model end-to-end
memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    MEM_DIM,
    TIME_DIM,
    message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
    aggregator_module=LastAggregator(),
).to(device)

gnn = GraphAttentionEmbedding(
    in_channels=MEM_DIM,
    out_channels=EMB_DIM,
    msg_dim=data.msg.size(-1),
    time_enc=memory.time_enc,
).to(device)

link_pred = LinkPredictor_Test(in_channels=EMB_DIM).to(device) # LinkPredictor_Test REMOVED .sigmoid() from forward

model = {'memory': memory,
         'gnn': gnn,
         'link_pred': link_pred}

optimizer = torch.optim.Adam(
    set(model['memory'].parameters()) | set(model['gnn'].parameters()) | set(model['link_pred'].parameters()),
    lr=LR,
)
criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler

# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    logger.debug('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{DATA}_results.json'

for run_idx in range(NUM_RUNS):
    logger.debug('-------------------------------------------------------------------------------')
    logger.debug(f"INFO: >>>>> Run: {run_idx} <<<<<")
    start_run = timeit.default_timer()

    # set the seed for deterministic results...
    torch.manual_seed(run_idx + SEED)
    set_random_seed(run_idx + SEED)

    # define an early stopper
    save_model_dir = f'{osp.dirname(osp.abspath(__file__))}/saved_models/'
    save_model_id = f'{MODEL_NAME}_{DATA}_{SEED}_{run_idx}'
    early_stopper = EarlyStopMonitor(save_model_dir=save_model_dir, save_model_id=save_model_id, 
                                    tolerance=TOLERANCE, patience=PATIENCE)

    # ==================================================== Train & Validation
    # loading the validation negative samples
    dataset.load_val_ns()

    val_perf_list = []
    start_train_val = timeit.default_timer()
    for epoch in range(1, NUM_EPOCH + 1):
        # training
        start_epoch_train = timeit.default_timer()
        loss, pos_logits_tgn, neg_logits_tgn = train(SUBSET) # RETURN LOGITS FROM TRAIN FUNCTION!!!!!!!!!
        logger.debug(
            f"Epoch: {epoch:02d}, Loss: {loss:.4f}, Training elapsed Time (s): {timeit.default_timer() - start_epoch_train: .4f}"
        )

    #     # validation
    #     start_val = timeit.default_timer()
    #     perf_metric_val = test(val_loader, neg_sampler, split_mode="val", subset=SUBSET)
    #     logger.debug(f"\tValidation {metric}: {perf_metric_val: .4f}")
    #     logger.debug(f"\tValidation: Elapsed time (s): {timeit.default_timer() - start_val: .4f}")
    #     val_perf_list.append(perf_metric_val)

    #     # check for early stopping
    #     if early_stopper.step_check(perf_metric_val, model):
    #         logger.debug(f"INFO: Early Stopping at epoch: {epoch}")
    #         break

    # train_val_time = timeit.default_timer() - start_train_val
    # logger.debug(f"Train & Validation: Elapsed Time (s): {train_val_time: .4f}")


# ==================================================== TESTING LOGITS
print("Positive logits:", pos_logits_tgn)
print("Negative logits:", neg_logits_tgn)



# ========================================================================================================
# ========================================================================================================
# ========================================================================================================
# Get Edgebank "logits" for ensemble testing
def ensemble_test(data, test_mask, neg_sampler, split_mode, subset='False'):
    r"""
    Evaluated the dynamic link prediction
    Evaluation happens as 'one vs. many', meaning that each positive edge is evaluated against many negative edges

    Parameters:
        data: a dataset object
        test_mask: required masks to load the test set edges
        neg_sampler: an object that gives the negative edges corresponding to each positive edge
        split_mode: specifies whether it is the 'validation' or 'test' set to correctly load the negatives
    Returns:
        perf_metric: the result of the performance evaluation
    """
    if subset == 'True':
        print("INFO: Subset is True")
        num_batches = 2
    else:
        print("INFO: Subset is False")
        num_batches = math.ceil(len(data['sources'][test_mask]) / BATCH_SIZE)
    perf_list = []
    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(data['sources'][test_mask]))
        pos_src, pos_dst, pos_t = (
            data['sources'][test_mask][start_idx: end_idx],
            data['destinations'][test_mask][start_idx: end_idx],
            data['timestamps'][test_mask][start_idx: end_idx],
        )
        neg_batch_list = neg_sampler.query_batch(pos_src, pos_dst, pos_t, split_mode=split_mode)
        
        for idx, neg_batch in enumerate(neg_batch_list):
            query_src = np.array([int(pos_src[idx]) for _ in range(len(neg_batch) + 1)])
            query_dst = np.concatenate([np.array([int(pos_dst[idx])]), neg_batch])

            y_pred = edgebank.predict_link(query_src, query_dst)
            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0]]),
                "y_pred_neg": np.array(y_pred[1:]),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])
            
        # update edgebank memory after each positive batch
        edgebank.update_memory(pos_src, pos_dst, pos_t)

    perf_metrics = float(np.mean(perf_list))

    return perf_metrics

def get_args_edgebank():
    parser = argparse.ArgumentParser('*** TGB: EdgeBank ***')
    parser.add_argument('--subset', type=str, help='Subset of the dataset', default='False', choices=['True', 'False'])
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='tgbl-comment', choices=['tgbl-coin', 'tgbl-comment', 'tgbl-flight', 'tgbl-review', 'tgbl-wiki'])
    parser.add_argument('--run', type=str, help='Run name', default='run1')
    parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=10)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--mem_mode', type=str, help='Memory mode', default='fixed_time_window', choices=['unlimited', 'fixed_time_window'])
    parser.add_argument('--time_window_ratio', type=float, help='Test window ratio', default=0.15)
    parser.add_argument('--bs', type=int, help='Batch size', default=200)

    try:
        args_e = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args_e, sys.argv 

start_overall = timeit.default_timer()


# set hyperparameters
args_edgebank, _ = get_args_edgebank()


MEMORY_MODE = args_edgebank.mem_mode # `unlimited` or `fixed_time_window`
TIME_WINDOW_RATIO = args_edgebank.time_window_ratio
run_name = args_edgebank.run

print(f"INFO: Subset: {SUBSET}")
MODEL_NAME = 'EdgeBank'

# data loading with `numpy`
dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)
data = dataset.full_data  
metric = dataset.eval_metric

# get masks
train_mask = dataset.train_mask
val_mask = dataset.val_mask
test_mask = dataset.test_mask

#data for memory in edgebank
hist_src = np.concatenate([data['sources'][train_mask]])
hist_dst = np.concatenate([data['destinations'][train_mask]])
hist_ts = np.concatenate([data['timestamps'][train_mask]])


# #! check if edges are sorted
# sorted = np.all(np.diff(data['timestamps']) >= 0)
# print (" INFO: Edges are sorted: ", sorted)

# Set EdgeBank with memory updater
edgebank = EdgeBankPredictor(
        hist_src,
        hist_dst,
        hist_ts,
        memory_mode=MEMORY_MODE,
        time_window_ratio=TIME_WINDOW_RATIO)

print("==========================================================")
print(f"============*** {MODEL_NAME}: {MEMORY_MODE}: {DATA} ***==============")
print("==========================================================")

evaluator = Evaluator(name=DATA)
neg_sampler = dataset.negative_sampler

# for saving the results...
results_path = f'{osp.dirname(osp.abspath(__file__))}/saved_results'
if not osp.exists(results_path):
    os.mkdir(results_path)
    print('INFO: Create directory {}'.format(results_path))
Path(results_path).mkdir(parents=True, exist_ok=True)
results_filename = f'{results_path}/{MODEL_NAME}_{MEMORY_MODE}_{DATA}_results.json'
print(edgebank.memory)