import timeit
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm
import math
import os
import os.path as osp
from pathlib import Path
import sys
import argparse

# internal imports
from tgb.linkproppred.evaluate import Evaluator
from modules.edgebank_predictor import EdgeBankPredictor
from tgb.utils.utils import set_random_seed
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.utils.utils import save_results

def ensemble_test(data, test_mask, neg_sampler, split_mode, subset='True', edgebank=None, metric=None):
    evaluator = Evaluator(name=DATA)
    if subset == 'True':
        print("INFO: Subset is True")
        num_batches = 2
    else:
        print("INFO: Subset is False")
        num_batches = math.ceil(len(data['sources'][test_mask]) / BATCH_SIZE)
    perf_list = []
    logits_list = []  # List to store logits for each batch
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
            logits_list.append(y_pred)  # Append logits to the list
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

    return perf_metrics, logits_list

def get_edgebank_logits():
    # data loading with `numpy`
    dataset = LinkPropPredDataset(name=DATA, root="datasets", preprocess=True)
    metric = dataset.eval_metric
    data = dataset.full_data  

    # get masks
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask

    # data for memory in edgebank
    hist_src = np.concatenate([data['sources'][train_mask]])
    hist_dst = np.concatenate([data['destinations'][train_mask]])
    hist_ts = np.concatenate([data['timestamps'][train_mask]])

    # Set EdgeBank with memory updater
    edgebank = EdgeBankPredictor(
            hist_src,
            hist_dst,
            hist_ts,
            memory_mode=MEMORY_MODE)

    neg_sampler = dataset.negative_sampler

    # loading the validation negative samples
    dataset.load_val_ns()

    # testing ...
    _, logits = ensemble_test(data, val_mask, neg_sampler, split_mode='val', subset='True', edgebank=edgebank, metric=metric)

    return logits

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

args, _ = get_args_edgebank()
SEED = args.seed  # set the random seed for consistency
set_random_seed(SEED)
MEMORY_MODE = args.mem_mode # `unlimited` or `fixed_time_window`
BATCH_SIZE = args.bs
K_VALUE = args.k_value
TIME_WINDOW_RATIO = args.time_window_ratio
DATA = args.data
run_name = args.run
SUBSET = args.subset

start_val = timeit.default_timer()
logits = get_edgebank_logits()
end_val = timeit.default_timer()

print(len(logits))