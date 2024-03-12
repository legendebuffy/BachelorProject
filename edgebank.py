"""
Dynamic Link Prediction with EdgeBank
NOTE: This implementation works only based on `numpy`

Reference: 
    - https://github.com/fpour/DGB/tree/main


"""

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

# ==================
# ==================
# ==================

def test(data, test_mask, neg_sampler, split_mode, subset=False):
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
    if subset:
        num_batches = 2
    else:
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

def get_args():
    parser = argparse.ArgumentParser('*** TGB: EdgeBank ***')
    parser.add_argument('--subset', type=bool, help='Use subset of the data', default=True)
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='tgbl-comment', choices=['tgbl-coin', 'tgbl-comment', 'tgbl-flight', 'tgbl-review', 'tgbl-wiki'])
    parser.add_argument('--run', type=str, help='Run name', default='run1')
    parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=10)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--mem_mode', type=str, help='Memory mode', default='fixed_time_window', choices=['unlimited', 'fixed_time_window'])
    parser.add_argument('--time_window_ratio', type=float, help='Test window ratio', default=0.15)
    parser.add_argument('--bs', type=int, help='Batch size', default=200)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args, sys.argv 

# ==================
# ==================
# ==================

start_overall = timeit.default_timer()

# set hyperparameters
args, _ = get_args()

SEED = args.seed  # set the random seed for consistency
set_random_seed(SEED)
MEMORY_MODE = args.mem_mode # `unlimited` or `fixed_time_window`
BATCH_SIZE = args.bs
K_VALUE = args.k_value
TIME_WINDOW_RATIO = args.time_window_ratio
DATA = args.data
run_name = args.run
subset = args.subset

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

# ==================================================== Test
# loading the validation negative samples
dataset.load_val_ns()

# testing ...
start_val = timeit.default_timer()
perf_metric_val = test(data, val_mask, neg_sampler, split_mode='val', subset=subset)
end_val = timeit.default_timer()

print(f"INFO: val: Evaluation Setting: >>> ONE-VS-MANY <<< ")
print(f"\tval: {metric}: {perf_metric_val: .4f}")
val_time = timeit.default_timer() - start_val
print(f"\tval: Elapsed Time (s): {val_time: .4f}")




# ==================================================== Test
# loading the test negative samples
dataset.load_test_ns()

# testing ...
start_test = timeit.default_timer()
perf_metric_test = test(data, test_mask, neg_sampler, split_mode='test', subset=subset)
end_test = timeit.default_timer()

print(f"INFO: Test: Evaluation Setting: >>> ONE-VS-MANY <<< ")
print(f"\tTest: {metric}: {perf_metric_test: .4f}")
test_time = timeit.default_timer() - start_test
print(f"\tTest: Elapsed Time (s): {test_time: .4f}")

save_results({'model': MODEL_NAME,
              'memory_mode': MEMORY_MODE,
              'data': DATA,
              'subset': subset,
              'run': run_name,
              'seed': SEED,
              'val '+metric: perf_metric_val,
              'val_time': val_time,
              'test '+metric: perf_metric_test,
              'test_time': test_time,
              }, 
    results_filename)
