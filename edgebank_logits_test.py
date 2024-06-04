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
import torch

# internal imports
from tgb.linkproppred.evaluate import Evaluator
from old_but_gold.modules.edgebank_predictor import EdgeBankPredictor
from tgb.utils.utils import set_random_seed
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.utils.utils import save_results
from utils.metrics import get_link_prediction_metrics

def ensemble_test(data, test_mask, neg_sampler, split_mode, subset='False', edgebank=None, metric=None):
    evaluator = Evaluator(name=DATA)
    if subset == 'True':
        print("INFO: Subset is True")
        num_batches = 2
    else:
        print("INFO: Subset is False")
        num_batches = math.ceil(len(data['sources'][test_mask]) / BATCH_SIZE)
    perf_list = []
    logits_list = []  # List to store logits for each batch
    test_labels = []  # List to store labels for each batch
    pr_aucs = []
    roc_aucs = []

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
            # labels should be a torch tensor
            labels = np.concatenate([np.array([1])] + [np.array([0])]*len(neg_batch))
            test_labels.append(labels)
            # compute MRR
            input_dict = {
                "y_pred_pos": np.array([y_pred[0]]),
                "y_pred_neg": np.array(y_pred[1:]),
                "eval_metric": [metric],
            }
            perf_list.append(evaluator.eval(input_dict)[metric])
            batch_metrics = get_link_prediction_metrics(torch.Tensor(y_pred), torch.Tensor(labels))
            pr_aucs.append(batch_metrics['pr_auc'])
            roc_aucs.append(batch_metrics['roc_auc'])

        # update edgebank memory after each positive batch
        edgebank.update_memory(pos_src, pos_dst, pos_t)

    perf_metrics = float(np.mean(perf_list))
    pr_auc = float(np.mean(pr_aucs))
    roc_auc = float(np.mean(roc_aucs))

    return perf_metrics, pr_auc, roc_auc, logits_list, test_labels

def get_edgebank_logits(subset, data):
    # data loading with `numpy`
    dataset = LinkPropPredDataset(name=data, root="datasets", preprocess=True)
    metric = dataset.eval_metric
    data = dataset.full_data  

    # get masks
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    # data for memory in edgebank
    hist_src = np.concatenate([data['sources'][train_mask]] + [data['sources'][val_mask]])
    hist_dst = np.concatenate([data['destinations'][train_mask]] + [data['sources'][val_mask]])
    hist_ts = np.concatenate([data['timestamps'][train_mask]] + [data['sources'][val_mask]])

    # Set EdgeBank with memory updater
    edgebank = EdgeBankPredictor(
            hist_src,
            hist_dst,
            hist_ts,
            memory_mode=MEMORY_MODE)

    neg_sampler = dataset.negative_sampler

    # loading the validation negative samples
    dataset.load_test_ns()

    # testing ...
    performance, pr_auc, roc_auc, logits, labels = ensemble_test(data, test_mask, neg_sampler, split_mode='test', subset=subset, edgebank=edgebank, metric=metric)

    return performance, pr_auc, roc_auc, logits, labels

def get_args_edgebank():
    parser = argparse.ArgumentParser('*** TGB: EdgeBank ***')
    parser.add_argument('--subset', type=str, help='Subset of the dataset', default='False', choices=['True', 'False'])
    parser.add_argument('-d', '--data', type=str, help='Dataset name', default='tgbl-comment', choices=['tgbl-coin', 'tgbl-comment', 'tgbl-flight', 'tgbl-review', 'tgbl-wiki'])
    parser.add_argument('--run', type=str, help='Run name', default='run1')
    parser.add_argument('--k_value', type=int, help='k_value for computing ranking metrics', default=10)
    parser.add_argument('--seed', type=int, help='Random seed', default=1)
    parser.add_argument('--mem_mode', type=str, help='Memory mode', default='fixed_time_window', choices=['unlimited', 'fixed_time_window'])
    parser.add_argument('--time_window_ratio', type=float, help='Test window ratio', default=0.15)
    parser.add_argument('--bs', type=int, help='Batch size', default=1000)
    parser.add_argument('--run_name', type=str, help='run_name', default="run_name")

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
DATA = args.data
SUBSET = args.subset
#K_VALUE = args.k_value
#TIME_WINDOW_RATIO = args.time_window_ratio
#run_name = args.run

performance, pr_auc, roc_auc, logits, labels = get_edgebank_logits(subset=SUBSET, data=DATA)

# print(labels)
print(len(labels), labels[0].shape, type(logits),type(logits[0]))
print(len(logits), logits[0].shape, type(logits),type(logits[0]))
print("\nRESULTS:")
print(f"MRR: {performance}")
print(f"PR AUC: {pr_auc}")
print(f"ROC AUC: {roc_auc}")

folder_name = f"./saved_results/EdgeBank/{args.data}/{args.run_name}/"
os.makedirs(folder_name, exist_ok=True)
torch.save(logits, f"{folder_name}EdgeBank_{args.data}_logits_test.pth")
torch.save(labels, f"{folder_name}EdgeBank_{args.data}_labels_test.pth")