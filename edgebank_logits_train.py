import logging
import timeit
import time
import datetime
import os
from tqdm import tqdm
import numpy as np
import shutil
import torch
from utils.utils import NegativeEdgeSampler
from utils.DataLoader import get_idx_data_loader, get_link_pred_data_TRANS_TGB 
from utils.load_configs import get_link_prediction_args
from models.EdgeBank import edge_bank_link_prediction
from utils.DataLoader import Data
from utils.utils import set_random_seed

def main():

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)
    args.subset = args.subset == 'True' # Convert string to boolean

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, dataset = \
        get_link_pred_data_TRANS_TGB(dataset_name=args.dataset_name)


    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
  
    # load the validation negative samples
    dataset.load_val_ns()
    for run in range(args.num_runs):
        start_run = timeit.default_timer()
        set_random_seed(seed=args.seed+run)

        args.save_model_name = f'{args.model_name}_{args.dataset_name}_seed_{args.seed}_run_{run}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        log_start_time = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H_%M_%S")
        fh = logging.FileHandler(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(log_start_time)}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'Configuration is {args}')
        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        # ================================================
        # ============== train & validation ==============
        # ================================================

        # For plotting
        train_logits = []
        train_labels = []

        train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)

        for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
            if args.subset and batch_idx > 2:
                break

            mem_id = 1 #if (batch_idx == 0 and epoch == 0) else 0
            history_data = Data(src_node_ids=train_data.src_node_ids[: train_data_indices[mem_id]],
                dst_node_ids=train_data.dst_node_ids[: train_data_indices[mem_id]],
                node_interact_times=train_data.node_interact_times[: train_data_indices[mem_id]],
                edge_ids=train_data.edge_ids[: train_data_indices[mem_id]],
                labels=train_data.labels[: train_data_indices[mem_id]])

            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

            _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
            batch_neg_src_node_ids = batch_src_node_ids

            positive_edges = (batch_src_node_ids, batch_dst_node_ids)
            negative_edges = (batch_neg_src_node_ids, batch_neg_dst_node_ids)

            # Forward pass
            kwargs = {'batch_src_node_ids': batch_src_node_ids, 
                    'batch_dst_node_ids': batch_dst_node_ids, 
                    'batch_node_interact_times': batch_node_interact_times,
                    'batch_neg_node_interact_times': batch_node_interact_times,
                    'num_neighbors': args.num_neighbors,
                    'batch_neg_src_node_ids': batch_neg_src_node_ids,
                    'batch_neg_dst_node_ids': batch_neg_dst_node_ids,
                    'batch_edge_ids': batch_edge_ids,
                    'time_gap': args.time_gap,
                    'history_data':history_data,
                    'positive_edges':positive_edges,
                    'negative_edges':negative_edges,
                    'edge_bank_memory_mode':args.edge_bank_memory_mode,
                    'time_window_mode':args.time_window_mode,
                    'test_ratio':args.test_ratio,
                    'device': args.device}
            positive_probabilities, negative_probabilities = edge_bank_link_prediction(history_data=kwargs['history_data'],
                                                                                positive_edges=kwargs['positive_edges'],
                                                                                negative_edges=kwargs['negative_edges'],
                                                                                edge_bank_memory_mode=kwargs['edge_bank_memory_mode'],
                                                                                time_window_mode=kwargs['time_window_mode'],
                                                                                time_window_proportion=kwargs['test_ratio'])

            predicts = torch.from_numpy(np.concatenate([positive_probabilities, negative_probabilities])).float()
            predicts = predicts.to(kwargs['device'])
            labels = torch.cat([torch.ones(len(positive_probabilities)), torch.zeros(len(negative_probabilities))], dim=0)
            train_labels.append(labels)
            train_logits.append(predicts)
    # print(train_logits)
    # print(len(train_logits), train_logits[0].shape)

    folder_name = f"./saved_results/EdgeBank/{args.dataset_name}/{args.run_name}/"
    os.makedirs(folder_name, exist_ok=True)
    torch.save(train_logits, f"{folder_name}EdgeBank_{args.dataset_name}_logits_train.pth")
    torch.save(train_labels, f"{folder_name}EdgeBank_{args.dataset_name}_labels_train.pth")
if __name__ == "__main__":
    main()