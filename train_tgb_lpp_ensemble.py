"""
Train a TG model and evaluate it with TGB package
NOTE:  The task is Transductive Dynamic Link Prediction
"""

import logging
import timeit
import time
import datetime
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
import os.path as osp

from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_pred_data_TRANS_TGB #get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

from tgb.linkproppred.evaluate import Evaluator
from evaluation.tgb_evaluate_LPP import eval_LPP_TGB

# DTU
from models.Ensemble import Ensemble
from models.Ensemble import LogisticRegressionModel


def main():

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, dataset = \
        get_link_pred_data_TRANS_TGB(dataset_name=args.dataset_name)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, test_metric_all_runs = [], []

    # Evaluatign with an evaluator of TGB
    metric = dataset.eval_metric
    evaluator = Evaluator(name=args.dataset_name)
    negative_sampler = dataset.negative_sampler

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


        # DTU ensembling
        if '_' in args.model_name:

            # Gather all the models for the ensemble
            ensemble_models = nn.ModuleList()
            ensemble_models_list = args.model_name.split('_')
            link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)

            print("Building ensemble model...")
            for model_name in ensemble_models_list:
                if model_name == 'TGAT':
                    dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, 
                                            neighbor_sampler=train_neighbor_sampler, time_feat_dim=args.time_feat_dim, 
                                            num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout, device=args.device)
                    
                elif model_name in ['JODIE', 'DyRep', 'TGN']:
                    src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                        compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times)
                    dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, 
                                                neighbor_sampler=train_neighbor_sampler, time_feat_dim=args.time_feat_dim, 
                                                model_name=model_name, num_layers=args.num_layers, num_heads=args.num_heads,
                                                dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift, 
                                                src_node_std_time_shift=src_node_std_time_shift,
                                                dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst, 
                                                dst_node_std_time_shift=dst_node_std_time_shift, device=args.device)
                elif model_name == 'CAWN':
                    dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                            time_feat_dim=args.time_feat_dim, position_feat_dim=args.position_feat_dim, walk_length=args.walk_length,
                                            num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
                elif model_name == 'TCL':
                    dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                        time_feat_dim=args.time_feat_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                        num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
                elif model_name == 'GraphMixer':
                    dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                                time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout, device=args.device)
                elif model_name == 'DyGFormer':
                    dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features, neighbor_sampler=train_neighbor_sampler,
                                                time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                                num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                                max_input_sequence_length=args.max_input_sequence_length, device=args.device)
                    
                else:
                    raise ValueError(f"Wrong value for model_name {model_name}!")
                
                # Combine models in list
                ensemble_model = nn.Sequential(dynamic_backbone, link_predictor)
                ensemble_models.append(ensemble_model)


            # Logistic regressor as the final layer
            combiner = LogisticRegressionModel(input_dim=len(ensemble_models), output_dim=1)

            # Ensemble model
            ensemble = Ensemble(ensemble_models, combiner, ensemble_models_list)

            logger.info(f'model -> {ensemble}')
            logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(ensemble) * 4} B, '
                        f'{get_parameter_sizes(ensemble) * 4 / 1024} KB, {get_parameter_sizes(ensemble) * 4 / 1024 / 1024} MB.')
            
            optimizer = create_optimizer(model=ensemble, optimizer_name=args.optimizer,
                                        learning_rate=args.learning_rate, weight_decay=args.weight_decay)
            
            # Loss-function for individual models: Binary Cross-Entropy Loss on logits
            loss_func = nn.BCEWithLogitsLoss()
            
            ensemble = convert_to_gpu(ensemble, device=args.device)

        # This script is for ensembling only
        else:
            raise ValueError(f"YOU ARE NOT ENSEMBLING, BRO??!"+
                             f"\nargs.mode_name: {args.model_name}\nensemble_models_list: {ensemble_models_list}")

        
        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        # ================================================
        # ============== train & validation ==============
        # ================================================
        val_perf_list = []
        print("Start training epochs..")
        for epoch in range(args.num_epochs):
            start_epoch = timeit.default_timer()
            ensemble.train()

            # store train losses and metrics
            train_losses, train_metrics = [], []

            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                if args.subset == 'True' and batch_idx >= 1:
                    break
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                # Forward pass
                kwargs = {'batch_src_node_ids': batch_src_node_ids, 
                          'batch_dst_node_ids': batch_dst_node_ids, 
                          'batch_node_interact_times': batch_node_interact_times,
                          'batch_neg_node_interact_times': batch_node_interact_times,
                          'num_neighbors': args.num_neighbors,
                          'batch_neg_src_node_ids': batch_neg_src_node_ids,
                          'batch_neg_dst_node_ids': batch_neg_dst_node_ids,
                          'batch_edge_ids': batch_edge_ids,
                          'time_gap': args.time_gap}
                loss, predictions, labels = ensemble.train_step(loss_func, optimizer, train_neighbor_sampler,  **kwargs)

                train_metrics.append(get_link_prediction_metrics(predictions, labels))

                train_idx_data_loader_tqdm.set_description(f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss}')

            # === validation
            # after one complete epoch, evaluate the model on the validation set
            val_metric = ensemble.eval_TGB(neighbor_sampler=full_neighbor_sampler, 
                                      evaluate_idx_data_loader=val_idx_data_loader, evaluate_data=val_data,  
                                      negative_sampler=negative_sampler, evaluator=evaluator, metric=metric,
                                      split_mode='val', k_value=10, num_neighbors=args.num_neighbors, time_gap=args.time_gap,
                                      subset=args.subset)
            val_perf_list.append(val_metric)
            
            epoch_time = timeit.default_timer() - start_epoch
            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}, elapsed time (s): {epoch_time:.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'Validation: {metric}: {val_metric: .4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = [(metric, val_metric, True)]
            early_stop = early_stopping.step(val_metric_indicator, ensemble)

            if early_stop or (args.subset and epoch >= 0):
                break

        # load the best model
        early_stopping.load_checkpoint(ensemble)

        total_train_val_time = timeit.default_timer() - start_run
        logger.info(f'Total train & validation elapsed time (s): {total_train_val_time:.6f}')
        
        # ========================================
        # ============== Final Test ==============
        # ========================================
        start_test = timeit.default_timer()
        # loading the test negative samples
        dataset.load_test_ns()
        test_metric = ensemble.eval_TGB(neighbor_sampler=full_neighbor_sampler, 
                                   evaluate_idx_data_loader=test_idx_data_loader, evaluate_data=test_data,  
                                   negative_sampler=negative_sampler, evaluator=evaluator, metric=metric,
                                   split_mode='test', k_value=10, num_neighbors=args.num_neighbors, time_gap=args.time_gap,
                                   subset=args.subset)
        test_time = timeit.default_timer() - start_test
        logger.info(f'Test elapsed time (s): {test_time:.4f}')
        logger.info(f'Test: {metric}: {test_metric: .4f}')

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
                "data": args.dataset_name,
                "model": args.model_name,
                "run": run,
                "seed": args.seed,
                f"validation {metric}": val_perf_list,
                f"test {metric}": test_metric,
                "test_time": test_time,
                "total_train_val_time": total_train_val_time,
            }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

        logger.info(f"run {run} total elapsed time (s): {timeit.default_timer() - start_run:.4f}")

if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    main()
