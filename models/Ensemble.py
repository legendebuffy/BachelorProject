import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from models.EdgeBank import edge_bank_link_prediction
from utils.DataLoader import Data
from evaluation.tgb_evaluate_LPP import query_pred_edge_batch
from itertools import combinations
from utils.metrics import get_link_prediction_metrics

class Ensemble(nn.Module):

    def __init__(self, base_models, combiner, model_names):
        super(Ensemble, self).__init__()
        self.base_models = nn.ModuleList(base_models)
        self.combiner = combiner
        self.model_names = model_names


    def compute_embeddings(self, model, model_name, kwargs, positive):
        if model_name in ['TGAT', 'CAWN', 'TCL']:
            batch_src_node_embeddings, batch_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=kwargs['batch_src_node_ids'],
                                                                dst_node_ids=kwargs['batch_dst_node_ids'],
                                                                node_interact_times=kwargs['batch_node_interact_times'],
                                                                num_neighbors=kwargs['num_neighbors'])

            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=kwargs['batch_neg_src_node_ids'],
                                                                dst_node_ids=kwargs['batch_neg_dst_node_ids'],
                                                                node_interact_times=kwargs['batch_node_interact_times'],
                                                                num_neighbors=kwargs['num_neighbors'])
        elif model_name in ['JODIE', 'DyRep', 'TGN']:
            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=kwargs['batch_neg_src_node_ids'],
                                                                dst_node_ids=kwargs['batch_neg_dst_node_ids'],
                                                                node_interact_times=kwargs['batch_node_interact_times'],
                                                                edge_ids=kwargs.get('edge_ids', None),
                                                                edges_are_positive=kwargs.get('edges_are_positive', False),
                                                                num_neighbors=kwargs['num_neighbors'])

            batch_src_node_embeddings, batch_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=kwargs['batch_src_node_ids'],
                                                                dst_node_ids=kwargs['batch_dst_node_ids'],
                                                                node_interact_times=kwargs['batch_node_interact_times'],
                                                                edge_ids=kwargs['batch_edge_ids'],
                                                                edges_are_positive=kwargs.get('edges_are_positive', True),
                                                                num_neighbors=kwargs['num_neighbors'])
        elif model_name in ['GraphMixer']:
            batch_src_node_embeddings, batch_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=kwargs['batch_src_node_ids'],
                                                                dst_node_ids=kwargs['batch_dst_node_ids'],
                                                                node_interact_times=kwargs['batch_node_interact_times'],
                                                                num_neighbors=kwargs['num_neighbors'],
                                                                time_gap=kwargs['time_gap'])

            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=kwargs['batch_neg_src_node_ids'],
                                                                dst_node_ids=kwargs['batch_neg_dst_node_ids'],
                                                                node_interact_times=kwargs['batch_node_interact_times'],
                                                                num_neighbors=kwargs['num_neighbors'],
                                                                time_gap=kwargs['time_gap'])
        elif model_name in ['DyGFormer']:
            batch_src_node_embeddings, batch_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=kwargs['batch_src_node_ids'],
                                                                dst_node_ids=kwargs['batch_dst_node_ids'],
                                                                node_interact_times=kwargs['batch_node_interact_times'])

            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=kwargs['batch_neg_src_node_ids'],
                                                                dst_node_ids=kwargs['batch_neg_dst_node_ids'],
                                                                node_interact_times=kwargs['batch_node_interact_times'])
        if positive:
            return batch_src_node_embeddings, batch_dst_node_embeddings
        else:
            return batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings

    def forward(self, loss_func, train_neighbor_sampler, **kwargs):
        logits = []
        losses = []
        labels = []

        for model, model_name in zip(self.base_models, self.model_names):
            if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if model_name in ['JODIE', 'DyRep', 'TGN']:
                model[0].memory_bank.__init_memory_bank__()
            batch_src_node_embeddings, batch_dst_node_embeddings = self.compute_embeddings(model, model_name, kwargs, positive=True)
            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = self.compute_embeddings(model, model_name, kwargs, positive=False)

            pos_logits = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1)
            neg_logits = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1)

            logit = torch.cat([pos_logits, neg_logits], dim=0)
            logit = logit.to(kwargs['device'])
            logits.append(logit)

            if len(labels) == 0:
                labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
                labels = labels.to(kwargs['device'])
            losses.append(loss_func(logit, labels))

        if "EdgeBank" in self.model_names:
            positive_probabilities, negative_probabilities = edge_bank_link_prediction(history_data=kwargs['history_data'],
                                                                                    positive_edges=kwargs['positive_edges'],
                                                                                    negative_edges=kwargs['negative_edges'],
                                                                                    edge_bank_memory_mode=kwargs['edge_bank_memory_mode'],
                                                                                    time_window_mode=kwargs['time_window_mode'],
                                                                                    time_window_proportion=kwargs['test_ratio'])

            predicts = torch.from_numpy(np.concatenate([positive_probabilities, negative_probabilities])).float()
            predicts = predicts.to(kwargs['device'])
            logits.append(predicts)

            labels = torch.cat([torch.ones(len(positive_probabilities)), torch.zeros(len(negative_probabilities))], dim=0)
            labels = labels.to(kwargs['device'])
            losses.append(loss_func(input=predicts, target=labels))

        combined_logits = torch.stack(logits, dim=-1)
        output_logit = self.combiner(combined_logits, return_logits=True).squeeze(1)
        return output_logit, torch.tensor(losses), labels


    def train_step(self, loss_func, optimizer, train_neighbor_sampler, **kwargs):
        weight_ensemble_individual = 0.8

        optimizer.zero_grad()
        output, individual_losses, labels = self.forward(loss_func, train_neighbor_sampler, **kwargs)

        weights = torch.ones(len(individual_losses))/len(individual_losses)
        weighted_losses = sum(weights*individual_losses)
        ensemble_loss = loss_func(output, labels)
        loss = ensemble_loss*weight_ensemble_individual + weighted_losses*(1-weight_ensemble_individual)

        loss.backward()
        optimizer.step()

        for model, model_name in zip(self.base_models, self.model_names):
            if model_name in ['JODIE', 'DyRep', 'TGN']:
                model[0].memory_bank.detach_memory_bank()

        predictions = torch.sigmoid(output)

        return loss.item(), predictions, labels, individual_losses


    
    def eval_TGB(self, 
                 device, 
                 edgebank_data, 
                 neighbor_sampler, 
                 evaluate_idx_data_loader,
                 evaluate_data,  
                 negative_sampler: object, 
                 evaluator, metric: str = 'mrr',
                 split_mode: str = 'test', 
                 k_value: int = 10, 
                 num_neighbors: int = 20, 
                 time_gap: int = 2000,
                 subset: bool = False
                ):
        
        perf_list = []
        test_metrics = []
        weight_info = {}

        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            if subset and batch_idx > 3:
                break
            
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            pos_src_orig = batch_src_node_ids - 1
            pos_dst_orig = batch_dst_node_ids - 1
            pos_t = np.array([int(ts) for ts in batch_node_interact_times])
            neg_batch_list = negative_sampler.query_batch(pos_src_orig, pos_dst_orig, 
                                                    pos_t, split_mode=split_mode)
            
            # incorporate the testing data before the current batch to history_data, which is similar to memory-based models
            history_data = Data(src_node_ids=np.concatenate([edgebank_data.src_node_ids, evaluate_data.src_node_ids[: evaluate_data_indices[0]]]),
                                dst_node_ids=np.concatenate([edgebank_data.dst_node_ids, evaluate_data.dst_node_ids[: evaluate_data_indices[0]]]),
                                node_interact_times=np.concatenate([edgebank_data.node_interact_times, evaluate_data.node_interact_times[: evaluate_data_indices[0]]]),
                                edge_ids=np.concatenate([edgebank_data.edge_ids, evaluate_data.edge_ids[: evaluate_data_indices[0]]]),
                                labels=np.concatenate([edgebank_data.labels, evaluate_data.labels[: evaluate_data_indices[0]]]))
            for idx, neg_batch in enumerate(neg_batch_list):
                if subset and idx > 3:
                    break

                logits = []
                labels = []
                        
                for model, model_name in zip(self.base_models, self.model_names):
            
                    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
                        model[0].set_neighbor_sampler(neighbor_sampler)

                    model.eval()

                    with torch.no_grad():
                        
                        neg_batch = np.array(neg_batch) # why +1 ?
                        if model == self.base_models[0]:
                            neg_batch = neg_batch + 1
                        batch_neg_src_node_ids = np.array([int(batch_src_node_ids[idx]) for _ in range(len(neg_batch))])
                        batch_neg_dst_node_ids = np.array(neg_batch)
                        batch_neg_node_interact_times = np.array([batch_node_interact_times[idx] for _ in range(len(neg_batch))])

                        positive_edges = (batch_src_node_ids, batch_dst_node_ids)
                        negative_edges = (batch_neg_src_node_ids, batch_neg_dst_node_ids)

                        kwargs = {
                            "batch_src_node_ids": np.array([batch_src_node_ids[idx]]),
                            "batch_dst_node_ids": np.array([batch_dst_node_ids[idx]]),
                            "batch_node_interact_times": np.array([batch_node_interact_times[idx]]),
                            "batch_edge_ids": np.array([batch_edge_ids[idx]]),
                            "batch_neg_src_node_ids": batch_neg_src_node_ids,
                            "batch_neg_dst_node_ids": batch_neg_dst_node_ids,
                            "batch_neg_node_interact_times": batch_neg_node_interact_times,
                            "num_neighbors": num_neighbors,
                            "time_gap": time_gap,
                            'history_data':history_data,
                            'positive_edges':positive_edges,
                            'negative_edges':negative_edges,
                            'edge_bank_memory_mode':'time_window_memory',
                            'time_window_mode':'fixed_proportion',
                            'test_ratio':0.15}

                        # negative edges
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                            query_pred_edge_batch(model_name=model_name, model=model, 
                                src_node_ids=batch_neg_src_node_ids, dst_node_ids=batch_neg_dst_node_ids, 
                                node_interact_times=batch_neg_node_interact_times, edge_ids=None,
                                edges_are_positive=False, num_neighbors=num_neighbors, time_gap=time_gap)
                        
                        # one positive edge
                        batch_pos_src_node_embeddings, batch_pos_dst_node_embeddings = \
                            query_pred_edge_batch(model_name=model_name, model=model, 
                                src_node_ids=np.array([batch_src_node_ids[idx]]), dst_node_ids=np.array([batch_dst_node_ids[idx]]), 
                                node_interact_times=np.array([batch_node_interact_times[idx]]), edge_ids=np.array([batch_edge_ids[idx]]),
                                edges_are_positive=True, num_neighbors=num_neighbors, time_gap=time_gap)

                        pos_logits = model[1](input_1=batch_pos_src_node_embeddings, input_2=batch_pos_dst_node_embeddings).squeeze(dim=-1)
                        neg_logits = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1)

                        logit = torch.cat([pos_logits, neg_logits], dim=0)
                        logit = logit.to(device)
                        logits.append(logit)

                        if len(labels) == 0:
                            labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
                            labels = labels.to(device)
                        test_metrics.append(get_link_prediction_metrics(predicts=torch.cat([pos_logits.sigmoid(), neg_logits.sigmoid()], dim=0), 
                                                                        labels=labels))

                if "EdgeBank" in self.model_names:
                    positive_probabilities, negative_probabilities = edge_bank_link_prediction(history_data=kwargs['history_data'],
                                                                                        positive_edges=kwargs['positive_edges'],
                                                                                        negative_edges=kwargs['negative_edges'],
                                                                                        edge_bank_memory_mode=kwargs['edge_bank_memory_mode'],
                                                                                        time_window_mode=kwargs['time_window_mode'],
                                                                                        time_window_proportion=kwargs['test_ratio'])

                    predicts = torch.from_numpy(np.concatenate([positive_probabilities[:1], negative_probabilities])).float()
                    predicts = predicts.to(device)
                    logits.append(predicts)
                    test_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                combined_logits = torch.stack(logits, dim=-1)
                output_pred = self.combiner(combined_logits, return_logits=False).squeeze(1)
                
                input_dict = {
                    'y_pred_pos': output_pred[labels == 1].cpu().detach().numpy(),
                    'y_pred_neg': output_pred[labels == 0].cpu().detach().numpy(),
                    'eval_metric': [metric]
                }

                perf_list.append(evaluator.eval(input_dict)[metric])

                

        # extracting weight info of ensemble
        names, weights, bias = self.combiner.get_weights()
        weight_info['bias'] = bias
        for name in names:
            weight_info[name] = weights[names.index(name)]

        avg_perf_metric = float(np.mean(np.array(perf_list)))
        pr_auc = np.mean([test_metric['pr_auc'] for test_metric in test_metrics])
        roc_auc = np.mean([test_metric['roc_auc'] for test_metric in test_metrics])

        return avg_perf_metric, weight_info, pr_auc, roc_auc
                                

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.num_features = input_dim
        # number of combinations of 2 features
        self.num_combinations = sum(1 for _ in combinations(range(self.num_features), 2))

        self.linear = nn.Linear(self.num_features + self.num_combinations, output_dim)

    def forward(self, x, return_logits=False):
        interaction_terms = self.compute_interactions(x)
        # with interaction term
        x_with_interacts = torch.cat([x, interaction_terms], dim=1)
        outputs_logit = self.linear(x_with_interacts)

        if return_logits:
            return outputs_logit
        else:
            outputs_pred = torch.sigmoid(outputs_logit)
            return outputs_pred

    def compute_interactions(self, x):
        interactions = []
        # Interaction terms for every pair of logits (cols)
        for i, j in combinations(range(self.num_features), 2):
            interaction = x[:, i] * x[:, j]
            interactions.append(interaction.unsqueeze(1))
        # store in matrix
        interaction_matrix = torch.cat(interactions, dim=1)
        return interaction_matrix
    
    def get_weights(self):
        feature_names = [f'Feature_{i+1}' for i in range(self.num_features)]
        interaction_pairs = [f'Interaction_{i+1}_{j+1}' for i, j in combinations(range(self.num_features), 2)]
        all_names = feature_names + interaction_pairs

        weights = self.linear.weight.detach().cpu().numpy().flatten()
        bias = self.linear.bias.item()
        return all_names, weights, bias