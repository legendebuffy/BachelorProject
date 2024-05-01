import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class Ensemble(nn.Module):

    def __init__(self, base_models, combiner, model_names):
        super(Ensemble, self).__init__()
        self.base_models = nn.ModuleList(base_models)
        self.combiner = combiner
        self.model_names = model_names


    def compute_embeddings(self, model, model_name, kwargs, positive):
        if model_name in ['TGAT', 'CAWN', 'TCL']:
            src_node_ids = kwargs["batch_src_node_ids"] if positive else kwargs["batch_neg_src_node_ids"]
            dst_node_ids = kwargs["batch_dst_node_ids"] if positive else kwargs["batch_neg_dst_node_ids"]
            node_interact_times = kwargs["batch_node_interact_times"] if positive else kwargs["batch_neg_node_interact_times"]
            num_neighbors = kwargs["num_neighbors"]
            return model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                                    dst_node_ids=dst_node_ids,
                                                                    node_interact_times=node_interact_times,
                                                                    num_neighbors=num_neighbors)
        elif model_name in ['JODIE', 'DyRep', 'TGN']:
            src_node_ids = kwargs["batch_src_node_ids"] if positive else kwargs["batch_neg_src_node_ids"]
            dst_node_ids = kwargs["batch_dst_node_ids"] if positive else kwargs["batch_neg_dst_node_ids"]
            node_interact_times = kwargs["batch_node_interact_times"] if positive else kwargs["batch_neg_node_interact_times"]
            edge_ids = None if not positive else kwargs["batch_edge_ids"]
            edges_are_positive = positive
            num_neighbors = kwargs["num_neighbors"]
            return model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                                    dst_node_ids=dst_node_ids,
                                                                    node_interact_times=node_interact_times,
                                                                    edge_ids=edge_ids,
                                                                    edges_are_positive=edges_are_positive,
                                                                    num_neighbors=num_neighbors)
        elif model_name in ['GraphMixer']:
            src_node_ids = kwargs["batch_src_node_ids"] if positive else kwargs["batch_neg_src_node_ids"]
            dst_node_ids = kwargs["batch_dst_node_ids"] if positive else kwargs["batch_neg_dst_node_ids"]
            node_interact_times = kwargs["batch_node_interact_times"] if positive else kwargs["batch_neg_node_interact_times"]
            num_neighbors = kwargs["num_neighbors"]
            time_gap = kwargs["time_gap"]
            return model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                                    dst_node_ids=dst_node_ids,
                                                                    node_interact_times=node_interact_times,
                                                                    num_neighbors=num_neighbors,
                                                                    time_gap=time_gap)
        elif model_name in ['DyGFormer']:
            src_node_ids = kwargs["batch_src_node_ids"] if positive else kwargs["batch_neg_src_node_ids"]
            dst_node_ids = kwargs["batch_dst_node_ids"] if positive else kwargs["batch_neg_dst_node_ids"]
            node_interact_times = kwargs["batch_node_interact_times"] if positive else kwargs["batch_neg_node_interact_times"]
            return model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                                    dst_node_ids=dst_node_ids,
                                                                    node_interact_times=node_interact_times)
        else:
            raise ValueError(f"Wrong value for model_name {model_name}!")


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
            logits.append(logit)

            if len(labels) == 0:
                labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
            
            losses.append(loss_func(logit, labels))

        combined_logits = torch.stack(logits, dim=-1)
        output_logit = self.combiner(combined_logits, return_logits=True).squeeze(1)
        return output_logit, torch.tensor(losses), labels


    def train_step(self, loss_func, optimizer, train_neighbor_sampler, **kwargs):
        optimizer.zero_grad()
        output, losses, labels = self.forward(loss_func, train_neighbor_sampler, **kwargs)

        weights = torch.ones(len(losses))/len(losses)
        weighted_losses = sum(weights*losses)
        ensemble_loss = loss_func(output, labels)
        loss = ensemble_loss*0.5 + weighted_losses*0.5

        #loss.requires_grad = True # OBS: wth is this??
        loss.backward()
        optimizer.step()

        for model, model_name in zip(self.base_models, self.model_names):
            if model_name in ['JODIE', 'DyRep', 'TGN']:
                model[0].memory_bank.detach_memory_bank()

        predictions = torch.sigmoid(output)

        return loss.item(), predictions, labels


    def eval_TGB(self, neighbor_sampler, evaluate_idx_data_loader,
                evaluate_data,  negative_sampler: object, evaluator, metric: str = 'mrr',
                split_mode: str = 'test', k_value: int = 10, num_neighbors: int = 20, time_gap: int = 2000,
                subset: str = 'False'):
        
        perf_list = []

        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            if subset == 'True' and batch_idx > 3:
                break
            
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            pos_src_orig = batch_src_node_ids - 1
            pos_dst_orig = batch_dst_node_ids - 1
            pos_t = np.array([int(ts) for ts in batch_node_interact_times])
            neg_batch_list = negative_sampler.query_batch(pos_src_orig, pos_dst_orig, 
                                                    pos_t, split_mode=split_mode)

            for idx, neg_batch in enumerate(neg_batch_list):
                if subset == 'True' and idx > 3:
                    break

                logits = []
                labels = []
                        
                for model, model_name in zip(self.base_models, self.model_names):
            
                    if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
                        model[0].set_neighbor_sampler(neighbor_sampler)

                    model.eval()

                    with torch.no_grad():
                        neg_batch = np.array(neg_batch) + 1
                        batch_neg_src_node_ids = np.array([int(batch_src_node_ids[idx]) for _ in range(len(neg_batch))])
                        batch_neg_dst_node_ids = np.array(neg_batch)
                        batch_neg_node_interact_times = np.array([batch_node_interact_times[idx] for _ in range(len(neg_batch))])

                        kwargs = {
                            "batch_src_node_ids": np.array([batch_src_node_ids[idx]]),
                            "batch_dst_node_ids": np.array([batch_dst_node_ids[idx]]),
                            "batch_node_interact_times": np.array([batch_node_interact_times[idx]]),
                            "batch_edge_ids": np.array([batch_edge_ids[idx]]),
                            "batch_neg_src_node_ids": batch_neg_src_node_ids,
                            "batch_neg_dst_node_ids": batch_neg_dst_node_ids,
                            "batch_neg_node_interact_times": batch_neg_node_interact_times,
                            "num_neighbors": num_neighbors,
                            "time_gap": time_gap
                        }

                        batch_pos_src_node_embeddings, batch_pos_dst_node_embeddings = self.compute_embeddings(model, model_name, kwargs, positive=True)
                        batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = self.compute_embeddings(model, model_name, kwargs, positive=False)

                        pos_logits = model[1](input_1=batch_pos_src_node_embeddings, input_2=batch_pos_dst_node_embeddings).squeeze(dim=-1)
                        neg_logits = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1)

                        logit = torch.cat([pos_logits, neg_logits], dim=0)
                        logits.append(logit)

                        if len(labels) == 0:
                            labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
        
                combined_logits = torch.stack(logits, dim=-1)
                output_pred = self.combiner(combined_logits, return_logits=False).squeeze(1)

                input_dict = {
                    'y_pred_pos': output_pred[labels == 1].cpu().detach().numpy(),
                    'y_pred_neg': output_pred[labels == 0].cpu().detach().numpy(),
                    'eval_metric': [metric]
                }

                perf_list.append(evaluator.eval(input_dict)[metric])

        avg_perf_metric = float(np.mean(np.array(perf_list)))

        return avg_perf_metric
                                

class LogisticRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, return_logits=False):
        outputs_logit = self.linear(x)
        if return_logits:
            return outputs_logit
        else:
            outputs_pred = torch.sigmoid(outputs_logit)
            return outputs_pred
