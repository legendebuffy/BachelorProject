import torch
import torch.nn as nn

# class Ensemble(nn.Module):
#     def __init__(self, base_models, combiner):
#         super(Ensemble, self).__init__()
#         self.base_models = nn.ModuleList(base_models)
#         self.combiner = combiner

#     def forward(self, x):
#         logits = [model(x) for model in self.base_models]
#         combined_logits = torch.cat(logits, dim=-1)
#         output = self.combiner(combined_logits)
#         return output

#     def train_step(self, x, y, loss_func, optimizer):
#         optimizer.zero_grad()
#         output = self.forward(x)
#         loss = loss_func(output, y)
#         loss.backward()
#         optimizer.step()
#         return loss.item()

# # define your base models and combiner
# base_models = [model1, model2, model3]  # replace with your actual models
# combiner = LogisticRegression()  # replace with your actual combiner

# # create the ensemble
# ensemble = Ensemble(base_models, combiner)

# # define your loss function and optimizer
# loss_func = nn.BCEWithLogitsLoss()
# optimizer = torch.optim.Adam(ensemble.parameters())

# # train the ensemble
# for epoch in range(num_epochs):
#     for batch in data_loader:
#         x, y = batch
#         loss = ensemble.train_step(x, y, loss_func, optimizer)
#         print(f'Epoch: {epoch}, Loss: {loss}')



# class Ensemble(nn.Module):
#     def __init__(self, base_models, combiner, model_names):
#         super(Ensemble, self).__init__()
#         self.base_models = nn.ModuleList(base_models)
#         self.combiner = combiner
#         self.model_names = model_names

#     def forward(self, x, train_neighbor_sampler, **kwargs):
#         logits = []
#         for model, model_name in zip(self.base_models, self.model_names):

#             if model_name in ['DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer']:
#                 model[0].set_neighbor_sampler(train_neighbor_sampler)
            
#             if model_name in ['JODIE', 'DyRep', 'TGN']:
#                 model[0].memory_bank.__init_memory_bank__()

#             if model_name in ['TGAT', 'CAWN', 'TCL']:
#                 batch_src_node_embeddings, batch_dst_node_embeddings = \
#                         model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=kwargs["batch_src_node_ids"],
#                                                                           dst_node_ids=kwargs["batch_dst_node_ids"],
#                                                                             node_interact_times=kwargs["batch_node_interact_times"],
#                                                                             num_neighbors=kwargs["num_neighbors"])
                
#                 batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
#                         model[0].compute_src_dst_node_temporal_embeddings(src_node_ids= kwargs["batch_neg_src_node_ids"],
#                                                                           dst_node_ids=kwargs["batch_neg_dst_node_ids"],
#                                                                             node_interact_times=kwargs["batch_neg_node_interact_times"],
#                                                                             num_neighbors=kwargs["num_neighbors"])
#             elif model_name in ['JODIE', 'DyRep', 'TGN']:
#                 batch_src_node_embeddings, batch_dst_node_embeddings = \
#                         model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=kwargs["batch_src_node_ids"],
#                                                                           dst_node_ids=kwargs["batch_dst_node_ids"],
#                                                                             node_interact_times=kwargs["batch_node_interact_times"],
#                                                                             edge_ids=None,
#                                                                             edges_are_positive=False,
#                                                                             num_neighbors=kwargs["num_neighbors"])
                
#                 batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
#                         model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=kwargs["batch_src_node_ids"],
#                                                                           dst_node_ids=kwargs["batch_dst_node_ids"],
#                                                                             node_interact_times=kwargs["batch_node_interact_times"],
#                                                                             edge_ids=kwards["batch_edge_ids"],
#                                                                             edges_are_positive=True,
#                                                                             num_neighbors=kwargs["num_neighbors"])
#             elif model_name in ['GraphMixer']:
#                 batch_src_node_embeddings, batch_dst_node_embeddings = \
#                         model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=kwargs["batch_src_node_ids"],
#                                                                           dst_node_ids=kwargs["batch_dst_node_ids"],
#                                                                             node_interact_times=kwargs["batch_node_interact_times"],
#                                                                             num_neighbors=kwargs["num_neighbors"],
#                                                                           time_gap= kwargs["time_gap"])
                    
#                 batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
#                         model[0].compute_src_dst_node_temporal_embeddings(src_node_ids= kwargs["batch_neg_src_node_ids"],
#                                                                           dst_node_ids=kwargs["batch_neg_dst_node_ids"],
#                                                                             node_interact_times=kwargs["batch_neg_node_interact_times"],
#                                                                             num_neighbors=kwargs["num_neighbors"],
#                                                                           time_gap= kwargs["time_gap"])
#             elif model_name in ['DyGformer']:
#                 batch_src_node_embeddings, batch_dst_node_embeddings = \
#                         model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=kwargs["batch_src_node_ids"],
#                                                                           dst_node_ids=kwargs["batch_dst_node_ids"],
#                                                                             node_interact_times=kwargs["batch_node_interact_times"])
                
#                 batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
#                         model[0].compute_src_dst_node_temporal_embeddings(src_node_ids= kwargs["batch_neg_src_node_ids"],
#                                                                           dst_node_ids=kwargs["batch_neg_dst_node_ids"],
#                                                                             node_interact_times=kwargs["batch_neg_node_interact_times"])
#             else:
#                 raise ValueError(f"Wrong value for model_name {model_name}!")
            
#             pos_logits = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1)
#             neg_logits = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1)

#             logits.append(torch.cat([pos_logits, neg_logits], dim=0))

#         combined_logits = torch.cat(logits, dim=-1)
#         output = self.combiner(combined_logits)
#         return output

#     def train_step(self, x, y, loss_func, optimizer, **kwargs):
#         optimizer.zero_grad()
#         output = self.forward(x, **kwargs)
#         loss = loss_func(output, y)
#         loss.backward()
#         optimizer.step()
#         return loss.item()



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
            node_interact_times = kwargs["batch_node_interact_times"]
            num_neighbors = kwargs["num_neighbors"]
            return model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                                    dst_node_ids=dst_node_ids,
                                                                    node_interact_times=node_interact_times,
                                                                    num_neighbors=num_neighbors)
        elif model_name in ['JODIE', 'DyRep', 'TGN']:
            src_node_ids = kwargs["batch_src_node_ids"] if positive else kwargs["batch_neg_src_node_ids"]
            dst_node_ids = kwargs["batch_dst_node_ids"] if positive else kwargs["batch_neg_dst_node_ids"]
            node_interact_times = kwargs["batch_node_interact_times"]
            edge_ids = None if positive else kwargs["batch_edge_ids"]
            edges_are_positive = not positive
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
            node_interact_times = kwargs["batch_node_interact_times"]
            num_neighbors = kwargs["num_neighbors"]
            time_gap = kwargs["time_gap"]
            return model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=src_node_ids,
                                                                    dst_node_ids=dst_node_ids,
                                                                    node_interact_times=node_interact_times,
                                                                    num_neighbors=num_neighbors,
                                                                    time_gap=time_gap)
        elif model_name in ['DyGformer']:
            src_node_ids = kwargs["batch_src_node_ids"] if positive else kwargs["batch_neg_src_node_ids"]
            dst_node_ids = kwargs["batch_dst_node_ids"] if positive else kwargs["batch_neg_dst_node_ids"]
            node_interact_times = kwargs["batch_node_interact_times"]
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
        output = self.combiner(combined_logits).squeeze(1)
        return output, torch.tensor(losses), labels

    def train_step(self, loss_func, optimizer, train_neighbor_sampler, **kwargs):
        optimizer.zero_grad()
        output, losses, labels = self.forward(loss_func, train_neighbor_sampler, **kwargs)

        weights = torch.ones(len(losses))/len(losses)
        weighted_losses = sum(weights*losses)
        ensemble_loss = loss_func(output, labels)
        loss = ensemble_loss*0.5 + weighted_losses*0.5

        loss.backward()
        optimizer.step()

        for model, model_name in zip(self.base_models, self.model_names):
            if model_name in ['JODIE', 'DyRep', 'TGN']:
                model[0].memory_bank.detach_memory_bank()

        return loss.item()


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

# if __name__ == '__main__':
#     # Assuming you have your models, combiner, and model names defined
#     base_models = [model1, model2, model3]  # Replace with your actual models
#     combiner = LogisticRegression()  # Replace with your actual combiner
#     model_names = ['TGAT', 'DyRep', 'GraphMixer']  # Replace with your actual model names

#     # Create an instance of your Ensemble class
#     ensemble = Ensemble(base_models, combiner, model_names)

#     # Define your loss function and optimizer
#     loss_func = nn.BCEWithLogitsLoss()  # Replace with your actual loss function
#     optimizer = torch.optim.Adam(ensemble.parameters())  # Replace with your actual optimizer

#     # Assuming you have your training data loader
#     for epoch in range(num_epochs):
#         for i, (x, y) in enumerate(train_loader):
#             kwargs = {}  # Fill this with the necessary keyword arguments for your forward method
#             loss = ensemble.train_step(loss_func, optimizer, train_neighbor_sampler,  **kwargs)
#             print(f'Epoch {epoch}, Batch {i}, Loss {loss}')