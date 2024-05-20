import argparse
import sys
import torch


def get_link_prediction_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('DyGLib: Interface for the link prediction task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='tgbl-wiki',
                        choices=['tgbl-flight', 'tgbl-wiki', 'tgbl-review', 'tgbl-coin', 'tgbl-comment'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--model_name', type=str, default='TGN_GraphMixer', help='name of the model, note that EdgeBank is only applicable for evaluation')
                        #choices=['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'EdgeBank', 'TCL', 'GraphMixer', 'DyGFormer'])
    # parser.add_argument('--gpu', type=int, default=0, help='ID of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node') 
    parser.add_argument('--sample_neighbor_strategy', default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers') 
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--edge_bank_memory_mode', type=str, default='unlimited_memory', help='how memory of EdgeBank works',
                        choices=['unlimited_memory', 'time_window_memory', 'repeat_threshold_memory'])
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=25, help='number of epochs')  # original value = 100, --> TGB:50
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=2, help='patience for early stopping')  # original value = 20
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=1, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--negative_sample_strategy', type=str, default='random', choices=['random', 'historical', 'inductive'],
                        help='strategy for the negative edge sampling')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--time_scale', type=str, default=None, help='Time Scale for discretization',
                        choices=['minutely', 'hourly', 'daily', 'weekly', 'monthly', 'yearly'])
    parser.add_argument('--train_time_gran', type=str, default='ct', choices=['ct', 'dt'],
                        help='Training time granularity.')
    parser.add_argument('--eval_time_gran', type=str, default='ct', choices=['ct', 'dt'],
                        help='Evaluation time granularity.')
    
    # DTU: To save logits for simple ensemble
    parser.add_argument('--logits', type=str, default="False", help='to save logits')
    parser.add_argument('--subset', type=str, default="False", help='to save subset of data')

    try:
        args = parser.parse_args()
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    if args.model_name == 'EdgeBank':
        assert is_evaluation, 'EdgeBank is only applicable for evaluation!'

    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)

    return args


def load_link_prediction_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the link prediction task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name == 'TGAT':
        args.num_neighbors = 20
        args.num_layers = 2
        args.dropout = 0.1
        args.sample_neighbor_strategy = 'recent'
    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
        args.num_neighbors = 10
        args.num_layers = 1
        if args.model_name == 'JODIE':
            args.dropout = 0.1
        elif args.model_name == 'DyRep':
            args.dropout = 0.1
        else:
            assert args.model_name == 'TGN'
            args.dropout = 0.1
        if args.model_name in ['TGN', 'DyRep']:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'CAWN':
        args.time_scaling_factor = 1e-6
        args.num_neighbors = 32
        args.dropout = 0.1
        args.sample_neighbor_strategy = 'time_interval_aware'
    elif args.model_name == 'EdgeBank':
        if args.negative_sample_strategy == 'random':
            if args.dataset_name in ['wikipedia', 'reddit', 'uci', 'Flights']:
                args.edge_bank_memory_mode = 'unlimited_memory'
            elif args.dataset_name in ['mooc', 'lastfm', 'enron', 'CanParl', 'USLegis']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'fixed_proportion'
            elif args.dataset_name in ['UNtrade', 'UNvote', 'Contacts']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'repeat_interval'
            else:
                assert args.dataset_name == 'SocialEvo'
                args.edge_bank_memory_mode = 'repeat_threshold_memory'
        elif args.negative_sample_strategy == 'historical':
            if args.dataset_name in ['uci', 'CanParl', 'USLegis']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'fixed_proportion'
            elif args.dataset_name in ['mooc', 'lastfm', 'enron', 'UNtrade', 'UNvote', 'Contacts']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'repeat_interval'
            else:
                assert args.dataset_name in ['wikipedia', 'reddit', 'SocialEvo', 'Flights']
                args.edge_bank_memory_mode = 'repeat_threshold_memory'
        else:
            assert args.negative_sample_strategy == 'inductive'
            if args.dataset_name in ['USLegis']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'fixed_proportion'
            elif args.dataset_name in ['uci', 'UNvote']:
                args.edge_bank_memory_mode = 'time_window_memory'
                args.time_window_mode = 'repeat_interval'
            else:
                assert args.dataset_name in ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron',
                                             'SocialEvo', 'Flights', 'CanParl', 'UNtrade', 'Contacts']
                args.edge_bank_memory_mode = 'repeat_threshold_memory'
    elif args.model_name == 'TCL':
        args.num_neighbors = 20
        args.num_layers = 2
        args.dropout = 0.1
        args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'GraphMixer':
        args.num_layers = 2
        args.num_neighbors = 20
        args.dropout = 0.1
        args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'DyGFormer':
        args.num_layers = 2
        args.max_input_sequence_length = 32
        args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        args.dropout = 0.1
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")


def get_node_classification_args():
    """
    get the args for the node classification task
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser('Interface for the node classification task')
    parser.add_argument('--dataset_name', type=str, help='dataset to be used', default='wikipedia', choices=['wikipedia', 'reddit'])
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--model_name', type=str, default='DyGFormer', help='name of the model',
                        choices=['JODIE', 'DyRep', 'TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer'])
    parser.add_argument('--gpu', type=int, default=0, help='number of gpu to use')
    parser.add_argument('--num_neighbors', type=int, default=20, help='number of neighbors to sample for each node')
    parser.add_argument('--sample_neighbor_strategy', default='recent',
                        choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
    parser.add_argument('--time_scaling_factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
    parser.add_argument('--num_walk_heads', type=int, default=8, help='number of heads used for the attention in walk encoder')
    parser.add_argument('--num_heads', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--num_layers', type=int, default=2, help='number of model layers')
    parser.add_argument('--walk_length', type=int, default=1, help='length of each random walk')
    parser.add_argument('--time_gap', type=int, default=2000, help='time gap for neighbors to compute node features')
    parser.add_argument('--time_feat_dim', type=int, default=100, help='dimension of the time embedding')
    parser.add_argument('--position_feat_dim', type=int, default=172, help='dimension of the position embedding')
    parser.add_argument('--edge_bank_memory_mode', type=str, default='unlimited_memory', help='how memory of EdgeBank works',
                        choices=['unlimited_memory', 'time_window_memory', 'repeat_threshold_memory'])
    parser.add_argument('--time_window_mode', type=str, default='fixed_proportion', help='how to select the time window size for time window memory',
                        choices=['fixed_proportion', 'repeat_interval'])
    parser.add_argument('--patch_size', type=int, default=1, help='patch size')
    parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
    parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam', 'RMSprop'], help='name of optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--patience', type=int, default=20, help='patience for early stopping')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='ratio of validation set')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='ratio of test set')
    parser.add_argument('--num_runs', type=int, default=5, help='number of runs')
    parser.add_argument('--test_interval_epochs', type=int, default=10, help='how many epochs to perform testing once')
    parser.add_argument('--load_best_configs', action='store_true', default=False, help='whether to load the best configurations')

    try:
        args = parser.parse_args()
        args.device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    except:
        parser.print_help()
        sys.exit()

    assert args.dataset_name in ['wikipedia', 'reddit'], f'Wrong value for dataset_name {args.dataset_name}!'
    if args.load_best_configs:
        load_node_classification_best_configs(args=args)

    return args


def load_node_classification_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the node classification task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name == 'TGAT':
        args.num_neighbors = 20
        args.num_layers = 2
        args.dropout = 0.1
        if args.dataset_name in ['reddit']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name in ['JODIE', 'DyRep', 'TGN']:
        args.num_neighbors = 10
        args.num_layers = 1
        args.dropout = 0.1
        args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'CAWN':
        args.time_scaling_factor = 1e-6
        args.num_neighbors = 32
        args.dropout = 0.1
        args.sample_neighbor_strategy = 'time_interval_aware'
    elif args.model_name == 'TCL':
        args.num_neighbors = 20
        args.num_layers = 2
        args.dropout = 0.1
        if args.dataset_name in ['reddit']:
            args.sample_neighbor_strategy = 'uniform'
        else:
            args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'GraphMixer':
        args.num_layers = 2
        if args.dataset_name in ['reddit']:
            args.num_neighbors = 10
        else:
            args.num_neighbors = 30
        args.dropout = 0.5
        args.sample_neighbor_strategy = 'recent'
    elif args.model_name == 'DyGFormer':
        args.num_layers = 2
        if args.dataset_name in ['reddit']:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ['reddit']:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")
