import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def get_link_prediction_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the link prediction task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    # average_precision = average_precision_score(y_true=labels, y_score=predicts)
    precision, recall, _ = precision_recall_curve(y_true=labels, probas_pred=predicts)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'pr_auc': pr_auc, 'roc_auc': roc_auc}


def get_node_classification_metrics(predicts: torch.Tensor, labels: torch.Tensor):
    """
    get metrics for the node classification task
    :param predicts: Tensor, shape (num_samples, )
    :param labels: Tensor, shape (num_samples, )
    :return:
        dictionary of metrics {'metric_name_1': metric_1, ...}
    """
    predicts = predicts.cpu().detach().numpy()
    labels = labels.cpu().numpy()

    roc_auc = roc_auc_score(y_true=labels, y_score=predicts)

    return {'roc_auc': roc_auc}
