import numpy as np

from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score, f1_score, label_ranking_loss, coverage_error
from .base_evaltmp import BaseEvalTmp
from utils.commonUtil import tensor2npy

class BinaryClassificationEvalTmp(BaseEvalTmp):
    def __init__(self, cfg):
        super().__init__(cfg)

    def eval(self, y_pd, y_gt, **kwargs):
        if not isinstance(y_pd, np.ndarray): y_pd = tensor2npy(y_pd)
        if not isinstance(y_gt, np.ndarray): y_gt = tensor2npy(y_gt)
        metric_result = {}
        ignore_metrics = kwargs.get('ignore_cd_metrics', {})
        for metric_name in self.evaltmp_cfg['use_metrics']:
            if metric_name not in ignore_metrics:
                metric_result[metric_name] = self._get_metrics(metric_name)(y_gt, y_pd)
        return metric_result

    def _get_metrics(self, metric):
        if metric == "auc":
            return roc_auc_score
        elif metric == "mse":
            return mean_squared_error
        elif metric == 'rmse':
            return lambda y_gt, y_pd: mean_squared_error(y_gt, y_pd) ** 0.5
        elif metric == "acc":
            return lambda y_gt, y_pd: accuracy_score(y_gt, np.where(y_pd >= 0.5, 1, 0))
        else:
            raise NotImplementedError