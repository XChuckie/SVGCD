
import torch
import pandas as pd
import numpy as np

from utils.commonUtil import tensor2npy
from .base_evaltmp import BaseEvalTmp
from utils.callback import ModeState

class CognitiveDiagnosisEvalTmp(BaseEvalTmp):
    def __init__(self, cfg):
        super().__init__(cfg)
    
    def eval(self, stu_stats:np.ndarray, Q_mat:np.ndarray, **kwargs):
        metric_result = {}
        ignore_metrics = kwargs.get('ignore_bc_metrics', {})

        for metric_name in self.evaltmp_cfg['use_metrics']:
            if metric_name not in ignore_metrics:  # 剔除分类指标：AUC、ACC、RMSE等
                if metric_name == "doa_all":
                    metric_result[metric_name] = self._get_metrics(metric_name)(
                        self.stu_id_total,
                        self.exer_id_total,
                        self.label_id_total,
                        user_emb=stu_stats,
                        Q_mat=Q_mat
                    )
                elif metric_name == "doa_train_val":
                    metric_result[metric_name] = self._get_metrics(metric_name)(
                        self.stu_id_train_val,
                        self.exer_id_train_val,
                        self.label_id_train_val,
                        Q_mat=Q_mat
                    )
                elif metric_name == "doa_test":
                    metric_result[metric_name] = self._get_metrics(metric_name)(
                        self.test_loader.dataset.dict_data['stu_id'],
                        self.test_loader.dataset.dict_data['exer_id'],
                        self.test_loader.dataset.dict_data['label'],
                        user_emb=stu_stats,
                        Q_mat=Q_mat
                    )
                else:
                    raise NotImplementedError
        return metric_result

    def _get_metrics(self, metric):
        if metric == "doa_all":
            return self.doa_report
        elif metric == "doa_test":
            return self.doa_report
        elif metric == "doa_train_val":
            return self.doa_report
        else:
            raise NotImplementedError

    def set_dataloaders(self, train_loader, test_loader, val_loader=None):
        super().set_dataloaders(train_loader, test_loader, val_loader)

        if self.val_loader:
            self.stu_id_total = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_data['stu_id'],
                self.val_loader.dataset.dict_data['stu_id'],
                self.test_loader.dataset.dict_data['stu_id'],
            ]))
            self.exer_id_total = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_data['exer_id'],
                self.val_loader.dataset.dict_data['exer_id'],
                self.test_loader.dataset.dict_data['exer_id'],
            ]))

            self.label_id_total = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_data['label'],
                self.val_loader.dataset.dict_data['label'],
                self.test_loader.dataset.dict_data['label'],
            ]))

            self.stu_id_train_val = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_data['stu_id'],
                self.val_loader.dataset.dict_data['stu_id'],
            ]))
            self.exer_id_train_val = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_data['exer_id'],
                self.val_loader.dataset.dict_data['exer_id'],
            ]))

            self.label_id_train_val = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_data['label'],
                self.val_loader.dataset.dict_data['label'],
            ]))
        else:
            self.stu_id_total = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_data['stu_id'],
                self.test_loader.dataset.dict_data['stu_id'],
            ]))
            self.exer_id_total = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_data['exer_id'],
                self.test_loader.dataset.dict_data['exer_id'],
            ]))

            self.label_id_total = tensor2npy(torch.cat([
                self.train_loader.dataset.dict_data['label'],
                self.test_loader.dataset.dict_data['label'],
            ]))

            self.stu_id_train_val = tensor2npy(self.train_loader.dataset.dict_data['stu_id'])
            self.exer_id_train_val = tensor2npy(self.train_loader.dataset.dict_data['exer_id'])
            self.label_id_train_val = tensor2npy(self.train_loader.dataset.dict_data['label'])

    def doa_report(self, stu_id, exer_id, label, user_emb, Q_mat):
        """拆分数据：(用户编号、习题编号、知识点、label、知识点掌握程度)
        """
        knowledges = []
        knowledge_item = []
        knowledge_user = []
        knowledge_truth = []
        knowledge_theta = []
        for user, item, score in zip(stu_id, exer_id, label):
            theta = user_emb[user]
            knowledge = Q_mat[item]
            if isinstance(theta, list) or isinstance(theta, np.ndarray):
                for i, (theta_i, knowledge_i) in enumerate(zip(theta, knowledge)):
                    if knowledge_i == 1:  # 仅考虑统计习题关联的知识点
                        knowledges.append(i) # Cpt ID：知识点编号
                        knowledge_item.append(item) # Item ID: 用户交互的习题编号
                        knowledge_user.append(user) # User ID：学生编号
                        knowledge_truth.append(score) # score：用户对习题的真实标签
                        knowledge_theta.append(theta_i) # matser：用户对知识点i的掌握程度
            else:  
                """其他情况: 学生能力dim和知识点长度不匹配等
                """
                raise NotImplementedError
        knowledge_df = pd.DataFrame({
            "knowledge": knowledges,
            "user_id": knowledge_user,
            "item_id": knowledge_item,
            "score": knowledge_truth,
            "theta": knowledge_theta
        })
        knowledge_ground_truth = []
        knowledge_prediction = []
        for _, group_df in knowledge_df.groupby("knowledge"):
            _knowledge_ground_truth = []
            _knowledge_prediction = []
            for _, item_group_df in group_df.groupby("item_id"):
                _knowledge_ground_truth.append(item_group_df["score"].values)
                _knowledge_prediction.append(item_group_df["theta"].values)
            knowledge_ground_truth.append(_knowledge_ground_truth)
            knowledge_prediction.append(_knowledge_prediction)

        return self.doa_eval(knowledge_ground_truth, knowledge_prediction)


    def doa_eval(self, y_true, y_pred):
        doa = []
        doa_support = 0
        z_support = 0
        for knowledge_label, knowledge_pred in zip(y_true, y_pred):
            _doa = 0
            _z = 0
            for label, pred in zip(knowledge_label, knowledge_pred):
                if sum(label) == len(label) or sum(label) == 0:  # 排除情况：用户交互习题且结果相同
                    continue
                pos_idx = []
                neg_idx = []
                for i, _label in enumerate(label): # 找出所有(1, 0) pair
                    if _label == 1:
                        pos_idx.append(i)
                    else:
                        neg_idx.append(i)
                pos_pred = pred[pos_idx]
                neg_pred = pred[neg_idx]
                invalid = 0
                for _pos_pred in pos_pred:
                    _doa += len(neg_pred[neg_pred < _pos_pred])
                    invalid += len(neg_pred[neg_pred == _pos_pred])  # 排除非有效数量
                _z += (len(pos_pred) * len(neg_pred)) - invalid
            if _z > 0:
                doa.append(_doa / _z)
                z_support += _z # 有效pair个数
                doa_support += 1 # 有效doa
        # return {
        #     "doa": np.mean(doa),
        #     "doa_know_support": doa_support,
        #     "doa_z_support": z_support,
        #     "doa_list": doa,
        # }
        return float(np.mean(doa))

