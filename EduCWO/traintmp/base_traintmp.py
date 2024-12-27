
import torch
import logging
import importlib
from torch.utils.data import DataLoader

from utils import UnifyConfig
from utils.commonUtil import set_same_seeds

class BaseTrainTmp(object):
    def __init__(self, cfg: UnifyConfig):
        self.cfg: UnifyConfig = cfg
        self.datatmp_cfg: UnifyConfig = cfg.datatmp_cfg
        self.evaltmp_cfg: UnifyConfig = cfg.evaltmp_cfg
        self.traintmp_cfg: UnifyConfig = cfg.traintmp_cfg
        self.frame_cfg: UnifyConfig = cfg.frame_cfg
        self.model_cfg: UnifyConfig = cfg.model_cfg
        self.logger: logging.Logger = cfg.logger

        # 模板类实例化：数据模板实例化、模型模板实例化以及评估模板实例化
        self.datatmp_cls = importlib.import_module('EduCWO.datatmp').__getattribute__(self.datatmp_cfg['cls'])
        self.model_cls = importlib.import_module('EduCWO.model').__getattribute__(self.model_cfg['cls'])
        self.evaltmp_clses = [
            importlib.import_module('EduCWO.evaltmp').__getattribute__(tmp) for tmp in self.evaltmp_cfg['clses'] 
        ]

        set_same_seeds(self.traintmp_cfg['seed'])
        self.datatmp = self.datatmp_cls.from_cfg(self.cfg)
        self.model = self.model_cls.from_cfg(self.cfg, self.datatmp)
        self.evaltmps = [cls(self.cfg) for cls in self.evaltmp_clses]
        self._check_params()

    def _check_params(self):
        assert self.traintmp_cfg['best_epoch_metric'] in set(i[0] for i in self.traintmp_cfg['early_stop_metrics'])

    def _get_optim(self):
        optimizer = self.traintmp_cfg['optim']
        lr = self.traintmp_cfg['lr']
        weight_decay = self.traintmp_cfg['weight_decay']
        eps = self.traintmp_cfg['eps']
        if optimizer == "sgd":
            optim = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
        elif optimizer == "adam":
            optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
        elif optimizer == "adagrad":
            optim = torch.optim.Adagrad(self.model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
        elif optimizer == "rmsprop":
            optim = torch.optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
        else:
            raise ValueError("unsupported optimizer")
        return optim

    def start(self):
        self.logger.info(f"TrainTmp {self.__class__.__base__} Started!")
        self.build_loaders()  # 数据加载器构建

        extra_data = self.datatmp.get_extra_data()  # 加载数据模板的额外数据
        if len(extra_data):
            self.model.add_extra_data(**extra_data)  # 将额外数据加载到模型模板
        else:
            self.model.add_extra_data()
             
        self.model.build_cfg()  # 模型的参数配置设定
        self.model.build_model()  # 模型操作配置, 可自定义操作配置
        # self.model._init_params()  # 模型参数初始化配置, 支持自定义模型参数配置 | PS：此处模型初始化参数修改到self.model.build_model()中
        self.model.to(self.model.device)  # 指定模型运行基础环境: CPU or GPU
        
    def build_loaders(self):
        if hasattr(self.datatmp, 'build_dataloaders'):
            train_loader, val_loader, test_loader = self.datatmp.build_dataloaders()
        else:
            pass  # 支持自定义数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
    
    def batch_dict2device(self, epoch, batch_id, batch_dict):
        dic = {}
        dic['epoch'] = epoch
        dic['batch_id'] = batch_id  # 将batch_id加到batch数据中
        for k, v in batch_dict.items():
            if not isinstance(v, list):
                dic[k] = v.to(self.model.device)
            else:
                dic[k] = v
        return dic

        
        