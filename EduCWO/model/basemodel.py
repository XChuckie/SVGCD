import logging
import torch.nn as nn
from abc import abstractmethod

from utils import UnifyConfig
from .utils import xavier_uniform_initialization, xavier_normal_initialization, kaiming_normal_initialization, kaiming_uniform_initialization

class BaseModel(nn.Module):
    def __init__(self, cfg, datatmp) -> None:
        super().__init__()
        self.cfg: UnifyConfig = cfg
        self.datatmp_cfg: UnifyConfig = cfg.datatmp_cfg
        self.evaltmp_cfg: UnifyConfig = cfg.evaltmp_cfg
        self.traintmp_cfg: UnifyConfig = cfg.traintmp_cfg
        self.frame_cfg: UnifyConfig = cfg.frame_cfg
        self.model_cfg: UnifyConfig = cfg.model_cfg
        self.logger: logging.Logger = cfg.logger
        # define extra global info
        self.device = self.traintmp_cfg['device']
        self.share_callback_dict = {
            "stop_training": False
        }
        # 增加数据模板调用接口
        self.dtp = datatmp
    
    def _init_params(self):
        if self.model_cfg['param_init_type'] == 'default':
            pass
        elif self.model_cfg['param_init_type'] == 'xavier_normal':
            self.apply(xavier_normal_initialization)
        elif self.model_cfg['param_init_type'] == 'xavier_uniform':
            self.apply(xavier_uniform_initialization)
        elif self.model_cfg['param_init_type'] == 'kaiming_normal':
            self.apply(kaiming_normal_initialization)
        elif self.model_cfg['param_init_type'] == 'kaiming_uniform':
            self.apply(kaiming_uniform_initialization)
        elif self.model_cfg['param_init_type'] == 'init_from_pretrained':
            self._load_params_from_pretrained()
        elif self.model_cfg['param_init_type'] == 'other':
            pass  # 支持自定义参数初始化操作
    
    def add_extra_data(self, **kwargs):
        pass
    
    @classmethod
    def from_cfg(cls, cfg, datatmp):
        """模型参数初始化接口
        """
        
        return cls(cfg, datatmp)

    @abstractmethod
    def build_cfg(self):
        """模型参数配置
        """
        pass

    @abstractmethod
    def build_model(self):
        """模型组件配置
        """
        pass
    
    def predict(self, **kwargs):
        pass

    def get_loss_dict(self, **kwargs):
        pass
    