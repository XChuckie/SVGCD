import logging

from utils import UnifyConfig
from utils.callback import CallbackList

class BaseEvalTmp(object):
    def __init__(self, cfg):
        self.cfg: UnifyConfig = cfg
        self.datatmp_cfg: UnifyConfig = cfg.datatmp_cfg
        self.evaltmp_cfg: UnifyConfig = cfg.evaltmp_cfg
        self.traintmp_cfg: UnifyConfig = cfg.traintmp_cfg
        self.frame_cfg: UnifyConfig = cfg.frame_cfg
        self.model_cfg: UnifyConfig = cfg.model_cfg
        self.logger: logging.Logger = cfg.logger
    
    def eval(self, **kwargs):
        pass
    
    def set_callback_list(self, callbacklist: CallbackList):
        self.callback_list = callbacklist
    
    def set_dataloaders(self, train_loader, test_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
    
    