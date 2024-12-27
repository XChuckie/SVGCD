from . import settings
from typing import Dict, Any
from utils import UnifyConfig


def get_global_cfg(
    dataset:str,
    traintmp_cfg_dict: Dict[str, Any],
    datatmp_cfg_dict:  Dict[str, Any],
    model_cfg_dict: Dict[str, Any],
    evaltmp_cfg_dict:  Dict[str, Any],
    frame_cfg_dict:  Dict[str, Any],
):
    cfg = UnifyConfig({
        'traintmp_cfg': UnifyConfig(), 'datatmp_cfg': UnifyConfig(),
        'model_cfg': UnifyConfig(), 'evaltmp_cfg': UnifyConfig(), 
        'frame_cfg': UnifyConfig()
    })
    cfg.dataset = dataset 

    # Golbal Configuration  information
    cfg.frame_cfg = UnifyConfig.from_py_module(settings)
    for k,v in frame_cfg_dict.items():
        assert k in cfg.frame_cfg
        assert type(cfg.frame_cfg[k]) is type(v)
        cfg.frame_cfg[k] = v
    
    cfg.dot_set('traintmp_cfg.cls',traintmp_cfg_dict['cls'])
    cfg.dot_set('datatmp_cfg.cls', datatmp_cfg_dict['cls'])
    cfg.dot_set('model_cfg.cls', model_cfg_dict['cls'])
    cfg.dot_set('evaltmp_cfg.clses', evaltmp_cfg_dict['clses'])

    for k, v in traintmp_cfg_dict.items():
        if k == 'cls':
            continue
        cfg['traintmp_cfg'][k] = v
    for k, v in datatmp_cfg_dict.items():
        if k == 'cls':
            continue
        cfg['datatmp_cfg'][k] = v
    for k, v in model_cfg_dict.items():
         if k == 'cls':
            continue
         cfg['model_cfg'][k] = v
    for k, v in evaltmp_cfg_dict.items():
        if k == 'clses':
            continue
        cfg['evaltmp_cfg'][k] = v     

    return cfg