import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Any
import importlib
import traceback
import logging
import shutil

from utils import UnifyConfig
from confs import get_global_cfg, init_all
import wandb
import yaml


def update_config_with_sweep_parameters(config):
    for key in wandb.config.keys():
        if key in config['traintmp_cfg_dict']:
            config['traintmp_cfg_dict'][key] = wandb.config[key]
        elif key in config['datatmp_cfg_dict']:
            config['datatmp_cfg_dict'][key] = wandb.config[key]
        elif key in config['model_cfg_dict']:
            config['model_cfg_dict'][key] = wandb.config[key]
        elif key in config['evaltmp_cfg_dict']:
            config['evaltmp_cfg_dict'][key] = wandb.config[key]

def main(
    dataset: str = None,
    traintmp_cfg_dict: Dict[str, Any] = {},
    datatmp_cfg_dict:  Dict[str, Any] = {},
    model_cfg_dict: Dict[str, Any] = {},
    evaltmp_cfg_dict:  Dict[str, Any] = {},
    frame_cfg_dict:  Dict[str, Any] = {},

):
    cfg: UnifyConfig = get_global_cfg(
        dataset, traintmp_cfg_dict,datatmp_cfg_dict, model_cfg_dict, evaltmp_cfg_dict, frame_cfg_dict
    )
    init_all(cfg)
    try:
        cfg.logger.info("====" * 15)
        cfg.logger.info(f"[ID]: {cfg.frame_cfg.ID}")
        cfg.logger.info(f"[DATASET]: {cfg.dataset}")
        cfg.logger.info(f"[ARGV]: {sys.argv}")
        cfg.logger.info(f"[ALL_CFG]: \n{cfg.dump_fmt()}")
        cfg.dump_file(f"{cfg.frame_cfg.temp_folder_path}/cfg.json")  
        cfg.logger.info("====" * 15)
        
        traintmp_cls = importlib.import_module('EduCWO.traintmp').__getattribute__(cfg.traintmp_cfg['cls'])
        traintmp = traintmp_cls(cfg)
        traintmp.start()

        cfg.logger.info(f"Task: {cfg.frame_cfg.ID} Completed!")
        logging.shutdown()
        shutil.move(cfg.frame_cfg.temp_folder_path,
                    cfg.frame_cfg.archive_folder_path)

    except Exception as e:  
        cfg.logger.error(traceback.format_exc())
        raise e

if __name__ == "__main__":
    with open("../confs/exMatCons/assist-0910/svgcd.yaml", 'r') as stream: 
        config = yaml.safe_load(stream) 
    # wandb.init(project='slt-educd', config=config, mode='online')  # wandb for visualizing
    # update_config_with_sweep_parameters(config)
    main(
        dataset = config['dataset'],
        traintmp_cfg_dict = config['traintmp_cfg_dict'],
        datatmp_cfg_dict = config['datatmp_cfg_dict'],
        model_cfg_dict = config['model_cfg_dict'],
        evaltmp_cfg_dict = config['evaltmp_cfg_dict']
    )
