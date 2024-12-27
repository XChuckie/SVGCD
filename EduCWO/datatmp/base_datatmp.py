import os
import copy
import math
import torch
import logging
import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Any

from torch.utils.data import Dataset, DataLoader
from utils import UnifyConfig


class DataTmpMode(Enum):
    TRAIN=1
    VALID=2
    TEST=3
    MANAGER=4

class BaseDataTmp(Dataset):
    def __init__(self, cfg, 
                train_dict: Dict[str, Any],
                test_dict: Dict[str, Any],
                feat_name2type: Dict[str, str],
                val_dict: Dict[str, Any] = None,
                mode: DataTmpMode=DataTmpMode.MANAGER,
                ):
        self.train_dict = train_dict
        self.test_dict = test_dict
        self.val_dict = val_dict
        self.feat_name2type = feat_name2type
        self.mode = mode 

        self.cfg: UnifyConfig = cfg
        self.traintmp_cfg: UnifyConfig = cfg.traintmp_cfg
        self.datatmp_cfg: UnifyConfig = cfg.datatmp_cfg
        self.model_cfg: UnifyConfig = cfg.model_cfg
        self.evaltmp_cfg: UnifyConfig = cfg.evaltmp_cfg
        self.frame_cfg: UnifyConfig = cfg.frame_cfg
        self.logger: logging.Logger = cfg.logger
        self.datatmp_cfg['dt_info'] = {}  
        self._stat_dataset_info()  
        self._init_data_after_dt_info()
        self.logger.info(self.datatmp_cfg['dt_info'])
    
    @staticmethod
    def collate_fn(batch):
        pass

    def build_dataloaders(self):
        batch_size = self.traintmp_cfg['batch_size']
        num_workers = self.traintmp_cfg['num_workers']
        eval_batch_size = self.traintmp_cfg['eval_batch_size']
        train_dataset, val_dataset, test_dataset = self.build_datasets()
        train_loader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, collate_fn=self.collate_fn)
        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=eval_batch_size, num_workers=num_workers, collate_fn=self.collate_fn)
        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(dataset=val_dataset, shuffle=False, batch_size=eval_batch_size, num_workers=num_workers, collate_fn=self.collate_fn)
        return train_loader, val_loader, test_loader
    
    def build_datasets(self):
        train_dataset = self._copy()  
        train_dataset.set_mode(DataTmpMode.TRAIN)

        val_dataset = None
        if self.val_dict is not None:
            val_dataset = self._copy()
            val_dataset.set_mode(DataTmpMode.VALID)
        
        test_dataset = self._copy()  
        test_dataset.set_mode(DataTmpMode.TEST)
        return train_dataset, val_dataset, test_dataset

    def set_mode(self, mode):
        self.mode = mode
        if self.mode is DataTmpMode.MANAGER:
            self.dict_data = None
        elif self.mode is DataTmpMode.TRAIN:
            self.dict_data = self.train_dict
        elif self.mode is DataTmpMode.VALID:
            self.dict_data = self.val_dict
        elif self.mode is DataTmpMode.TEST:
            self.dict_data = self.test_dict
        else:
            raise ValueError(f"unknown type of mode:{self.mode}")
        self.length = next(iter(self.dict_data.values())).shape[0]
    def _copy(self):
        return copy.copy(self)

    def get_extra_data(self):
        return {}

    def _init_data_after_dt_info(self):
        pass
    
    @classmethod
    def from_cfg(cls, cfg):
        pass

    @classmethod
    def read_data(cls, cfg):
        train_df, val_df, test_df = None, None, None
        feat_name2type = {}
        if cfg.datatmp_cfg['is_dataset_divided']:
            train_df, val_df, test_df = cls._read_data_from_divided(cfg)
            train_feat_name2type, train_df = cls._convert_df_to_std_tmp(train_df)
            test_feat_name2type, test_df = cls._convert_df_to_std_tmp(test_df)
            feat_name2type.update(train_feat_name2type)
            feat_name2type.update(test_feat_name2type)
            if val_df is not None:
                val_feat_name2type, val_df = cls._convert_df_to_std_tmp(val_df)
                feat_name2type.update(val_feat_name2type)  
        else:  
            raise NotImplementedError
        return feat_name2type, train_df, val_df, test_df
    
    @classmethod
    def _read_data_from_divided(cls, cfg):
        train_file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-train.inter.csv'
        val_file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-val.inter.csv'
        test_file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-test.inter.csv'
        sep = cfg.datatmp_cfg['seperator']
        train_headers = pd.read_csv(train_file_path, nrows=0).columns.tolist()
        test_headers = pd.read_csv(test_file_path, nrows=0).columns.tolist()
        exclude_feats = ["cpt_seq"] 

        inter_train_df = pd.read_csv(
            train_file_path, sep=sep, encoding='utf-8', usecols=set(train_headers) - set(exclude_feats)
        )
        inter_test_df = pd.read_csv(
            test_file_path, sep=sep, encoding='utf-8', usecols=set(test_headers) - set(exclude_feats)
        )
        inter_val_df = None
        if os.path.exists(val_file_path):
            val_headers = pd.read_csv(val_file_path, nrows=0).columns.tolist()
            inter_val_df = pd.read_csv(
                val_file_path, sep=sep, encoding='utf-8', usecols=set(val_headers) - set(exclude_feats)
            )
        return inter_train_df, inter_val_df, inter_test_df
    
    @staticmethod
    def _convert_df_to_std_tmp(df: pd.DataFrame, inplace=True):
        feat_name2type = {}
        for col in df.columns:
            col_name, col_type = col.split(":")
            feat_name2type[col_name] = col_type
            if col_type == 'token': df[col] = df[col].astype('int64')
            elif col_type == 'float': df[col] = df[col].astype('float32')
            elif col_type == 'token_seq': pass
            else:
                raise ValueError(f"unknown field type of {col_type}")
            if inplace is True:
                df.rename(columns={col: col_name}, inplace=True) 
            else:
                raise NotImplementedError
        return feat_name2type, df
    
    @classmethod
    def read_Q_correlation(cls, cfg):
        file_path = f'{cfg.frame_cfg.data_folder_path}/{cfg.dataset}-Q.csv'
        sep = cfg.datatmp_cfg['seperator']
        df_Q = pd.read_csv(file_path, sep=sep, encoding='utf-8', usecols=['exer_id:token', 'cpt_seq:token_seq'])
        feat_name2type, df_Q = cls._convert_df_to_std_tmp(df_Q)
        df_Q['cpt_seq'] = df_Q['cpt_seq'].astype(str).apply(lambda x: [int(i) for i in x.split(',')])
        return feat_name2type, df_Q

    def _stat_dataset_info(self):
        self.train_num = self.train_dict['stu_id'].shape[0]
        self.val_num = 0 if self.val_dict is None else self.val_dict['stu_id'].shape[0]
        self.test_num = self.test_dict['stu_id'].shape[0]

        self.stu_count = (torch.cat(
            [self.train_dict['stu_id'], self.test_dict['stu_id']] if self.val_dict is None else 
            [self.train_dict['stu_id'], self.val_dict['stu_id'], self.test_dict['stu_id']]
        ).max()).item() + 1
        self.exer_count = (torch.cat(
            [self.train_dict['exer_id'], self.test_dict['exer_id']] if self.val_dict is None else 
            [self.train_dict['exer_id'], self.val_dict['exer_id'], self.test_dict['exer_id']]
        ).max()).item() + 1

        self.datatmp_cfg['dt_info'].update({
            'stu_count': self.stu_count,
            'exer_count': self.exer_count,
            'trainset_count': self.train_num,
            'valset_count': self.val_num,
            'testset_count': self.test_num
        })

    @staticmethod
    def df2dict(dic):
        return {k: torch.from_numpy(dic[k].to_numpy()) for k in dic}




