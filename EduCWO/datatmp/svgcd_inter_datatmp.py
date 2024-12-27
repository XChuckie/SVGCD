import os
import random

import torch
import pandas as pd
import numpy as np
import scipy.sparse as sp
from itertools import chain
from typing import Dict, Any
from collections import defaultdict

from torch.utils.data import default_collate
from .base_datatmp import BaseDataTmp


class SVGCDInterDataTmp(BaseDataTmp):
    def __init__(self, cfg, 
                train_dict: Dict[str, Any],
                test_dict: Dict[str, Any],
                feat_name2type: Dict[str, str],
                df_Q,
                val_dict: Dict[str, Any] = None,
                ):
        self.df_Q = df_Q
        super().__init__(cfg, train_dict, test_dict, feat_name2type, val_dict) 

    @staticmethod
    def collate_fn(batch):
        elem = batch[0]
        ret_dict = {key: default_collate([d[key] for d in batch]) for key in elem if key not in {'Q_mat'}}
        ret_dict['Q_mat'] = elem['Q_mat']

        return ret_dict
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        dic =  {
            k: v[index] for k,v in self.dict_data.items()
        }
        dic['Q_mat'] = self.Q_mat

        return dic
    def get_extra_data(self):
        extra_dict = super().get_extra_data()
        return extra_dict
    
    def get_se_SparseGraph(self, add_virtual_node_num=0, add_loop=False):
        def built_norm_adj_mat(inter_adj_mat, _add_virtual_node_num):
                tmp_inter_adj_mat= inter_adj_mat.todok() 
                rowsum = np.array(tmp_inter_adj_mat.sum(axis=-1)) + _add_virtual_node_num + 1e-10 
                d_inv = np.power(rowsum, -0.5).flatten()  
                d_inv[np.isinf(d_inv)] = 0.0
                d_mat = sp.diags(d_inv)
                inter_norm_adj_mat = d_mat.dot(tmp_inter_adj_mat)
                inter_norm_adj_mat = inter_norm_adj_mat.dot(d_mat)
                return inter_norm_adj_mat.tocsr()
        dataPath = f'{self.cfg.frame_cfg.data_folder_path}'
        try:
            se_adj_mat = sp.load_npz(dataPath + '/se_adj_mat.npz')  
            se_correct_adj_mat = sp.load_npz(dataPath + '/se_correct_adj_mat.npz')  
            se_incorrect_adj_mat = sp.load_npz(dataPath + '/se_incorrect_adj_mat.npz')  
            se_norm_adj_mat = sp.load_npz(dataPath + '/se_norm_adj_mat.npz') 
            se_correct_norm_adj_mat = sp.load_npz(dataPath + '/se_correct_norm_adj_mat.npz')
            se_incorrect_norm_adj_mat = sp.load_npz(dataPath + '/se_incorrect_norm_adj_mat.npz')
            print("successfully loaded...")
        except:
            import time
            stu_count = self.datatmp_cfg['dt_info']['stu_count']
            exer_count = self.datatmp_cfg['dt_info']['exer_count']
            # DataPath = f'{self.cfg.frame_cfg.data_folder_path}/{self.cfg.dataset}'
            seData = pd.read_csv(dataPath + f"/{self.cfg.dataset}-train.inter.csv", sep=',', encoding='utf-8')
            '''ALL: student-exercise interactions'''
            se_data = pd.DataFrame({
                'stu_id': seData['stu_id:token'],
                'exer_id': seData['exer_id:token'],
                'label': seData['label:float']
            }).drop_duplicates()
            '''Correct: student-exercise interactions'''
            se_correct_data =  se_data[se_data['label'] == 1][['stu_id', 'exer_id']]
            '''Wrong: student-exercise interactions'''
            se_incorrect_data =  se_data[se_data['label'] == 0][['stu_id', 'exer_id']]
            print("generating matrix about between stu and exer...")
            s = time.time()
            '''Built: student-exercise interactive matrix'''
            se_mat = sp.csr_matrix((np.ones(len(se_data['exer_id'])), (se_data['stu_id'], se_data['exer_id'])), shape=(stu_count, exer_count))
            se_correct_mat = sp.csr_matrix((np.ones(len(se_correct_data['exer_id'])), (se_correct_data['stu_id'], se_correct_data['exer_id'])), shape=(stu_count, exer_count))
            se_incorrect_mat = sp.csr_matrix((np.ones(len(se_incorrect_data['exer_id'])), (se_incorrect_data['stu_id'], se_incorrect_data['exer_id'])), shape=(stu_count, exer_count))
            sp.save_npz(dataPath + '/se_mat.npz', se_mat) 
            sp.save_npz(dataPath + '/se_correct_mat.npz', se_correct_mat)  
            sp.save_npz(dataPath + '/se_incorrect_mat.npz', se_incorrect_mat) 
            '''Built: student-exercise adjacency matrix'''
            def built_neighbor_adj_mat(inter_mat):
                tmp_inter_mat = inter_mat.tolil()
                inter_adj_mat = sp.dok_matrix((stu_count + exer_count, stu_count + exer_count), dtype=np.float32)
                inter_adj_mat = inter_adj_mat.tolil()
                inter_adj_mat[:stu_count, stu_count:] = tmp_inter_mat
                inter_adj_mat[stu_count:, :stu_count] = tmp_inter_mat.T
                return inter_adj_mat.tocsr()  
            se_adj_mat = built_neighbor_adj_mat(se_mat)  
            se_correct_adj_mat = built_neighbor_adj_mat(se_correct_mat)  
            se_incorrect_adj_mat = built_neighbor_adj_mat(se_incorrect_mat)  
            sp.save_npz(dataPath + '/se_adj_mat.npz', se_adj_mat)
            sp.save_npz(dataPath + '/se_correct_adj_mat.npz', se_correct_adj_mat)  
            sp.save_npz(dataPath + '/se_incorrect_adj_mat.npz', se_incorrect_adj_mat)  
            '''Built: student-exercise normalized Laplace matrix'''
            se_norm_adj_mat = built_norm_adj_mat(se_adj_mat, add_virtual_node_num)  
            se_correct_norm_adj_mat = built_norm_adj_mat(se_correct_adj_mat, add_virtual_node_num)  
            se_incorrect_norm_adj_mat = built_norm_adj_mat(se_incorrect_adj_mat, add_virtual_node_num)  
            sp.save_npz(dataPath + '/se_norm_adj_mat.npz', se_norm_adj_mat)
            sp.save_npz(dataPath + '/se_correct_norm_adj_mat.npz', se_correct_norm_adj_mat)
            sp.save_npz(dataPath + '/se_incorrect_norm_adj_mat.npz', se_incorrect_norm_adj_mat)
            e = time.time()
            print(f"costing {e - s}s, completed matrix buliting between stu and exer...")
        if add_loop or add_virtual_node_num:
            if add_loop:
                se_adj_mat = (se_adj_mat + sp.eye(se_adj_mat.shape[0]))  * 1.0 
                se_correct_adj_mat = (se_correct_adj_mat + sp.eye(se_correct_adj_mat.shape[0]))  * 1.0
                se_incorrect_adj_mat = (se_incorrect_adj_mat + sp.eye(se_incorrect_adj_mat.shape[0]))  * 1.0
            se_norm_adj_mat = built_norm_adj_mat(se_adj_mat, add_virtual_node_num)  
            se_correct_norm_adj_mat = built_norm_adj_mat(se_correct_adj_mat, add_virtual_node_num) 
            se_incorrect_norm_adj_mat = built_norm_adj_mat(se_incorrect_adj_mat, add_virtual_node_num)  
        se_norm_graph = self.covert_csr_to_tensorSparse(se_norm_adj_mat)  
        se_norm_graph = se_norm_graph.coalesce()  
        se_correct_norm_graph = self.covert_csr_to_tensorSparse(se_correct_norm_adj_mat)
        se_correct_norm_graph = se_correct_norm_graph.coalesce()
        se_incorrect_norm_graph = self.covert_csr_to_tensorSparse(se_incorrect_norm_adj_mat)
        se_incorrect_norm_graph = se_incorrect_norm_graph.coalesce()
        se_norm_graph_info = {
            'se_norm_graph': se_norm_graph,
            'se_correct_norm_graph': se_correct_norm_graph,
            'se_incorrect_norm_graph': se_incorrect_norm_graph
        }
        return se_norm_graph_info

    def get_row_div(self, graph, add_virtual_node_num=0):
        indices = graph._indices()
        rowsum = torch.bincount(indices[0], minlength=graph.size(0)) + add_virtual_node_num  
        d_inv = torch.reciprocal(rowsum)
        return d_inv
    def get_edge_index(self):
        se_norm_graph_info = self.get_se_SparseGraph()
        return se_norm_graph_info['se_norm_graph'].coalesce().indices() 
    def covert_csr_to_tensorSparse(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col])
        adj_matrix = torch.sparse.FloatTensor(torch.tensor(indices).long(), torch.FloatTensor(coo.data), torch.Size(coo.shape))
        return adj_matrix

    def _init_data_after_dt_info(self):
        super()._init_data_after_dt_info()
        
        self.Q_mat = self._get_Q_mat_from_df_arr(
            self.df_Q, 
            self.datatmp_cfg['dt_info']['exer_count'], 
            self.datatmp_cfg['dt_info']['cpt_count']
        )
        
    def _get_Q_mat_from_df_arr(self, df_Q_arr, exer_count, cpt_count):
        """ Get Q Matrix
        """
        # Q_mat = torch.zeros((exer_count, cpt_count), dtype=torch.int64)
        Q_mat = torch.full((exer_count, cpt_count), 0, dtype=torch.float32)
        for _, item in df_Q_arr.iterrows():
            for cpt_id in item['cpt_seq']: Q_mat[item['exer_id'], cpt_id] = 1.0
        return Q_mat

    @classmethod
    def from_cfg(cls, cfg):
        feat_name2type, train_df, val_df, test_df = cls.read_data(cfg)  
        Q_feat_name2type, df_Q = cls.read_Q_correlation(cfg)  
        feat_name2type.update(Q_feat_name2type)

        return cls(
            cfg=cfg,
            train_dict = cls.df2dict(train_df),
            test_dict = cls.df2dict(test_df),
            feat_name2type = feat_name2type,
            df_Q=df_Q,
            val_dict=cls.df2dict(val_df) if val_df is not None else None,
        )
    
    @classmethod
    def read_data(cls, cfg):
        return super().read_data(cfg)
    
    @classmethod
    def read_Q_correlation(cls, cfg):
        return super().read_Q_correlation(cfg)
    
    def _stat_dataset_info(self):
        super()._stat_dataset_info()
        self.cpt_count = len(set(list(chain.from_iterable(self.df_Q['cpt_seq'].to_list()))))
        
        self.datatmp_cfg['dt_info'].update({
            'cpt_count': self.cpt_count,
        })