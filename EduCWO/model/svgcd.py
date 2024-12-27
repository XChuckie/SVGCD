from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .basemodel import BaseModel
from .common_utils import PosLinear

init = nn.init.xavier_uniform_


class SVGCD(BaseModel):
    def __init__(self, cfg, datatmp):
        super().__init__(cfg, datatmp) 
    
    def build_cfg(self):
        self.n_user = self.datatmp_cfg['dt_info']['stu_count']
        self.n_exer = self.datatmp_cfg['dt_info']['exer_count']
        self.n_cpt = self.datatmp_cfg['dt_info']['cpt_count']
    
    def add_extra_data(self, **kwargs):
        super().add_extra_data()
        
    def build_model(self):
        self.stu_emb = nn.Parameter(init(torch.empty(self.n_user, self.model_cfg['emb_dim'])))
        self.exer_emb = nn.Parameter(init(torch.empty(self.n_exer, self.model_cfg['emb_dim'])))
        self.cpt_emb = nn.Parameter(init(torch.empty(self.n_cpt, self.model_cfg['emb_dim'])))
        self.exer_disc = nn.Parameter(init(torch.empty(self.n_exer, 1)))
        # Cognitve Diagnosis Layer 
        '''Constraint students and exercises to conform to the monotonicity assumption'''
        self.prednet_stu = PosLinear(self.n_cpt , self.n_cpt)
        self.prednet_exer = PosLinear(self.n_cpt, self.n_cpt)
        '''Verifying the correctness of the diagnosis results'''
        self.prednet_1= PosLinear(self.n_cpt, self.model_cfg['dnn_units'][0])
        self.drop_1 = nn.Dropout(p=self.model_cfg['dropout_rate'])
        self.prednet_2= PosLinear(self.model_cfg['dnn_units'][0], self.model_cfg['dnn_units'][1])
        self.drop_2 = nn.Dropout(p=self.model_cfg['dropout_rate'])
        self.prednet_final = PosLinear(self.model_cfg['dnn_units'][1], 1)
        # Semantic GNN Layer
        se_norm_graph_info = self.dtp.get_se_SparseGraph(add_virtual_node_num=1, add_loop=True)  
        self.se_ct_norm_graph = se_norm_graph_info['se_correct_norm_graph'].to(self.device)
        self.se_ict_norm_graph = se_norm_graph_info['se_incorrect_norm_graph'].to(self.device)
        # Semantic-specific Graph Reconstruction
        def get_upper_lower(se_norm_graph):
            ind_row, ind_col  = se_norm_graph._indices()
            values = se_norm_graph._values()
            mask_upper = (ind_row <= ind_col)
            ind_upper = torch.stack((ind_row[mask_upper], ind_col[mask_upper]), dim=0)
            valuer_upper = values[mask_upper]
            mask_lower = (ind_row > ind_col)
            ind_lower = torch.stack((ind_row[mask_lower], ind_col[mask_lower]), dim=0)
            valuer_lower = values[mask_lower]
            ct_ind = torch.cat((ind_upper, ind_lower), dim=1)
            ct_value = torch.cat((valuer_upper, valuer_lower), dim=0)
            return ind_upper, ind_lower, ct_ind, ct_value
        self.ct_ind_upper, self.ct_ind_lower, self.ct_ind, self.ct_value  = get_upper_lower(self.se_ct_norm_graph)        
        self.ict_ind_upper, self.ict_ind_lower, self.ict_ind, self.ict_value = get_upper_lower(self.se_ict_norm_graph)
        '''Graph Encoder'''
        self.ct_encoder_std = nn.Sequential(nn.Linear(self.n_cpt , self.n_cpt ), nn.Linear(self.n_cpt, self.n_cpt), nn.Softplus())  
        self.ict_encoder_std = nn.Sequential(nn.Linear(self.n_cpt , self.n_cpt), nn.Linear(self.n_cpt, self.n_cpt), nn.Softplus())
        '''Graph Decoder'''
        self.ct_decoder = nn.Sequential(nn.Linear(self.n_cpt , self.n_cpt ), nn.Linear(self.n_cpt , 1)) 
        self.ict_decoder = nn.Sequential(nn.Linear(self.n_cpt , self.n_cpt ), nn.Linear(self.n_cpt , 1))
        # Semantic-specific Contrastive Learning
        self.tau = self.model_cfg['cl_tau']

    def graph_cl_learning(self, stu_abit_all, exer_diff_all, bs_stu_id=[], bs_exer_id=[]):
        loss=0
        selected_stus, selected_exers = None, None

        selected_stus=[stu_abit_all[0][bs_stu_id], stu_abit_all[1][bs_stu_id]]
        selected_exers=[exer_diff_all[0][bs_exer_id], exer_diff_all[1][bs_exer_id]]
        for embed in [selected_stus, selected_exers]:
            embed0 = F.normalize(embed[0], p = 2, dim = -1)
            embed1 = F.normalize(embed[1], p = 2, dim = -1)
            ratings = torch.matmul(embed0, torch.transpose(embed1, 0, 1))

            ratings_diag = torch.diag(ratings)
            numerator = torch.exp(ratings_diag / self.tau)
            denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
            
            ssm_loss = - torch.mean(torch.log(numerator / denominator))
            loss = loss + ssm_loss
        return loss

    def graph_emb_compute(self, se_ct_adj, se_ict_adj, x_emb, y_emb, reduction='sum', dropout=False):
        all_embed = torch.cat((x_emb, y_emb), dim=0) 
        emb_lists = [all_embed]
        agg_emb_ct, agg_emb_ict = all_embed, all_embed
        emb_ct_lists, emb_ict_list = [], []
        for _ in range(self.model_cfg['n_gnn_layer']):
            if not dropout:
                '''Fusion results form Semantic-correct GNN and Semantic-wrong GNN'''
                agg_emb_ct = torch.sparse.mm(se_ct_adj, emb_lists[-1])
                agg_emb_ict = torch.sparse.mm(se_ict_adj, emb_lists[-1])

                agg_emb = agg_emb_ct + agg_emb_ict
                emb_lists.append(agg_emb)
            else:
                '''Semantic-correct GNN and Semantic-wrong GNN'''
                agg_emb_ct = torch.sparse.mm(se_ct_adj, agg_emb_ct)
                agg_emb_ict = torch.sparse.mm(se_ict_adj, agg_emb_ict)
                    
                emb_ct_lists.append(agg_emb_ct)
                emb_ict_list.append(agg_emb_ict)
        if reduction == "mean":
            if not dropout:
                light_out = torch.mean(torch.stack(emb_lists, dim=1), dim=1) 
                stu_abit_all, exer_diff_all = [light_out[:self.n_user, :]], [light_out[self.n_user:, :]]
            else:
                light_out_ct = torch.mean(torch.stack(emb_ct_lists, dim=1), dim=1)
                light_out_ict = torch.mean(torch.stack(emb_ict_list, dim=1), dim=1)
                stu_abit_all = [light_out_ct[:self.n_user, :], light_out_ict[:self.n_user, :]]
                exer_diff_all = [light_out_ct[self.n_user:, :], light_out_ict[self.n_user:, :]]
        elif reduction == "final":
            light_out = emb_lists[-1]
            stu_abit_all = [light_out[:self.n_user, :]]
            exer_diff_all = [light_out[self.n_user:, :]]
        elif reduction == "sum":
            if not dropout:
                light_out = torch.sum(torch.stack(emb_lists, dim=1), dim=1) 
                stu_abit_all, exer_diff_all = [light_out[:self.n_user, :]], [light_out[self.n_user:, :]]
            else:
                light_out_ct = torch.sum(torch.stack(emb_ct_lists, dim=1), dim=1)
                light_out_ict = torch.sum(torch.stack(emb_ict_list, dim=1), dim=1)
                stu_abit_all = [light_out_ct[:self.n_user, :], light_out_ict[:self.n_user, :]]
                exer_diff_all = [light_out_ct[self.n_user:, :], light_out_ict[self.n_user:, :]]
        elif reduction == "concact":
            if not dropout:
                light_out = torch.cat(emb_lists, dim=-1)
                stu_abit_all, exer_diff_all = [light_out[:self.n_user, :]], [light_out[self.n_user:, :]]
            else:
                light_out_ct = torch.cat(emb_ct_lists, dim=-1) 
                light_out_ict = torch.cat(emb_ict_list, dim=-1) 
                stu_abit_all = [light_out_ct[:self.n_user, :], light_out_ict[:self.n_user, :]]
                exer_diff_all = [light_out_ct[self.n_user:, :], light_out_ict[self.n_user:, :]]

        return stu_abit_all, exer_diff_all

    def graph_representations(self, dropout=False, Q_mat=None, vgae_kl=False):
        def _get_stu_exer_to_cpt(stu_emb, exer_emb, cpt_emb):
            # Student ability
            batch, dim = stu_emb.size()
            stu_emb = stu_emb.view(batch, 1, dim).repeat(1, self.n_cpt, 1)
            stu_cpt_emb = cpt_emb.repeat(batch, 1).view(batch, self.n_cpt, -1)
            stu_state = (stu_emb * stu_cpt_emb).sum(dim=-1, keepdim=False) 
            # Exercise difficulty
            batch, dim = exer_emb.size()
            exer_emb = exer_emb.view(batch, 1, dim).repeat(1, self.n_cpt, 1)
            exer_cpt_emb = cpt_emb.repeat(batch, 1).view(batch, self.n_cpt, -1)
            exer_diff = (exer_emb * exer_cpt_emb).sum(dim=-1, keepdim=False) 
            return stu_state, exer_diff
        stu_emb_tmp, exer_emb_tmp, cpt_emb_tmp = self.stu_emb, self.exer_emb, self.cpt_emb  # initial emb 
        stu_state, exer_diff = _get_stu_exer_to_cpt(stu_emb_tmp, exer_emb_tmp, cpt_emb_tmp)  # Initialization of knowledge integration
        # Whether update GNN Graph Structure
        if dropout:
            if self.model_cfg['dp_type'] == 'vgae': 
                se_ct_adj, se_ict_adj = self.se_ct_norm_graph, self.se_ict_norm_graph
                stu_abit_all, exer_diff_all = self.graph_emb_compute(se_ct_adj, se_ict_adj, stu_state, exer_diff, reduction='mean', dropout=True)

                if vgae_kl: 
                    '''Graph Reconstructive-Contrastive Learning'''
                    se_ct_adj_new, se_ict_adj_new = self.vgae_genetare(stu_abit_all, exer_diff_all, Q_mat=Q_mat)
                    stu_abit_mix, exer_diff_mix = self.graph_emb_compute(se_ct_adj_new, se_ict_adj_new, stu_state, exer_diff, reduction='sum', dropout=False)
                    return stu_abit_all, exer_diff_all, stu_abit_mix, exer_diff_mix
                else:
                    '''Contrastive Learning View'''
                    with torch.no_grad():
                        se_ct_adj_new, se_ict_adj_new = self.vgae_genetare(stu_abit_all, exer_diff_all, Q_mat=Q_mat)
                    stu_abit_all_new, exer_diff_all_new = self.graph_emb_compute(se_ct_adj_new, se_ict_adj_new, stu_state, exer_diff, reduction='sum', dropout=dropout)
                    return stu_abit_all_new, exer_diff_all_new
        else:
            se_ct_adj, se_ict_adj = self.se_ct_norm_graph, self.se_ict_norm_graph
            stu_abit_all, exer_diff_all = self.graph_emb_compute(se_ct_adj, se_ict_adj, stu_state, exer_diff)
            return stu_abit_all[0], exer_diff_all[0]
    
    def vgae_encoder(self, stu_abit_all, exer_diff_all):
        ct_node_emb = torch.concat([stu_abit_all[0], exer_diff_all[0]], axis=0)
        ict_node_emb = torch.concat([stu_abit_all[1], exer_diff_all[1]], axis=0)

        ct_mean, ct_std = ct_node_emb, self.ct_encoder_std(ct_node_emb)
        ict_mean, ict_std = ict_node_emb, self.ict_encoder_std(ict_node_emb)

        ct_guassian_noise = torch.randn(ct_mean.shape).cuda()
        ict_guassian_noise = torch.randn(ict_mean.shape).cuda()

        ct_ret = ct_guassian_noise * torch.exp(ct_std) + ct_mean
        ict_ret = ict_guassian_noise * torch.exp(ict_std) + ict_mean
        
        return [ct_ret, ct_mean, ct_std], [ict_ret, ict_mean, ict_std]

    def vgae_genetare(self, stu_abit_all, exer_diff_all, Q_mat=None):
        def graph_norm(se_graph):
            ind = se_graph._indices()
            row, col = ind[0, :], ind[1, :]
            rowsum = torch.sparse.sum(se_graph, dim=-1).to_dense()
            d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
            row_inv_sqrt = d_inv_sqrt[row]
            col_inv_sqrt = d_inv_sqrt[col]
            values = torch.mul(se_graph._values(), row_inv_sqrt)
            values = torch.mul(values, col_inv_sqrt)
            return torch.sparse.FloatTensor(se_graph._indices(), values, se_graph.shape)
        ct_info, ict_info = self.vgae_encoder(stu_abit_all, exer_diff_all)
        ct_res_input = torch.cat((ct_info[0][self.ct_ind_upper[0]] - ct_info[0][self.ct_ind_upper[1]], ct_info[0][self.ct_ind_lower[1]] - ct_info[0][self.ct_ind_lower[0]]), dim=0)
        ct_edge_pred = torch.sigmoid(self.ct_decoder(ct_res_input)) 
        ict_res_input = torch.cat((ict_info[0][self.ict_ind_upper[1]] - ct_info[0][self.ict_ind_upper[0]], ct_info[0][self.ict_ind_lower[0]] - ct_info[0][self.ict_ind_lower[1]]), dim=0)
        ict_edge_pred = torch.sigmoid(self.ict_decoder(ict_res_input))

        val_ct_new = torch.squeeze(ct_edge_pred)
        val_ict_new = torch.squeeze(ict_edge_pred)

        se_ct_graph = torch.sparse.FloatTensor(self.ct_ind, val_ct_new, self.se_ct_norm_graph.shape)
        se_ict_graph = torch.sparse.FloatTensor(self.ict_ind, val_ict_new, self.se_ict_norm_graph.shape)
        se_ct_norm_graph = graph_norm(se_ct_graph)
        se_ict_norm_graph = graph_norm(se_ict_graph)
        return se_ct_norm_graph, se_ict_norm_graph

    def forward(self, batch_s_emb, batch_e_emb, batch_e_disc, e_Qmat):
        batch_s_emb, batch_e_emb = torch.sigmoid(self.prednet_stu(batch_s_emb)), torch.sigmoid(self.prednet_exer(batch_e_emb))

        batch_e_disc = torch.sigmoid(batch_e_disc)
        o = e_Qmat * (batch_s_emb - batch_e_emb) * batch_e_disc
        input_x = self.drop_1(torch.tanh(self.prednet_1(o)))
        input_x = self.drop_2(torch.tanh(self.prednet_2(input_x)))
        output = self.prednet_final(input_x).sigmoid()
        return output
            
    def forward_test(self, stu_id, exer_id, Q_mat, **kwargs):
        stu_abit, exer_diff = self.graph_representations()
        batch_s_emb, batch_e_emb = torch.sigmoid(self.prednet_stu(stu_abit[stu_id])), torch.sigmoid(self.prednet_exer(exer_diff[exer_id]))

        batch_e_disc = torch.sigmoid(self.exer_disc[exer_id])
        e_Qmat = Q_mat[exer_id]
        o = e_Qmat * (batch_s_emb - batch_e_emb) * batch_e_disc
        input_x = self.drop_1(torch.tanh(self.prednet_1(o)))
        input_x = self.drop_2(torch.tanh(self.prednet_2(input_x)))
        output = self.prednet_final(input_x).sigmoid()
        return output

    @torch.no_grad()
    def predict(self, stu_id, exer_id, Q_mat, **kwargs):
        return {
            'y_pd': self.forward_test(stu_id, exer_id, Q_mat).flatten(),
        }

    def regularize(self, model):
        reg_loss = 0
        for W in model.parameters():
            reg_loss += W.norm(2).square()
        return reg_loss

    def cal_loss(self, **kwargs):
        stu_id = kwargs['stu_id']
        exer_id = kwargs['exer_id']
        Q_mat = kwargs['Q_mat']
        stu_abit, exer_diff = self.graph_representations()  
        bce_loss, reg_loss = torch.tensor(0), torch.tensor(0)

        y_pd = self(stu_abit[stu_id], exer_diff[exer_id], self.exer_disc[exer_id], Q_mat[exer_id])
        bce_loss = F.binary_cross_entropy(
            input=y_pd.flatten(), target=kwargs['label']
        )
        if reg_loss.item() == 0: 
            loss = bce_loss
            loss_dict = {'bce_loss': bce_loss}
        else: 
            loss = bce_loss + reg_loss
            loss_dict = {'bce_loss': bce_loss, 'reg_loss': reg_loss}

        return loss, loss_dict
    
    def cal_loss_cl(self, **kwargs):
        stu_id = kwargs['stu_id']
        exer_id = kwargs['exer_id']
        Q_mat = kwargs['Q_mat']
        stu_ct_abit_all, exer_ct_diff_all = [], []
        stu_ict_abit_all, exer_ict_diff_all = [], []
        bce_loss, cl_loss = torch.tensor(0), torch.tensor(0)

        for _ in [0, 1]:
            stu_abit_all_new, exer_diff_all_new= self.graph_representations(Q_mat=Q_mat, dropout=True)
            stu_ct_abit_all.append(stu_abit_all_new[0]), exer_ct_diff_all.append(exer_diff_all_new[0])
            stu_ict_abit_all.append(stu_abit_all_new[1]), exer_ict_diff_all.append(exer_diff_all_new[1])
        ct_cl_loss = self.graph_cl_learning(stu_ct_abit_all, exer_ct_diff_all, stu_id[kwargs['label']==1], exer_id[kwargs['label']==1])
        ict_cl_loss = self.graph_cl_learning(stu_ict_abit_all, exer_ict_diff_all, stu_id[kwargs['label']==0], exer_id[kwargs['label']==0])
        cl_loss = (ct_cl_loss * len(stu_id[kwargs['label']==1]) / len(stu_id) + ict_cl_loss * len(stu_id[kwargs['label']==0]) / len(stu_id)) * self.model_cfg['cl_weight']
        
        if bce_loss.item() == 0: 
            loss = cl_loss
            loss_dict = {'cl_loss': cl_loss}
        else:
            loss = bce_loss + cl_loss
            loss_dict = {'sub_bce_loss': bce_loss, 'cl_loss': cl_loss}
        return loss, loss_dict

    def cal_loss_kl(self, **kwargs):
        stu_id = kwargs['stu_id']
        exer_id = kwargs['exer_id']
        Q_mat = kwargs['Q_mat']
        stu_abit_all, exer_diff_all, stu_abit_mix, exer_diff_mix = self.graph_representations(dropout=True, vgae_kl=True)
        ct_info, ict_info = self.vgae_encoder(stu_abit_all, exer_diff_all)
        
        '''KL-divergence loss'''
        ct_kl_divergence = - 0.5 * (1 + 2 * torch.log(ct_info[2]) - ct_info[1]**2 - ct_info[2]**2).sum(dim=1)
        ict_kl_divergence = - 0.5 * (1 + 2 * torch.log(ict_info[2]) - ict_info[1]**2 - ict_info[2]**2).sum(dim=1)
        kl_divergence = (ct_kl_divergence * len(stu_id[kwargs['label']==1]) / len(stu_id) + ict_kl_divergence * len(stu_id[kwargs['label']==0]) / len(stu_id)) * self.model_cfg['beta'] 
        '''Difference loss between Reconstructed Graph and True Graph'''
        y_pd = self(stu_abit_mix[0][stu_id], exer_diff_mix[0][exer_id], self.exer_disc[exer_id], Q_mat[exer_id])
        res_bce_loss = F.binary_cross_entropy(
            input=y_pd.flatten(), target=kwargs['label'], reduction='none'
        )

        loss = (res_bce_loss + kl_divergence.mean()).mean() 
        loss_dict = {'res_bce_loss': res_bce_loss.mean(), 'kl_loss': kl_divergence.mean()}

        return loss, loss_dict

