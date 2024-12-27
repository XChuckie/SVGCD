import torch
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from .base_traintmp import BaseTrainTmp
from utils import UnifyConfig, set_same_seeds, tensor2npy
from utils.callback import ModelCheckPoint, EarlyStopping, History, BaseLogger, CallbackList


class SVGCDInterTrainTmp(BaseTrainTmp):
    def __init__(self, cfg: UnifyConfig):
        super().__init__(cfg)
    
    def _check_params(self):
        super()._check_params()
    
    def build_loaders(self):
        super().build_loaders()

    def start(self):
        super().start()
        
        # Callbacks
        es_metrics = self.traintmp_cfg["early_stop_metrics"]
        num_stop_rounds = self.traintmp_cfg['num_stop_rounds']
        modelCheckPoint = ModelCheckPoint(
            es_metrics, save_folder_path=f"{self.frame_cfg.temp_folder_path}/pths/"
        )
        earlystopping = EarlyStopping(es_metrics, num_stop_rounds=num_stop_rounds, start_round=1)
        history_cb = History(folder_path=f"{self.frame_cfg.temp_folder_path}/history/", plot_curve=True)
        callbacks = [
            modelCheckPoint, earlystopping, history_cb, 
            BaseLogger(self.logger, group_by_contains=['loss'])
        ]
        self.callback_list = CallbackList(callbacks=callbacks, model=self.model, logger=self.logger)

        for evaltmp in self.evaltmps: 
            evaltmp.set_callback_list(self.callback_list)
            evaltmp.set_dataloaders(train_loader=self.train_loader,   
                                    val_loader=self.val_loader, 
                                    test_loader=self.test_loader
                                )

        # Start Training 
        set_same_seeds(self.traintmp_cfg['seed'])
        if self.val_loader is not None:
            self.fit(train_loader=self.train_loader, val_loader=self.val_loader)
        else:
            self.fit(train_loader=self.train_loader, val_loader=self.test_loader)
        
        if self.val_loader is not None:
            metric_name = self.traintmp_cfg['best_epoch_metric']
            metric = [m for m in modelCheckPoint.metric_list if m.name == metric_name][0]
            fpth =  f"{self.frame_cfg.temp_folder_path}/pths/best-epoch-{metric.best_epoch:03d}-for-{metric.name}.pth"
            self.model.load_state_dict(torch.load(fpth))
            metrics = self.inference(self.test_loader)
            for key, metricws in metrics.items():
                if key == 'total':
                    self.logger.info({f"{metric}": metricws[metric] for metric in metricws})
                else:
                    self.logger.info({f"{key}_{metric}": metricws[metric] for metric in metricws})
            History.dump_json(metrics, f"{self.frame_cfg.temp_folder_path}/result.json")
        
        if self.traintmp_cfg['unsave_best_epoch_pth']: 
            shutil.rmtree(f"{self.frame_cfg.temp_folder_path}/pths/")

    def get_optim(self, model):

        optimizer = self.traintmp_cfg['optim']
        lr = self.traintmp_cfg['lr']
        weight_decay = self.traintmp_cfg['weight_decay']
        eps = self.traintmp_cfg['eps']
        if optimizer == "adam":
           self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps)
        else:
            raise ValueError("unsupported optimizer")

    def fit(self, train_loader, val_loader):
        self.model.train()
        self.get_optim(self.model)
        self.callback_list.on_train_begin()
        
        for epoch in range(self.traintmp_cfg['epoch_num']):
            self.callback_list.on_epoch_begin(epoch + 1)
            logs = defaultdict(lambda: np.full((len(train_loader),), np.nan, dtype=np.float32))

            for batch_id, batch_dict in enumerate(
                tqdm(train_loader, ncols=self.frame_cfg['TQDM_NCOLS'], desc="[EPOCH={:03d}]".format(epoch + 1))
            ):
                self.optimizer.zero_grad()
                batch_dict = self.batch_dict2device(epoch, batch_id, batch_dict)

                loss_cl, loss_cl_dict = self.model.cal_loss_cl(**batch_dict)
                loss_cl.backward()
                self.optimizer.step()
                loss_vgae, loss_vgae_dict = self.model.cal_loss_kl(**batch_dict)
                loss_vgae.backward()
                self.optimizer.step()
                loss_dict = {**loss_cl_dict, **loss_vgae_dict}
                loss_main, loss_main_dict = self.model.cal_loss(**batch_dict)
                loss_main.backward()
                self.optimizer.step()
                loss_dict = {**loss_main_dict}
                loss_dict = {**loss_main_dict, **loss_cl_dict}
                
                loss_dict = {**loss_main_dict, **loss_cl_dict, **loss_vgae_dict}
                self.optimizer.zero_grad()

                for k in loss_dict: 
                    logs[k][batch_id] = loss_dict[k].item() if loss_dict[k] is not None else np.nan
            for name in logs: logs[name] = float(np.nanmean(logs[name]))

            if val_loader is not None:
                val_metrics = self.inference(val_loader)
                for key, metricws in val_metrics.items():
                    if key == 'total':
                        logs.update({f"{metric}": metricws[metric] for metric in metricws})
                    else:
                        logs.update({f"{key}_{metric}": metricws[metric] for metric in metricws})

            self.callback_list.on_epoch_end(epoch + 1, logs=logs)
            if self.model.share_callback_dict.get('stop_training', False):
                break
            
        self.callback_list.on_train_end()
    
    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        pd_list = list(range(len(loader)))
        gt_list = list(range(len(loader)))
        for idx, batch_dict in enumerate(tqdm(loader, ncols=self.frame_cfg['TQDM_NCOLS'], desc="[PREDICT]")):
            batch_dict = self.batch_dict2device(batch_dict)
            eval_dict = self.model.predict(**batch_dict)
            pd_list[idx] = eval_dict['y_pd']
            gt_list[idx] = batch_dict['label']
        y_pd = torch.hstack(pd_list)
        y_gt = torch.hstack(gt_list)

        eval_data_dict = {
            'y_pd': y_pd,
            'y_gt': y_gt,
        }
        if hasattr(self.model, 'get_stu_status'):
            eval_data_dict.update({
                'stu_stats': tensor2npy(self.model.get_stu_status()),
            })
        if hasattr(loader.dataset, 'Q_mat'):
            eval_data_dict.update({
                'Q_mat': tensor2npy(loader.dataset.Q_mat)
            })

        eval_result = {}
        eval_result['total'] = {}
        for evaltmp in self.evaltmps: 
            eval_result['total'].update(
                evaltmp.eval(
                ignore_cd_metrics=self.traintmp_cfg['ignore_cd_metrics'],
                ignore_bc_metrics=self.traintmp_cfg['ignore_bc_metrics'],
                **eval_data_dict
                )
            )
        return eval_result

    @torch.no_grad()
    def inference(self, loader):
        self.model.eval()
        pd_list = list(range(len(loader)))
        gt_list = list(range(len(loader)))
        for idx, batch_dict in enumerate(tqdm(loader, ncols=self.frame_cfg['TQDM_NCOLS'], desc="[PREDICT]")):
            batch_dict = self.batch_dict2device(0, 0, batch_dict)
            eval_dict = self.model.predict(**batch_dict)
            pd_list[idx] = eval_dict['y_pd']
            gt_list[idx] = batch_dict['label']
        y_pd = torch.hstack(pd_list)
        y_gt = torch.hstack(gt_list)
        eval_data_dict = {
            'y_pd': y_pd,
            'y_gt': y_gt,
        }

        eval_result = {}
        eval_result['total'] = {}
        
        for evaltmp in self.evaltmps: 
            eval_result['total'].update(
                evaltmp.eval(
                ignore_cd_metrics=self.traintmp_cfg['ignore_cd_metrics'], 
                ignore_bc_metrics=self.traintmp_cfg['ignore_bc_metrics'],
                **eval_data_dict
                )
            )

        return eval_result