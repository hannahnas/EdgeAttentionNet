import torch
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
from skipattention import SkipAttentionNet
from criterions import cross_entropy_loss2d
from evaluation import compute_eval_measures
# from evaluation import compute_eval_measures

class SkipAttentionModel(pl.LightningModule):
    def __init__(self, hyper_params):
        super().__init__()

        self.hyper_params = hyper_params
        self.model = SkipAttentionNet(hyper_params)
        

    def forward(self, batch):
        rgb = batch['rgb']
        depth = batch['depth']
        mask = batch['mask']
        edges = batch['edges']
        masked_depth = (1 - mask) * depth 

        depth = self.model(edges, rgb, masked_depth)

        return depth
    
    def _get_loss(self, batch):
        depth_gt = batch['depth']
        mask = batch['mask']

        depth_pred = self.forward(batch)

        l1_loss = F.l1_loss(depth_pred * mask, depth_gt * mask, reduction='mean') / mask.sum()

        return l1_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hyper_params['lr'])

        return {"optimizer": optimizer, "monitor": "total_val_loss"}

    def training_step(self, batch, batch_idx):
        l1_loss = self._get_loss(batch)
        self.log('train_l1_depth_loss', l1_loss)

        return l1_loss

    def validation_step(self, batch, batch_idx):
        l1_loss = self._get_loss(batch)
        self.log('val_l1_depth_loss', l1_loss)


    def test_step(self, batch, batch_idx):
        l1_loss = self._get_loss(batch)
        self.log('test_l1_depth_loss', l1_loss)

        depth_pred = self.forward(batch)
        abs_rel, _, rmse, _, a1, a2, a3, _ = compute_eval_measures(batch['depth'], depth_pred, batch['mask'])
        self.log('abs rel', abs_rel)
        self.log('delta 1.25', a1)
        self.log('delta 1.25^2', a2)
        self.log('delta 1.25^3', a3)
        self.log('rmse', rmse)