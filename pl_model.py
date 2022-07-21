import torch
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
from encoder import ResNetEncoder
from decoder import AttentionDecoder
from criterions import cross_entropy_loss2d
from evaluation import compute_eval_measures

class EdgeAttentionModel(pl.LightningModule):
    def __init__(self, hyper_params):
        super().__init__()

        self.hyper_params = hyper_params
        gated = hyper_params['gated']

        self.rgb_encoder = ResNetEncoder(in_channels=3, gated=gated)
        self.depth_encoder = ResNetEncoder(in_channels=1, gated=gated)

        self.decoder = AttentionDecoder()

        weights = torch.tensor([1., 1. / 0.03720825]).to('cuda:0')
        self.cross_entropy_fn = torch.nn.CrossEntropyLoss(weight=weights)


    def forward(self, batch):
        rgb = batch['rgb']
        depth = batch['depth']
        mask = batch['mask']
        masked_depth = (1 - mask) * depth 

        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(masked_depth)

        merged_feat = torch.cat([rgb_feat, depth_feat], dim=1)

        depth, edges = self.decoder(merged_feat)

        return depth, edges
    
    def _get_losses(self, batch):
        depth_gt = batch['depth']
        edges_gt = torch.cat([batch['edges'].logical_not(), batch['edges']], dim=1)
        mask = batch['mask']

        depth_pred, edges_pred = self.forward(batch)

        # pos_samples = edges_gt.sum()
        # neg_samples = (edges_gt == 0).sum()
        # weight = neg_samples / pos_samples
        # print(weight)

        l1_loss = self.hyper_params['lambda depth'] * F.l1_loss(depth_pred * mask, depth_gt * mask, reduction='mean') / mask.sum()
        bce_loss = self.hyper_params['lambda edge'] * self.cross_entropy_fn(edges_pred, edges_gt)

        
        # bce_loss =  self.hyper_params['lambda edge'] * cross_entropy_loss2d(edges_pred, edges_gt, cuda=True)
        # bce_loss = self.hyper_params['lambda edge'] * F.binary_cross_entropy_with_logits(edges_pred, edges_gt, reduction='mean', pos_weight=torch.Tensor([weight]).to('cuda'))

        return l1_loss, bce_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hyper_params['lr'])

        return {"optimizer": optimizer, "monitor": "total_val_loss"}

    def training_step(self, batch, batch_idx):
        l1_loss, bce_loss = self._get_losses(batch)
        total_loss = l1_loss + bce_loss
        self.log('train_l1_depth_loss', l1_loss)
        self.log('train_bce_edge_loss', bce_loss)
        self.log('total_train_loss', total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        l1_loss, bce_loss = self._get_losses(batch)
        total_loss = l1_loss + bce_loss
        self.log('val_l1_depth_loss', l1_loss)
        self.log('train_bce_edge_loss', bce_loss)
        self.log('total_val_loss', total_loss)


    def test_step(self, batch, batch_idx):
        l1_loss, bce_loss = self._get_losses(batch)
        self.log('test_l1_depth_loss', l1_loss)
        self.log('train_bce_edge_loss', bce_loss)

        depth_pred, _ = self.forward(batch)
        abs_rel, _, rmse, _, a1, a2, a3, _ = compute_eval_measures(batch['depth'], depth_pred, batch['mask'])
        self.log('abs rel', abs_rel)
        self.log('delta 1.25', a1)
        self.log('delta 1.25^2', a2)
        self.log('delta 1.25^3', a3)
        self.log('rmse', rmse)