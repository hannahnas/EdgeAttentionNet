import torch
import pytorch_lightning as pl
from torch import optim
import torch.nn.functional as F
from encoder import ResNetEncoder
from decoder import AttentionDecoder
# from evaluation import compute_eval_measures

class EdgeAttentionModel(pl.LightningModule):
    def __init__(self, gated):
        super().__init__()

        self.rgb_encoder = ResNetEncoder(in_channels=3, gated=gated)
        self.depth_encoder = ResNetEncoder(in_channels=1, gated=gated)

        self.decoder = AttentionDecoder()


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
        edges_gt = batch['edges']
        mask = batch['mask']

        depth_pred, edges_pred = self.forward(batch)

        l1_loss = F.l1_loss(depth_pred * mask, depth_gt * mask) / mask.sum()
        bce_loss = F.binary_cross_entropy_with_logits(edges_pred, edges_gt, pos_weight=torch.Tensor([20.3]).to('cuda'))

        return l1_loss, bce_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)

        return {"optimizer": optimizer, "monitor": "val_loss"}

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