from modules import ResNetBlockEnc, GatedConv2d, ResNetBlockDec
import torch
import torch.nn as nn
from InpaintDataset import InpaintDataset
from torch.utils.data import DataLoader


class SkipEncoder(nn.Module):
    def __init__(self, in_channels, gated=False):
        super().__init__()
        self._create_network(in_channels, gated=gated)
        self._init_params()

    def _create_network(self, in_channels, gated):

        if gated:
            self.layer_0 = nn.Sequential(
                GatedConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        else:
            self.layer_0 = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )

        self.layer_1 = nn.Sequential(
            ResNetBlockEnc(64, 64, subsample=False, gated=gated),
            ResNetBlockEnc(64, 64, subsample=False, gated=gated)
        )

        self.layer_2 = nn.Sequential(
            ResNetBlockEnc(64, 128, subsample=True, gated=gated),
            ResNetBlockEnc(128, 128, subsample=False, gated=gated)
        )

        self.layer_3 = nn.Sequential(
            ResNetBlockEnc(128, 256, subsample=True, gated=gated),
            ResNetBlockEnc(256, 256, subsample=False, gated=gated)
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.layer_0(x)
        x1 = self.layer_1(x0)
        x2 = self.layer_2(x1)
        x3 = self.layer_3(x2)

        return x1, x2, x3



class SkipAttentionNet(nn.Module):
    def __init__(self, gated_depth_encoder=False):
        super().__init__()
        self.edge_encoder = SkipEncoder(in_channels=1)
        self.rgb_encoder = SkipEncoder(in_channels=3)
        self.depth_encoder = SkipEncoder(in_channels=1, gated=gated_depth_encoder)

        self._create_network(out_channels=1)
        self._init_params()

    def _create_network(self, out_channels):

        self.reduce_features = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.dec_layer_3 = nn.Sequential(
            ResNetBlockDec(256, 256, subsample=False, gated=False),
            ResNetBlockDec(256, 128, subsample=True, gated=False)
        )

        self.dec_layer_2 = nn.Sequential(
            ResNetBlockDec(128, 128, subsample=False, gated=False),
            ResNetBlockDec(128, 64, subsample=True, gated=False)
        )

        self.dec_layer_1 = nn.Sequential(
            ResNetBlockDec(64, 64, subsample=False, gated=False),
            ResNetBlockDec(64, 32, subsample=True, gated=False)
        )

        self.dec_layer_0 = nn.Conv2d(32, out_channels, 3, padding=1)
        

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, edges, rgb, depth):

        edge_x1, edge_x2, edge_x3 = self.edge_encoder(edges)
        rgb_x1, rgb_x2, rgb_x3 = self.rgb_encoder(rgb)
        depth_x1, depth_x2, depth_x3 = self.depth_encoder(depth)

        features = torch.cat([rgb_x3, depth_x3], dim=1)
        features = self.reduce_features(features)

        dec_x2 = self.dec_layer_3(features)
        
        # skip attention
        attn2 = torch.sigmoid(edge_x2)
        dec_x2 = dec_x2 + (attn2 * rgb_x2) + (attn2 * depth_x2) # shape: B, 128, 52, 52

        dec_x1 = self.dec_layer_2(dec_x2)
        # skip attention
        attn1 = torch.sigmoid(edge_x1)
        dec_x1 = dec_x1 + (attn1 * rgb_x1) + (attn1 * depth_x1) # shape: B, 64, 104, 104

        dec_x0 = self.dec_layer_1(dec_x1)

        depth_pred = self.dec_layer_0(dec_x0)

        return depth_pred
        

if __name__ == '__main__':
    train_set = InpaintDataset(split = 'train', samples=4)
    loader = DataLoader(train_set, batch_size=4, shuffle=True,
                            drop_last=True, pin_memory=True, num_workers=1)

    model = SkipAttentionNet()

    for batch in loader:
        edges = batch['edges']
        rgb = batch['rgb']
        depth = batch['depth']
        mask = batch['mask']
        masked_depth = (1 - mask) * depth

        out = model(edges, rgb, masked_depth)
        print(out.shape)
        break

