import torch
import pytorch_lightning as pl
from torch import optim
# train
import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from InpaintDataset import InpaintDataset
from matplotlib import cm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from pl_model import EdgeAttentionModel


device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
CHECKPOINT_PATH = './model_checkpoints'


class VisulizationCallback(pl.Callback):
    def __init__(self, input, set, every_n_epochs=1):
        super().__init__()
        self.input = input
        self.every_n_epochs = every_n_epochs
        self.set = set


    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Do image completion
            with torch.no_grad():
                pl_module.eval()
                reconst_depth, reconst_edges = pl_module(self.input)
                pl_module.train()

            # RGB reconstructions
            mask = self.input['mask']
            depth = self.input['depth']
            reconst_depth = mask * reconst_depth + (1-mask) * depth
            rgb_input = self.input['rgb']
            mask = self.input['mask'].repeat(1, 3, 1, 1)
            edge = self.input['edges'].repeat(1, 3, 1, 1)
            reconst_edges = reconst_edges.repeat(1, 3, 1, 1)

            # DEPTH reconstructions
            
            # apply cmap for visualization purposes
            depth_map = np.apply_along_axis(cm.viridis, 0, depth.cpu().numpy() / depth.max().item())
            depth_gt = torch.from_numpy(np.squeeze(depth_map))[:, :3, :, :].to('cuda')
            
            depth_map_input = torch.from_numpy(np.squeeze(
                depth_map)).to('cuda') * (1 - self.input['mask']) #+ self.input['mask']  # apply mask
            depth_map_input = depth_map_input[:, :3, :, :]

            reconst_depth = reconst_depth / depth.max()
            reconst_depth_map = np.apply_along_axis(
                cm.viridis, 0, reconst_depth.cpu().numpy())
            reconst_depth_map = torch.from_numpy(
                np.squeeze(reconst_depth_map)).to('cuda')
            reconst_depth_map = reconst_depth_map[:, :3, :, :]

            results = torch.stack(
                [rgb_input, depth_map_input, reconst_edges, edge, reconst_depth_map, depth_gt], dim=1).flatten(0, 1)
            results_grid = torchvision.utils.make_grid(
                results, nrow=6, value_range=(0, 1))
            trainer.logger.experiment.add_image(
                f'{self.set} reconstructions', results_grid, global_step=trainer.global_step)



def train(hyper_params, train_loader, val_loader, test_loader, train_set, val_set):
    conv = 'Gated' if hyper_params['gated'] else ''

    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, f"{conv}{hyper_params['model name']}_epochs{hyper_params['epochs']}"),
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=hyper_params['epochs'],
                         log_every_n_steps=5,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="min", monitor="total_val_loss"),
                                    VisulizationCallback(get_images(
                                        train_set, 8, device), 'train', every_n_epochs=1),
                                    VisulizationCallback(get_images(
                                        val_set, 8, device), 'validation', every_n_epochs=1),
                                    LearningRateMonitor("epoch")])
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None

    model = EdgeAttentionModel(gated=hyper_params['gated'])

    trainer.fit(model, train_loader, val_loader)

    # # Test best model on validation and test set
    val_result = trainer.test(
        model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(
        model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}

    return model, result

def get_images(train_set, num, device):
    inds = np.random.choice(np.arange(len(train_set)), num)
    object = {
        'rgb': torch.stack([train_set[i]['rgb'] for i in inds], dim=0).to(device),
        'depth': torch.stack([train_set[i]['depth'] for i in inds], dim=0).to(device),
        'mask': torch.stack([train_set[i]['mask'] for i in inds], dim=0).to(device),
        'edges': torch.stack([train_set[i]['edges'] for i in inds], dim=0).to(device)
    }

    return object

def run_experiment(hyper_params):
    # Reproducability
    pl.seed_everything(42)
    # torch.backends.cudnn.determinstic = True
    # torch.backends.cudnn.benchmark = False

    train_set = InpaintDataset(split = 'train')
    print('N datapoints train set:', len(train_set))

    val_set = InpaintDataset(split = 'val')
    print('N datapoints validation set:', len(val_set))

    test_set = InpaintDataset(split = 'test')
    print('N datapoints test set:', len(test_set))

    train_loader = DataLoader(train_set, batch_size=hyper_params['batch size'], shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=4)

    val_loader = DataLoader(val_set, batch_size=hyper_params['batch size'],
                            shuffle=False, drop_last=False, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=8,
                             shuffle=False, drop_last=False, num_workers=4)

    model, result = train(hyper_params, train_loader,
                          val_loader, test_loader, train_set, val_set)

    return model, result

if __name__ == '__main__':
    import sys
    args = sys.argv
    if (len(args) > 2):
        DEF_ENC = int(args[1])
        USE_EDG = int(args[2])
    else:
        DEF_ENC = True
        USE_EDG = True

    hyper_params = {
        'model name': 'posweight20EdgeAttention',
        'gated': False,
        'batch size': 2,
        'epochs': 5
    }

    # train
    model, result = run_experiment(hyper_params)
    print(result)

    # save validation predictions in obj files
    # alle hyperparams as command line arguments

    pass