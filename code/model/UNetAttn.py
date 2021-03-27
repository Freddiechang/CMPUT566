import torch
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
import pytorch_lightning as pl
from torchvision.utils import make_grid

from loss.loss import nss, cc
from model.UNetParts import *

class UNetAttn(pl.LightningModule):

    def __init__(self):
        super().__init__()

        bilinear = True
        factor = 2 if bilinear else 1
        output_channels = 1

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, output_channels)
        self.attn = SelfAttn(512 // factor)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--encoder_layers', type=int, default=12)
        parser.add_argument('--data_path', type=str, default='/some/path')
        return parser

    def custom_forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x, _ = self.attn(x)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        
        x = self.up4(x, x1)
        x = self.outc(x)
        out_x = x - x.view(x.shape[0], -1).min(dim=1)[0].view(x.shape[0], 1, 1, 1)
        out_x = out_x / out_x.view(x.shape[0], -1).max(dim=1)[0].view(x.shape[0], 1, 1, 1)
        return out_x

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.custom_forward(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        img, annotation, fixation = batch['image'], batch['annotation'], batch['fixation']
        prediction = self.custom_forward(img)
        l = F.mse_loss(prediction, annotation)
        # Logging to TensorBoard by default
        self.log('train_loss', l)
        return l

    def validation_step(self, batch, batch_idx):
        img, annotation, fixation = batch['image'], batch['annotation'], batch['fixation']
        prediction = self.custom_forward(img)
        l = nss(prediction, fixation).sum()
        # Logging to TensorBoard by default
        self.log('val_acc', l, on_step=False, on_epoch=True, sync_dist=True)
        if batch_idx % 50 == 0:
            tensorboard = self.logger.experiment
            grid = make_grid(img)
            tensorboard.add_image('image', grid, global_step=self.global_step)
            grid = make_grid(annotation)
            tensorboard.add_image('gt', grid, global_step=self.global_step)
            grid = make_grid(prediction)
            tensorboard.add_image('pred', grid, global_step=self.global_step)
        return l
    
    def test_step(self, batch, batch_idx):
        img, annotation, fixation = batch['image'], batch['annotation'], batch['fixation']
        prediction = self.custom_forward(img)
        l = nss(prediction, fixation).mean()
        # Logging to TensorBoard by default
        self.log('test_acc', l, on_step=False, on_epoch=True)
        return l

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer