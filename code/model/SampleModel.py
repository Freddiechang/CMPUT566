import torch
from torch import nn
import torch.nn.functional as F
from argparse import ArgumentParser
import pytorch_lightning as pl

from loss.loss import nss, cc

class SampleModel(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1)
        )


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--encoder_layers', type=int, default=12)
        parser.add_argument('--data_path', type=str, default='/some/path')
        return parser

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        result = self.net(x)
        return result

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        img, annotation, fixation = batch['image'], batch['annotation'], batch['fixation']
        prediction = self.net(img)
        l = -1 * nss(prediction, fixation).sum()
        # Logging to TensorBoard by default
        self.log('train_loss', l)
        return l

    def validation_step(self, batch, batch_idx):
        img, annotation, fixation = batch['image'], batch['annotation'], batch['fixation']
        prediction = self.net(img)
        l = -1 * nss(prediction, fixation).sum()
        # Logging to TensorBoard by default
        self.log('val_loss', l)
        return l
    
    def test_step(self, batch, batch_idx):
        img, annotation, fixation = batch['image'], batch['annotation'], batch['fixation']
        prediction = self.net(img)
        l = -1 * nss(prediction, fixation).sum()
        # Logging to TensorBoard by default
        self.log('test_loss', l)
        return l

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer