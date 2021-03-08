import os
from torchvision import transforms
import pytorch_lightning as pl

from model.UNetAttn import UNetAttn
from data.salicon import SALICONDataModule

from options import parser

# add model specific args
parser = UNetAttn.add_model_specific_args(parser)
args = parser.parse_args()

data_module = SALICONDataModule(args)
autoencoder = UNetAttn()
trainer = pl.Trainer.from_argparse_args(args)
trainer.fit(autoencoder, data_module)
trainer.test(datamodule=data_module)