import os
from torchvision import transforms
import pytorch_lightning as pl

from model.SampleModel import SampleModel
from data.salicon import SALICONDataModule

from options import parser

# add model specific args
parser = SampleModel.add_model_specific_args(parser)
args = parser.parse_args()

data_module = SALICONDataModule(args)
autoencoder = SampleModel()
trainer = pl.Trainer.from_argparse_args(args)
trainer = pl.Trainer()
trainer.fit(autoencoder, data_module)
trainer.test(datamodule=data_module)