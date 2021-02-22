import os
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from model.LitAutoEncoder import LitAutoEncoder

from options import parser

# add model specific args
parser = LitAutoEncoder.add_model_specific_args(parser)
args = parser.parse_args()

dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)
autoencoder = LitAutoEncoder()
trainer = pl.Trainer.from_argparse_args(args)
trainer = pl.Trainer()
trainer.fit(autoencoder, train_loader)