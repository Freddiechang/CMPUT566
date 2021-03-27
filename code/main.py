import os
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model.UNetAttn import UNetAttn
from data.salicon import SALICONDataModule

from options import parser
# model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath='saved_models',
    filename='model-{epoch:03d}-{val_acc:.2f}',
    save_top_k=10,
    mode='max',
    save_weights_only=True,
    period=1
)
# add model specific args
parser = UNetAttn.add_model_specific_args(parser)
args = parser.parse_args()
data_module = SALICONDataModule(args)
autoencoder = UNetAttn()
trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback])
trainer.fit(autoencoder, data_module)
trainer.test(datamodule=data_module)