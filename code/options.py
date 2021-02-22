from argparse import ArgumentParser
import pytorch_lightning as pl
parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--data_root', type=str, default='../dataset/salicon-api')
parser.add_argument('--normalize', default=False, action='store_true',
                    help='normalize the input')
parser.add_argument('--norm_mean', type=str, default="0.5+0.5+0.5")
parser.add_argument('--norm_std', type=str, default="1.0+1.0+1.0")
parser.add_argument('--resize', action='store_true',
                    help='resize the input')
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--dataloader_workers', type=int, default=4)
parser.add_argument('--shuffle', default=False, action='store_true')
parser.add_argument('--prefetch', type=int, default=2)
# add all the available trainer options to argparse
# ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
parser = pl.Trainer.add_argparse_args(parser)