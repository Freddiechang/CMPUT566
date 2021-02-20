from argparse import ArgumentParser
import pytorch_lightning as pl
parser = ArgumentParser()

# add PROGRAM level args
parser.add_argument('--conda_env', type=str, default='some_name')
parser.add_argument('--data_root', type=str, default='../dataset/salicon-api')
parser.add_argument('--normalize', action='store_true',
                    help='normalize the input')
parser.add_argument('--resize', action='store_true',
                    help='resize the input')
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=256)
# add all the available trainer options to argparse
# ie: now --gpus --num_nodes ... --fast_dev_run all work in the cli
parser = pl.Trainer.add_argparse_args(parser)