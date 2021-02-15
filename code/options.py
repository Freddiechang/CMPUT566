from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--gpus', default=None)
parser.add_argument('--min_epochs', default=1)
parser.add_argument('--max_epochs', default=1000)
args = parser.parse_args()