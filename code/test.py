import pytorch_lightning as pl

from model.UNetAttn import UNetAttn
from data.salicon import SALICONDataModule, SALICON
from data.cat2000 import CAT2000DataModule, CAT2000
from options import parser


saved_model_path = "comparison/mse_attn_1_lr-2/model-epoch=196-val_acc=390.18.ckpt"
parser = UNetAttn.add_model_specific_args(parser)
args = parser.parse_args()
if "salicon" in args.data_root:
    data_module = SALICONDataModule(args)
elif "CAT" in args.data_root:
    data_module = CAT2000DataModule(args)
model = UNetAttn()
model = model.load_from_checkpoint(saved_model_path)
trainer = pl.Trainer.from_argparse_args(args)
results = trainer.test(model, datamodule=data_module)
print(results)


# calculate dataset average

#from loss.loss import nss, cc
# import torch
# from tqdm import tqdm

# args = parser.parse_args()
#if "salicon" in args.data_root:
#    data_module = SALICON(args)
#elif "CAT" in args.data_root:
#    data_module = CAT2000(args)
# result = torch.zeros(1)
# for i in tqdm(data_module):
#     prediction = i['annotation'].unsqueeze(0)
#     fixation = i['fixation'].unsqueeze(0)
#     result += nss(prediction, fixation).squeeze(0)
# print(result, result/len(data_module))

# dataset average nss 1.5691
