from model.UNetAttn import UNetAttn
from data.salicon import SALICON
from interpretability.ig_saliency import integrated_gradients
from options import parser

saved_model_path = "comparison/mse_attn/model-epoch=196-val_acc=390.18.ckpt"

parser = UNetAttn.add_model_specific_args(parser)
args = parser.parse_args()
# change this to use other datasets
data_module = SALICON(args, mode="test")

# need to modify UnetAttn.py: custom_forward() to disable attn
model = UNetAttn()
model = model.load_from_checkpoint(saved_model_path)

# index can be 0 - 1499
index = 100
x = data_module[index]
result = integrated_gradients(
    x["image"].unsqueeze(0).numpy(), 
    model, 
    baseline=None, 
    steps=20, 
    gpu_device="cuda:0")


