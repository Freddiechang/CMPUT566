from model.UNetAttn import UNetAttn
from model.UNetAttn import UNetNoAttn
from data.CAT2000 import CAT2000
from interpretability.ig_saliency import integrated_gradients
from options import parser
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# saved_model_path = "CMPUT566-main/code/comparison/mse_attn/model-epoch=196-val_acc=390.18.ckpt"
# saved_model_path = "CMPUT566-main/code/comparison/mse_attn/cc_attn-epoch=196-val_acc=395.79.ckpt"
saved_model_path_1 = "CMPUT566-main/code/comparison/mse_attn/mse_attn-epoch=196-val_acc=390.18.ckpt"
saved_model_path_2 = "CMPUT566-main/code/comparison/mse_attn/mse_noattn-epoch=194-val_acc=397.98.ckpt"
# saved_model_path = "CMPUT566-main/code/comparison/mse_attn/mse_noattn-epoch=194-val_acc=397.98.ckpt"

parser = UNetAttn.add_model_specific_args(parser)
args = parser.parse_args(args=[])
# change this to use other datasets
data_module = CAT2000(args)

# need to modify UnetAttn.py: custom_forward() to disable attn
model_1 = UNetAttn()
model_1 = model_1.load_from_checkpoint(saved_model_path_1)

model_2 = UNetNoAttn()
model_2 = model_2.load_from_checkpoint(saved_model_path_2)


print("done")

# index can be 0 - 1999
for i in range(100):
    index = i
    print(index)
    steps_set = 50
    threshold_set = 0.5
    x = data_module[index]

    result, outputs = integrated_gradients(
        x["image"].unsqueeze(0).numpy(),
        model_2,
        baseline=None,
        steps=steps_set,
        threshold=threshold_set,
        gpu_device="cuda:0")

    image_sample = x['image'].squeeze().detach().numpy()
    image_sample_show = np.transpose(image_sample, (1, 2, 0))
    fixation_sample = x['fixation'].squeeze().detach().numpy()
    annotation_sample = x['annotation'].squeeze().detach().numpy()

    output_sample = outputs[steps_set - 1].squeeze()
    output_select = output_sample > threshold_set

    image_select = image_sample * output_select
    image_select_show = np.transpose(image_select, (1, 2, 0))

    attribution_map = np.average(result.squeeze(), axis=0)
    attribution_max = np.max(np.abs(attribution_map))
    attribution_map = attribution_map / attribution_max

    plt.ioff()
    plt.figure(figsize=(30, 20))
    plt.subplot(2, 3, 1), plt.title('Image', fontsize='xx-large')
    plt.imshow(image_sample_show)
    plt.subplot(2, 3, 2), plt.title('Correct Fixation', fontsize='xx-large')
    plt.imshow(1 - fixation_sample, "binary")
    plt.subplot(2, 3, 3), plt.title("Correct Saliency", fontsize='xx-large')
    plt.imshow(1 - annotation_sample, "binary")
    plt.subplot(2, 3, 4), plt.title("Predicted Saliency", fontsize='xx-large')
    plt.imshow(1 - output_sample, "binary")
    plt.subplot(2, 3, 5), plt.title("Predicted Saliency > threshold", fontsize='xx-large')
    plt.imshow(image_select_show)
    plt.subplot(2, 3, 6), plt.title("Attribution", fontsize='xx-large')
    plt.imshow(attribution_map, "bwr", vmax=1, vmin=-1)
    plt.savefig("result/1_without/" + str(index) + ".png")
    plt.close()



# For the figure shown
index = 1570
print(index)
steps_set = 50
threshold_set = 0.5
x = data_module[index]

result_1, outputs_1 = integrated_gradients(
    x["image"].unsqueeze(0).numpy(),
    model_1,
    baseline=None,
    steps=steps_set,
    threshold=threshold_set,
    gpu_device="cuda:0")

result_2, outputs_2 = integrated_gradients(
    x["image"].unsqueeze(0).numpy(),
    model_2,
    baseline=None,
    steps=steps_set,
    threshold=threshold_set,
    gpu_device="cuda:0")

'''
index_ig = 0
output_sample_1 = outputs_1[index_ig].squeeze()
output_sample_2 = outputs_2[index_ig].squeeze()
output_sample_3 = outputs_1[steps_set - 1].squeeze()
output_sample_4 = outputs_2[steps_set - 1].squeeze()
plt.figure()
plt.subplot(2, 2, 1), plt.title('A_dim', fontsize='xx-large')
plt.imshow(1 - output_sample_1, "binary")
plt.subplot(2, 2, 2), plt.title('B_dim', fontsize='xx-large')
plt.imshow(1 - output_sample_2, "binary")
plt.subplot(2, 2, 3), plt.title('A', fontsize='xx-large')
plt.imshow(1 - output_sample_3, "binary")
plt.subplot(2, 2, 4), plt.title('B', fontsize='xx-large')
plt.imshow(1 - output_sample_4, "binary")
'''

image_sample = x['image'].squeeze().detach().numpy()
image_sample_show = np.transpose(image_sample, (1, 2, 0))
fixation_sample = x['fixation'].squeeze().detach().numpy()
annotation_sample = x['annotation'].squeeze().detach().numpy()

output_sample_1 = outputs_1[steps_set - 1].squeeze()
output_select_1 = output_sample_1 > threshold_set

image_select_1 = image_sample * output_select_1
image_select_show_1 = np.transpose(image_select_1, (1, 2, 0))

attribution_map_1 = np.average(result_1.squeeze(), axis=0)
attribution_max_1 = np.max(np.abs(attribution_map_1))
attribution_map_1 = attribution_map_1 / attribution_max_1

output_sample_2 = outputs_2[steps_set - 1].squeeze()
output_select_2 = output_sample_2 > threshold_set

image_select_2 = image_sample * output_select_2
image_select_show_2 = np.transpose(image_select_2, (1, 2, 0))

attribution_map_2 = np.average(result_2.squeeze(), axis=0)
attribution_max_2 = np.max(np.abs(attribution_map_2))
attribution_map_2 = attribution_map_2 / attribution_max_2

plt.ioff()
plt.figure(figsize=(20, 20))
plt.subplot(3, 3, 1), plt.title('Image', fontsize='xx-large')
plt.imshow(image_sample_show)
plt.subplot(3, 3, 2), plt.title('Correct Fixation', fontsize='xx-large')
plt.imshow(1 - fixation_sample, "binary")
plt.subplot(3, 3, 3), plt.title("Correct Saliency", fontsize='xx-large')
plt.imshow(1 - annotation_sample, "binary")
plt.subplot(3, 3, 4), plt.title("Predicted Saliency (With Attention)", fontsize='xx-large')
plt.imshow(1 - output_sample_1, "binary")
plt.subplot(3, 3, 5), plt.title("Predicted Saliency > threshold (With Attention)", fontsize='xx-large')
plt.imshow(image_select_show_1)
plt.subplot(3, 3, 6), plt.title("Attribution (With Attention)", fontsize='xx-large')
plt.imshow(attribution_map_1, "bwr", vmax=1, vmin=-1)
plt.subplot(3, 3, 7), plt.title("Predicted Saliency (Without Attention)", fontsize='xx-large')
plt.imshow(1 - output_sample_2, "binary")
plt.subplot(3, 3, 8), plt.title("Predicted Saliency > threshold (Without Attention)", fontsize='xx-large')
plt.imshow(image_select_show_2)
plt.subplot(3, 3, 9), plt.title("Attribution (Without Attention)", fontsize='xx-large')
plt.imshow(attribution_map_2, "bwr", vmax=1, vmin=-1)
plt.savefig("result/used_2.png")