## Visual Saliency

Visual saliency is the extent of attraction of objects to our visual system. The study of visual saliency can help us to have a deeper understanding of the human visual system.  Researching in this area can lead to many applications, such as image/video segmentation, image/video compression, foreground annotation, perceptual quality assessment, etc.

![salicon_data_sample](https://github.com/Freddiechang/CMPUT566/blob/main/documents/proposal/imgs/example1_original.jpg)
![salicon_data_sample](https://github.com/Freddiechang/CMPUT566/blob/main/documents/proposal/imgs/example1_saliency.jpg)

### Methodology

We choose CNN to predict the saliency maps. The filters in CNNs function as feature extractors, and as the depth of the convolutional layer go up, the features tend to be more sophisticated and high level.
We use UNet as the backbone of our model. We use UNet as the backbone of our model. UNet has a symmetric expanding path made of several skip connections that enables precise localization. This feature can help assign correct visual saliency values to the corresponding locations. As for the attention mechanism, we implement it as a Pytorch module and then make this module part of UNet's data path. The goal of this module is to find long-distance dependencies across different regions in the image.

* **Disclaimer**: This project has been implemented as the CMPUT 566, University of Alberta course assignment and may not be maintained/contributed actively.

### How to Run

Requirments:

* Python +3.6
* pytorch +1.81
* torchvision +0.9.1
* pytorch-lightning +1.2.5
* tqdm +4.59.0
* scipy +1.6.2

Run:

The `main.py` script is the entry point of the project. This script fits a model and tests it based on the input options. Some of the available options are:

* `--data_root`: this parameter sets the directory of a dataset for test and train. Each dataset has a *DataModule* under the `data/` directory, which demonstrates the dataset structure.
* `--batch_size`: indicates the batch size for trainer.

You can see all available options using `python3 main.py -h` command.

To run the project with a different dataset, you need to change` --data_root` by overriding it on the `options.py` class or passing the value as an argument. Also, you need to change the *DataModule* on the `main.py` script.

Example:

```python
python3 main.py --gpus 2 --accelerator ddp  --max_epochs 200 --resize --log_every_n_steps 30
```



### Project Structure

* `/code`: This directory contains implementation of the model and *DataModule*
  * `/data`: contains *DataModule* releated to each dataset.
  * `/interpretability`: contains saliency integrated gradients for interpretability
  * `/loss`: contains script for calculating cost functions
  * `/model`: contains the Unet and attention layers parts 
* `/document/proposal`: Latex files of the final proposal
* `/dataset`: contains multiple directory each of them coresponding to one dataset(**NOT AVAILABLE IN THE REPOSITPRY**)
  * `/figrim`: this directory, as an example,contains the [FIGRIM](http://figrim.mit.edu/index_eyetracking.html) dataset. you can see `/code/data/figrim.py` as an exmple for implementing a *DataModule*.
    * `/FIXATIONLOCS` : fixation locations
    * `/FIXATIONMAP`: fixation maps
    * `/Targets`: Images 

### Links

* You can find related datasets [here](http://saliency.mit.edu/datasets.html).
* Our final proposal is available at this  [link](https://github.com/Freddiechang/CMPUT566/blob/main/documents/proposal/proposal.tex).
* We used **Pytorch** libary and **PyTorch Lightning** framework in our implementation.



### Contributors

* Sijie Ling
* Shupei Zhang
* Mehdi Akbarian Rastaghi

  
