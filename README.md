# DPatch: An Adversarial Patch Attack on Object Detectors
This is a [PyTorch](https://github.com/pytorch/pytorch)
implementation of DPATCH.
Please refer to the paper https://arxiv.org/abs/1806.02299



## Installation
1. Please install PyTorch following the instuctions on the official website. The version here should be 0.4.0 
   ```bash
    conda install pytorch torchvision -c pytorch
    ```

2. The yolo codes are referred to https://github.com/longcw/yolo2-pytorch. Download the pretrained yolo model [yolo-voc.weights.h5](https://drive.google.com/open?id=0B4pXCfnYmG1WUUdtRHNnLWdaMEU) 


3. Download the dataset VOCdevkit

    ```bash
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
    ```

2. Extract all of these tars into one directory named `VOCdevkit`

    ```bash
    tar xvf VOCtrainval_06-Nov-2007.tar
    tar xvf VOCtest_06-Nov-2007.tar
    tar xvf VOCdevkit_08-Jun-2007.tar
    ```

3. It should have this basic structure

    ```bash
    $VOCdevkit/                           # development kit
    $VOCdevkit/VOCcode/                   # VOC utility code
    $VOCdevkit/VOC2007                    # image sets, annotations, etc.
    # ... and several other directories ...
    ```
    
4. Since the program loading the data in `DPatch/data` by default,
you can set the data path as following.
    ```bash
    cd DPatch
    mkdir data
    cd data
    ln -s $VOCdevkit VOCdevkit2007
    ```


## Train a DPatch
The trained DPATCH are saved in trained_patch/${target_class}/ 
    ```bash
    python train.py
    ```
    
## Test
Download a trained DPATCH https://drive.google.com/open?id=1_G5xXWIJWNGuss4KZbBQ9pMvuthmL_wc , or you can test your trained DPatch and set the path in cfgs/config.py (Line 102) 
    ```bash
    python test.py –-attack untargeted
    python test.py –-attack targeted
    ```


## Demo
The demo pictures are saved in demo/patch, the detection results are saved in demo/
    ```bash
    python demo.py
    ```
    
Click the following picture to watch the video demo.
[![Watch the video](https://img.youtube.com/vi/-aPbU9q1gFU/maxresdefault.jpg)](https://youtu.be/-aPbU9q1gFU)
