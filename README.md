# Multi-Basis Input Convolutional Neural Network 
# (MBI-CNN)
This version of the tensorflow CIFAR-10 code is modified to train multiple models using multiple bases of the same input data.  Inferencing is accomplished with late fusion by adding the output of the softmax.



## Resources


## Installing / Getting started

A minimal setup you need to get running.

### Ubuntu 14.04
_NOTE: This requires CUDA and cuDNN for GPU support_

#### Get necessary packages
```shell
sudo apt-get update
sudo apt-get install git python3 python3-pip
sudo -H pip3 install virutualenvwrapper
```
#### CUDA 8.0
Go to <https://developer.nvidia.com/cuda-downloads> and download cuda-repo-ubuntu1404-8-0-local_8.0.44-1_amd64.deb

```shell
sudo dpkg -i cuda-repo-ubuntu1404-8-0-local_8.0.44-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda
```
#### cuDNN v5.1
Go to <https://developer.nvidia.com/cudnn> and download cudnn-8.0-linux-x64-v5.1-ga.tgz

```shell
cd ~/<where cudnn tgz file is>/
tar xvzf cudnn-8.0-linux-x64-v5.1-ga.tgz
sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
sudo apt-get install libcupti-dev
```

#### Set up the environment
```shell
mkdir ~/projects && cd ~/projects
git clone https://github.gatech.edu/kds17/mbi.git
mkvirtualenv mbi
echo "proj_name=\$(basename \$VIRTUAL_ENV) 
cd ~/projects/\$proj_name" >> ~/.venv/postactivate
deactivate
workon mbi
pip install -r requirements.txt
python
```

#### Test the environment
```python
>>> import cv2
>>> import tensorflow as tf
```
If you did not receive any errors while importing the packages then you should be good to go!

## Running the code

### Training
```shell
python mbi_train.py
```
This will begin training RGB, FFT, HSV, and DCT basis models for 2000 steps.  The folder /tmp/mbi_experiment/ will be created and will contain all checkpoints and useful visualizations for each model. 

If you want to create the folders elsewhere or change some other parameter you can set the flags:
```shell
python mbi_train --data_dir='/some/where/' --fft_dir='/some/where/else' --max_steps=100
```

### Evaluation
```shell
python mbi_eval.py
```

### Visualization
#### RGB - Convolution Layer #1
![RGB Conv1 Layer](/examples/rgb_conv1.gif?raw=true "RGB Conv1 Layer")
#### FFT - Convolution Layer #1
![FFT Conv1 Layer](/examples/fft_conv1.gif?raw=true "FFT Conv1 Layer")
#### DCT - Convolution Layer #1
![DCT Conv1 Layer](/examples/dct_conv1.gif?raw=true "DCT Conv1 Layer")
#### HSV - Convolution Layer #1
![HSV Conv1 Layer](/examples/hsv_conv1.gif?raw=true "HSV Conv1 Layer")

## Licensing

The code in this project is licensed under Apache license.
