# Pytorch Tutorial - Simple Object Detector

Repo for new students willing to learn Pytorch. It implements a simple object detector based on VGG16 with two braches -- probability of the object being in the image and its bounding box. Images must be of size 224x224x3.

<p align="center">
    <img src="https://raw.githubusercontent.com/albertpumarola/Pytorch-Tutorial-Object-Detector/master/imgs/readme_img.png">
</p>

### Prerequisites
To understand the repo I recommend first doing the official Pytorch [Beginner Tutorials](http://pytorch.org/tutorials/).

### Dependencies
Apart from PyTorch and Tensorboard:
```
pip install torchvision  # pretrained VGG16 parameters
pip install matplotlib   # for visualization
pip install tensorboardX # for visualization
```

### Train/Test Example
Train:
```
bash launch/run_train.sh
```
Test:
```
bash launch/run_test.sh
```

For any doubt contact `albert.pumarola.peris at gmail.com`
