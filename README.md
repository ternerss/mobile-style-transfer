# Mobile Style Transfer

Implementation of Neural Style Transfer based on MobileNet architecture with Sobel loss idea.

<p align = 'center'>
<img src = 'resources/src.jpg' height = '246px'>
<img src = 'resources/style.jpeg' height = '246px'>
<img src = 'resources/stylized.png' width = '300px'></a>
</p>
<p align = 'center'>
</p>

## Installation

1. Clone this repository:
```bash
git clone https://github.com/ternerss/mobile-style-transfer.git
cd mobile-style-transfer
```
2. install requirements:
```
pip install -r requirements.txt
```
## Usage

![pytorch](https://img.shields.io/badge/pytorch-v1.11.0-orange.svg?style=plastic)
![cuda](https://img.shields.io/badge/cuda-v11.3-green.svg?style=plastic)
![gcc](https://img.shields.io/badge/torchvision-v0.12.0-yellow.svg?style=plastic)

Dataset: [COCO 2017 Val images](http://images.cocodataset.org/zips/val2017.zip)

- Train:
 ```bash
python train.py --dataroot data/ --style data/styles/wave.jpeg --batch_size 16 --lr 0.001 --epochs 30
```

- Test:
```bash
python test.py --input data/val/building.jpeg --weights weights mobile_style_transfer.pth --out out/ 
```

## Main idea

<img src="resources/idea.png" width="784px"/> 

## References
* [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)
* [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/)
* [Instance Normalization](https://arxiv.org/abs/1607.08022)
* [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)
