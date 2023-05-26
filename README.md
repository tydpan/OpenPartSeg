# OPS: Towards Open-World Segmentation of Parts (CVPR 2023)
[Tai-Yu Pan](https://tydpan.github.io/), [Qing Liu](https://qliu24.github.io/), [Wei-Lun Chao](https://sites.google.com/view/wei-lun-harry-chao/home), [Brian Price](https://www.brianpricephd.com/)

![](demo/demo.gif)

## Installation

See [Mask2Former](https://github.com/facebookresearch/Mask2Former).
Additaional packages for OPS:
```
pip install -e kmeans_pytorch
pip install -U scikit-image
```

Tested environment:
```
python==3.8.13
torch==1.13.0+cu116
torchaudio==0.13.0+cu116
torchvision==0.14.0+cu116
cudatoolkit==11.6.0

numpy==1.23.5
numba==0.56.3
scikit-image==0.20.0
```