# OPS: Towards Open-World Segmentation of Parts (CVPR 2023)
[Tai-Yu Pan](https://tydpan.github.io/), [Qing Liu](https://qliu24.github.io/), [Wei-Lun Chao](https://sites.google.com/view/wei-lun-harry-chao/home), [Brian Price](https://www.brianpricephd.com/)

[[`BibTeX`](#CitingOPS)]

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

## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of OPS is licensed under a [MIT License](LICENSE).


## <a name="CitingOPS"></a>Citing OPS

If you use OPS in your research, please use the following BibTeX entry.

```BibTeX
@inproceedings{pan2023ops,
  title={Towards Open-World Segmentation of Parts},
  author={Tai-Yu Pan and Qing Liu and Wei-Lun Chao and Brian Price},
  journal={CVPR},
  year={2023}
}
```

However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).


## Acknowledgement

Code is largely based on Mask2Former (https://github.com/facebookresearch/Mask2Former).