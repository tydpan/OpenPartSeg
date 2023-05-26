# OPS: Towards Open-World Segmentation of Parts (CVPR 2023)
[Tai-Yu Pan](https://tydpan.github.io/), [Qing Liu](https://qliu24.github.io/), [Wei-Lun Chao](https://sites.google.com/view/wei-lun-harry-chao/home), [Brian Price](https://www.brianpricephd.com/)

[[`BibTeX`](#citing-ops)]

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

## Demo

### Option 1: Streamlit
Interacitve mode to draw masks. Need to install streamlit
```
pip install streamlit streamlit-drawable-canvas
```
run
```
streamlit run predict_part_web.py CKPT
```

### Option 2: 
run
```
python predict_part.py CKPT --input IMG_RGBA_1 IMG_RGBA_2
```
Output images are located at demo/outputs if not specified.


## Training
```
python train_net_part.py --config-file configs/part_segmentation/SETTING --num-gpus N
```
For more command line options, please see [Mask2Former](https://github.com/facebookresearch/Mask2Former).


## License

Shield: [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The majority of OPS is licensed under a [MIT License](LICENSE).

However portions of the project are available under separate license terms: Swin-Transformer-Semantic-Segmentation is licensed under the [MIT license](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).


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


## Acknowledgement

Code is largely based on Mask2Former (https://github.com/facebookresearch/Mask2Former).