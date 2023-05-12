# Two-way Multi-Label Loss

The Pytorch implementation for the CVPR2023 paper (highlight) of "[Two-way Multi-Label Loss](https://staff.aist.go.jp/takumi.kobayashi/publication/2023/CVPR2023.pdf)" by [Takumi Kobayashi](https://staff.aist.go.jp/takumi.kobayashi/).

### Citation

If you find our project useful in your research, please cite it as follows:

```
@inproceedings{kobayashi2023cvpr,
  title={Two-way Multi-Label Loss},
  author={Takumi Kobayashi},
  booktitle={Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2023}
}
```

## Introduction

A natural image frequently contains multiple classification targets.
The **multi-label** classification has been tackled mainly in a binary cross-entropy (BCE) framework, disregarding softmax loss which is a standard loss in single-label classification.
This work proposes a multi-label loss by bridging a gap between the softmax loss and the multi-label scenario. 
The proposed loss function is formulated on the basis of relative comparison among classes which also enables us to further improve discriminative power of features by enhancing classification margin. 
The loss function is so flexible as to be applicable to a multi-label setting in two ways for discriminating classes as well as samples.
For more detail, please refer to our [paper](https://staff.aist.go.jp/takumi.kobayashi/publication/2023/CVPR2023.pdf).

<img width=400 src="https://user-images.githubusercontent.com/53114307/227201136-9487b198-ed3a-4099-8dcd-ee4677ccffef.png">

## Usage

The proposed loss for multi-label annotation is implemented by `TwoWayLoss` class in `utils/criterion.py`.

### Training
For example, ResNet-50 is trained with our loss on MSCOCO dataset by distributed training over 4 GPUs;
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  ./dataset/MSCOCO --arch resnet50  --dataset MSCOCO --output ./result/ --distributed
```

Note that MSCOCO dataset must be downloaded at `./datasets/MSCOCO/` before training.


## Results

#### MSCOCO

| Method  | mAP@class | mAP@sample |
|---|---|---|
| Softmax | 58.00   | 83.60 | 
| BCE | 67.71 | 79.65|
| Ours| 74.11   |  86.66 |


## Contact
takumi.kobayashi (At) aist.go.jp