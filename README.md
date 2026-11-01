# Vision Transformer

ViT (Vision Transformer) is a method that achieves state-of-the-art results for image classification.

## Installation

```console
git clone https://github.com/bareform/dependencies.git
conda update -n base -c defaults conda
conda env create -f dependencies/environment.yml
conda activate bareform

git clone https://github.com/bareform/vit.git
cd vit
```

<details>
  <summary> Dependencies (click to expand) </summary>
  
  ## Dependencies
  - Python 3.10
  - datasets
  - numpy
  - torch
  - torchvision
    
</details>

## Quick Start

To train a low-precision `ImageNet-1K` ViT:

```
python3 -m utils.trainer --config="./configs/imagenet-1k.toml"
```

Using the configuration provided at `./configs/imagenet-1k.toml`, we achieve mean top-1 accuracy of 60.57% with a standard deviation of 0.26% and a top-5 accuracy of 81.33% with a standard deviation of 0.17% on the validation set.

You can download the pre-trained models [here](https://huggingface.co/luethan2025/vit) and use the provided Jupyter Notebook `inference.ipynb` to visualize attention maps.

## Method

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby

Google

> While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

## Citation

The original paper can be found at:
```
@misc{dosovitskiy2020vit,
    title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
    author={Alexey Dosovitskiy and Lucas Beyer and Alexander Kolesnikov and Dirk Weissenborn and Xiaohua Zhai and Thomas Unterthiner and Mostafa Dehghani and Matthias Minderer and Georg Heigold and Sylvain Gelly and Jakob Uszkoreit and Neil Houlsby},
    year={2020},
    eprint={2010.11929},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
}
```
