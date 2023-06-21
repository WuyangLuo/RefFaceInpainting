### Reference-Guided Large-Scale Face Inpainting with Identity and Texture Control 
TCSVT 2023 [[Paper]](https://arxiv.org/pdf/2303.07014.pdf)

Face inpainting aims at plausibly predicting missing
pixels of face images within a corrupted region. Most existing
methods rely on generative models learning a face image distribution from a big dataset, which produces uncontrollable results,
especially with large-scale missing regions. To introduce strong
control for face inpainting, we propose a novel reference-guided
face inpainting method that fills the large-scale missing region
with identity and texture control guided by a reference face
image.

![RefFaceInpainting teaser](image/teaser.jpg)

### Requirements

- The code has been tested with PyTorch 1.10.1 and Python 3.7.11. We train our model with a NIVIDA RTX3090 GPU.

### Dataset Preparation
Download our dataset celebID from [BaiDuYun (password:5asv)](https://pan.baidu.com/s/1vbGJ1Gr3v71ulneSfQaN8Q) | [GoogleDrive](https://drive.google.com/file/d/1dIvKsW36j2D7AN2SBh-ZinF9X9iZoCon/view?usp=sharing) and set the relevant paths in `configs/config.yaml` and `test.py`

### Training
Train a model, run:
```
python train.py
```

### Testing

Download the pretrained model from [BaiDuYun (password:spwk)](https://pan.baidu.com/s/1RM2thrjKo_WbA972GTB1iA) | [GoogleDrive](https://drive.google.com/file/d/1qn1fKj-4iwykSZl_GT9kjz2UTnbMlU36/view?usp=sharing). Generate inpainted results guided by different reference images, run:

```
python test.py
```

### Citation:
If you use this code for your research, please cite our paper.
```
@article{luo2023reference,
  title={Reference-Guided Large-Scale Face Inpainting with Identity and Texture Control},
  author={Luo, Wuyang and Yang, Su and Zhang, Weishan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  publisher={IEEE}
}
```


### Acknowledgment
We use [zllrunning's model](https://github.com/zllrunning/face-parsing.PyTorch) to obtain face segmentation maps, [1adrianb's model](https://github.com/1adrianb/face-alignment) to align face and detect landmarks, [foamliu's model](https://github.com/foamliu/InsightFace-v2) to compute Arcface loss.
