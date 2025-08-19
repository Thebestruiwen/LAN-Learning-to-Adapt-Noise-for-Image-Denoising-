# LAN: Learning to Adapt Noise for Image Denoising
Changjin Kim, Tae Hyun Kim, Sungyong Baik

A sample code for our framework.

Paper link : [[CVPR]](https://openaccess.thecvf.com/content/CVPR2024/html/Kim_LAN_Learning_to_Adapt_Noise_for_Image_Denoising_CVPR_2024_paper.html)


> **Abstract:** *Removing noise from images a.k.a image denoising can be a very challenging task since the type and amount of noise can greatly vary for each image due to many factors including a camera model and capturing environments. While there have been striking improvements in image denoising with the emergence of advanced deep learning architectures and real-world datasets recent denoising networks struggle to maintain performance on images with noise that has not been seen during training. One typical approach to address the challenge would be to adapt a denoising network to new noise distribution. Instead in this work we shift our attention to the input noise itself for adaptation rather than adapting a network. Thus we keep a pretrained network frozen and adapt an input noise to capture the fine-grained deviations. As such we propose a new denoising algorithm dubbed Learning-to-Adapt-Noise (LAN) where a learnable noise offset is directly added to a given noisy image to bring a given input noise closer towards the noise distribution a denoising network is trained to handle. Consequently the proposed framework exhibits performance improvement on images with unseen noise displaying the potential of the proposed research direction.*

## Table of Contents
- [Overview](#overview)
- [Prepare Model and Dataset](#prepare-model-and-dataset)
- [Adaptation](#adaptation)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)


## Overview
<p align="center">
  <img src="./assets/overview.png" width="600"/>
</p>
<p align="center">
  <img src="./assets/framework.png" width="600"/>
</p>

## Prepare Model and Dataset
```bash
git clone https://github.com/chjinny/LAN.git
python prepare.py
```
- Dataset : [PolyU](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset)
- Model : [Restormer](https://github.com/swz30/Restorme)
  - Download the [pretrained weight file](https://drive.google.com/drive/folders/1Qwsjyny54RZWa7zC4Apg7exixLBo4uF0) and place it as ```./checkpoint/real_denoising.pth```.

## Adaptation
```bash
python main.py --method {lan, finetune} --self-loss {zsn2n, nbr2nbr}
```

## Citation
```bibtex
@inproceedings{kim2024lan,
  title={LAN: Learning to Adapt Noise for Image Denoising},
  author={Kim, Changjin and Kim, Tae Hyun and Baik, Sungyong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={25193--25202},
  year={2024}
}
```

## Acknowledgement

The codes are based on follows:
- [Restormer](https://github.com/swz30/Restormer)
- [Neighbor2Neighbor](https://github.com/TaoHuang2018/Neighbor2Neighbor)
- [Zero-Shot Noisr2Noise](https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing)

We thank the authors for sharing their codes.
