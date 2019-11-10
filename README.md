# TASED-Net
[TASED-Net: Temporally-Aggregating Spatial Encoder-Decoder Network for Video Saliency Detection (ICCV 2019)](http://openaccess.thecvf.com/content_ICCV_2019/html/Min_TASED-Net_Temporally-Aggregating_Spatial_Encoder-Decoder_Network_for_Video_Saliency_Detection_ICCV_2019_paper.html)

## Overview
TASED-Net is a novel fully-convolutional network architecture for video saliency detection. The main idea is simple but effective: spatially decoding 3D video features while jointly aggregating all the temporal information. TASED-Net significantly outperforms previous state-of-the-art approaches on all three major large-scale datasets of video saliency detection: DHF1K, Hollywood2, and UCFSports. We observe that our model is especially better at attending to salient moving objects.

TASED-Net is currently leading the leaderboard of [DHF1K online benchmark](https://mmcheng.net/videosal/).

| Model | Year | &nbsp; NSS&#8593; &nbsp; | &nbsp; CC &#8593; &nbsp; | &nbsp; SIM&#8593; &nbsp; | AUC-J&#8593; | s-AUC&#8593; |
|:-------------|:--------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| [TASED-Net](http://openaccess.thecvf.com/content_ICCV_2019/html/Min_TASED-Net_Temporally-Aggregating_Spatial_Encoder-Decoder_Network_for_Video_Saliency_Detection_ICCV_2019_paper.html) &nbsp; | 2019 | 2.667 | 0.470 | 0.361 | 0.895 | 0.712 |
| [SalEMA](https://arxiv.org/abs/1907.01869) | 2019 | 2.574 | 0.449 | 0.466 | 0.890 | 0.667 |
| [STRA-Net](https://www.ncbi.nlm.nih.gov/pubmed/31449021) | 2019 | 2.558 | 0.458 | 0.355 | 0.895 | 0.663 |
| [ACLNet](https://arxiv.org/abs/1801.07424) | 2018 | 2.354 | 0.434 | 0.315 | 0.890 | 0.601 |
| [SalGAN](https://arxiv.org/abs/1701.01081) | 2017 | 2.043 | 0.370 | 0.262 | 0.866 | 0.709 |
| SALICON | 2015 | 1.901 | 0.327 | 0.232 | 0.857 | 0.590 |
| GBVS | 2007 | 1.474 | 0.283 | 0.186 | 0.828 | 0.554 |

## Video Saliency Detection
Video saliency detection aims to model the gaze fixation patterns of humans when viewing a dynamic scene. Because the predicted saliency map can be used to prioritize the video information across space and time, this task has a number of applications such as video surveillance, video captioning, video compression, etc.

## Examples
We compare our TASED-Net to [ACLNet](https://arxiv.org/abs/1801.07424), which was the previously leading state-of-the-art method. As shown in the examples below, TASED-Net is better at attending to the salient information. We also would like to point out that TASED-Net has a much smaller network size (82 MB v.s. 252 MB).

![](example/comparison1.gif)

![](example/comparison2.gif)

## Code Usage
First, clone this repository and download [this weight
file](https://drive.google.com/uc?export=download&id=11DLJkuhRHHdRziYc2dQBiyPzf6QGn041).
Then, just run the code using

`$ python run_example.py`

This will generate frame-wise saliency maps.
You can also specify the input and output directories as command-line arguments. For example,

`$ python run_example.py ./example ./output`

## Notes
- We observed that there is a trade-off between AUC scores and the others. The released model is a modified version to increase AUC-J and s-AUC scores (NSS, CC, and SIM scores are slightly lower than the reported figures). This version achieves 0.898 AUC-J and 0.716 s-AUC on the test set of DHF1K (2.621 NSS, 0.466 CC, 0.336 SIM).

- We recommend using PNG image files as input (although examples of this repository are in JPEG format).

- For the encoder of TASED-Net, we use the [S3D network](https://arxiv.org/abs/1712.04851). We pretrained S3D on Kinetics-400 dataset using PyTorch and it achieves 72.08% top1 accuracy (top5: 90.35%) on the validation set of the dataset. We release [our weight file for S3D](https://github.com/kylemin/S3D.git) together this project. If you find it useful, you might want to consider citing our work.

- For training, we recommend using [ViP](https://github.com/MichiganCOG/ViP.git), which is the video platform for general purposes in PyTorch. Otherwise, you can use `run_train.py`. Before running the training code, make sure to download [our weight file for S3D](https://github.com/kylemin/S3D.git).


## Citation
```
@inproceedings{min2019tased,
  title={TASED-Net: Temporally-Aggregating Spatial Encoder-Decoder Network for Video Saliency Detection},
  author={Min, Kyle and Corso, Jason J},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={2394--2403},
  year={2019}
}
```
