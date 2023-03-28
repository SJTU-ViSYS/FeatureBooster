# FeatureBooster Weights File and Demo Script

## Introduction

FeatureBooster is a research project with Midea Corporate Research Center. This repo contains the PyTorch demo code and pretrained weights for FeatureBooster. The FeatureBooster network reuse existing local feature descriptors. It takes the original descriptors and the geometric properties of keypoints as the input, and uses an MLP-based self-boosting stage and a Transformer-based cross-boosting stage to enhance the descriptors. For more details, please see:

* Full paper PDF: [FeatureBooster: Boosting Feature Descriptors with a Lightweight Neural Network](https://arxiv.org/abs/2211.15069)

* Authors: *Xinjiang Wang, Zeyu Liu, Yu Hu, Wei Xi, Wenxian Yu, Danping Zou*

<p align="center">
  <img src="assets/FeatureBooster.png" width="800">
</p>


## Prerequisites

**Step1**: Cloning the repository and creating a virtual environment
```bash
git clone --recursive https://github.com/SJTU-ViSYS/FeatureBooster.git
cd FeatureBooster/
conda env create -f environment.yml
conda activate featurebooster
```
**Step2**: Installing pyCOLMAP for SIFT extractor. Please following the instruction [here](https://github.com/colmap/pycolmap#building-from-source). 

**Step3**: Build the ORBSLAM2 features
```bash
cd extractors/orbslam2_features/
mkdir build
cd build
cmake -DPYTHON_LIBRARY=~/anaconda3/envs/featurebooster/lib/libpython3.8.so \
      -DPYTHON_INCLUDE_DIR=~/anaconda3/envs/featurebooster/include/python3.8 \
      -DPYTHON_EXECUTABLE=~/anaconda3/envs/featurebooster/bin/python3.8 ..
make -j
```

## Models

The trained weights of FeatureBoosters for different descriptors are provided in `models/`. Boost-F and Boost-B indicate real-valued boosted and binary boosted descriptors, respectively. At present time, the off-the-shelf weights for following feature extractors are provided:
* [ORB](https://github.com/raulmur/ORB_SLAM2) (ORB extractor of ORB-SLAM2)
* [SIFT](https://github.com/colmap/pycolmap) (SIFT extractor of COLMAP)
* [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork)
* [ALIKE](https://github.com/Shiaoming/ALIKE) (ALIKE-L)

## Feature extraction

`extract_features.py` can be used to extract various local features for a given list of images. Currently, the following local features are supported:
* ORB, ORB+Boost-B
* SIFT, SIFT+Boost-F, SIFT+Boost-B
* SuperPoint, SuperPoint+Boost-F, SuperPoint+Boost-B
* ALIKE(-L), ALIKE+Boost-F, ALIKE+Boost-B
* RootSIFT
* SOSNet
* HardNet

*Note: The extraction of SOSNet and HardNet are based on [Kornia](https://github.com/kornia/kornia).*

The output format is [`npz`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) and the output feature files contain two arrays: 

- `keypoints` [`N x M`] array containing the positions of keypoints `x, y` and other geometric properties, such as the scales, the detection score, the oriention.
- `descriptors` [`N x D`] array containing the descriptors. For real-valued descriptors, the data type of elements in this array is `np.float32`. For binary descriptor, the type is `np.uint8`.

ORB+Boost-B features for HPatches dataset can be extracted by running:

```bash
python extract_features.py --descriptor ORB+Boost-B --image_list_file image_list_hpatches_sequences.txt
```

## Evaluation on Hpatches
**Step1**: Downloading the [HPatches dataset](https://github.com/hpatches/hpatches-dataset) 
```bash
cd hpatches_sequences
bash download.sh
```
**Step2**: Extracting the features following the instruction in [Feature extraction](#feature-extraction).

**Step3**: Running the notebook `hpatches_sequences/HPatches-Sequences-Matching-Benchmark.ipynb`. The new methods can be added in cell 4 of the notebook, while the features are supposed to be stored in the [`npz`](https://docs.scipy.org/doc/numpy/reference/generated/numpy.savez.html) format following the description on [Feature extraction](#feature-extraction).

## BibTex Citation

```bibtex
@inproceedings{wang2022featurebooster,
  title={FeatureBooster: Boosting Feature Descriptors with a Lightweight Neural Network},
  author={Wang, Xinjiang and Liu, Zeyu and Hu, yu and Xi, Wei and Yu, Wenxian and Zou, Danping},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```

## Acknowledgement
We borrowed a lot of codes from [D2-Net](https://github.com/mihaidusmanu/d2-net). Thanks for [Mihai Dusmanu](https://github.com/mihaidusmanu)'s excellent works!