# PrecNeRF
This is the official repository of the paper "Precise Integral in NeRFs: Overcoming the Approximation Errors of Numerical Quadrature" published in WACV2025.

##  Environment Setup
We use [nerfstudio](https://github.com/nerfstudio-project/nerfstudio) as our framework to implement PrecNeRF. First setup the conda environment:
```
conda create --name nerfstudio -y python=3.8
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
export TCNN_CUDA_ARCHITECTURES=86
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
Then install nerfstudio and PrecNeRF as:
```
git clone https://github.com/Moreland-cas/PrecNeRF
conda activate nerfstudio
cd PrecNeRF/nerfstudio
pip install -e .
cd ../SegNeRF
pip install -e .
```
##  Data Preparation
We use three datasets (nerf_synthetic, Synthetic_NSVF, TanksAndTemple) to train and evaluate PrecNeRF. Please refer to the original website and repository ([NeRF Official Google Drive](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1), [NSVF](https://github.com/facebookresearch/NSVF)) to download the data and organize the structure as:
```
/PrecNeRF
    /data
        /nerf_synthetic
        /NSVF
            /Synthetic_NSVF
            /TanksAndTemple
    /SegNeRF
    /nerfstudio
```

## Training
We use the `ns-train` command provided by nerfstudio to train NeRF and PrecNeRF.
### nerf_synthetic
```
ns-train vanilla-nerf --output-dir path_to_outputs --data path_to_data/nerf_synthetic/xxx
ns-train segnerf --output-dir path_to_outputs  --data path_to_data/nerf_synthetic/xxx
```
### Synthetic_NSVF
```
ns-train vanilla-nerf --output-dir path_to_outputs --data  path_to_data/NSVF/Synthetic_NSVF/xxx
ns-train segnerf nsvf --output-dir path_to_outputs --data path_to_data/NSVF/Synthetic_NSVF/xxx
```
### TanksAndTemple
```
ns-train vanilla-nerf --output-dir path_to_outputs --data  path_to_data/NSVF/TanksAndTemple/xxx
ns-train segnerf --output-dir path_to_outputs --data path_to_data/NSVF/TanksAndTemple/xxx
```
Note: When training on nerf_synthetic, the config used for `dataparser` should be `BlenderDataParserConfig()`, whereas `NsvfDataParserConfig()` for Synthetic_NSVF and TanksAndTemple.

## Evaluation
We use the `ns-eval` command to evaluate the trained models.
```
ns-eval --load_config path_to_outputs/your_model/configs.yaml --output_path path_to_dump
```

## Citation
If you find [PrecNeRF](https://github.com/Moreland-cas/PrecNeRF) useful in your research works, please consider citing:
```
@inproceedings{zhang2025precnerf,
  title={Precise Integral in NeRFs: Overcoming the Approximation Errors of Numerical Quadrature},
  author={Zhang, Boyuan and He, Zhenliang and Kan, Meina and Shan, Shiguang},
  booktitle={IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year={2025}
}
```
