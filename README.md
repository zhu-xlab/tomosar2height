# TomoSAR2Height

This repository provides the code to train and evaluate **TomoSAR2Height**, a method for reconstructing building heights (nDSMs) from TomoSAR point clouds.

## Installation

* Clone the repository
```bash
git clone git@github.com:chenzhaiyu/tomosar2height.git
cd tomosar2height
```

* Create a conda environment and install dependencies
```bash
conda create --name tomosar2height python=3.11 mamba
conda activate tomosar2height
mamba install pytorch torchvision pytorch-cuda=11.7 pytorch-scatter open3d affine laspy matplotlib Pillow plyfile PyYAML rasterio scikit-learn scipy tabulate tqdm transformations trimesh urllib3 wandb shapely hydra-core omegaconf -c pyg -c pytorch -c nvidia -c open3d-admin -c conda-forge
```

## Usage

### Data preparation

```bash
python scripts/build_dataset.py dataset=munich
```

### Training

Train TomoSAR2Height using different data modalities:

* Using point clouds only
```bash
python train.py dataset=munich use_cloud=true use_image=false use_footprint=false wandb=true run_suffix='_suffix' training.max_iteration=10000 gpu_id=0
```

* Using point clouds and images
```bash
python train.py dataset=munich use_cloud=true use_image=true use_footprint=false wandb=true run_suffix='_suffix' training.max_iteration=10000 gpu_id=0
```

* Using point clouds, images, and footprint supervision
```bash
python train.py dataset=munich use_cloud=true use_image=true use_footprint=true wandb=true run_suffix='_suffix' training.max_iteration=10000 gpu_id=0
```

### Evaluation

Evaluate a trained TomoSAR2Height model with:
```bash
python test.py dataset=munich use_cloud=true use_image=false use_footprint=true run_suffix='_suffix' gpu_id=0
```
