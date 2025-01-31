# TomoSAR2Height

This repository provides the code for training and evaluating **TomoSAR2Height**, a method for reconstructing building heights (nDSMs) from spaceborne TomoSAR point clouds.

## ⚙️ Installation

* Clone the repository
```bash
git clone git@github.com:chenzhaiyu/tomosar2height.git
cd tomosar2height
```

* Set up a Conda environment and install dependencies
```bash
conda create --name tomosar2height python=3.10
conda activate tomosar2height
conda install pytorch==2.3.0 torchvision==0.18.0 pytorch-cuda=11.8 pytorch-scatter affine laspy matplotlib rasterio scikit-learn scipy tabulate tqdm transformations trimesh urllib3 wandb hydra-core hydra-colorlog omegaconf gdal=3.6 -c pyg -c pytorch -c nvidia -c conda-forge
pip install open3d==0.18.0
```

## 🚀 Usage

### 📂 Data preparation

Prepare the dataset (about 10 seconds):
```bash
# Build the Berlin data
python scripts/build_dataset.py dataset=berlin

# Build the Munich data
python scripts/build_dataset.py dataset=munich
```

### 🎯 Training

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

### 📊 Evaluation

* Evaluate a trained TomoSAR2Height model (about 10 seconds):
```bash
# Berlin data (point cloud only)
python test.py dataset=berlin use_cloud=true use_image=false use_footprint=false run_suffix='_alto' gpu_id=0

# Munich data (point cloud only)
python test.py dataset=munich use_cloud=true use_image=false use_footprint=false run_suffix='_alto' gpu_id=0 model.encoder_kwargs.unet_kwargs.depth=6
```
The results will be saved at `./outputs/TomoSAR2Height-{dataset}{run_suffix}/tiff_test`.
