# TomoSAR2Height

This repository provides the code for training and evaluating **TomoSAR2Height**, a method for reconstructing building heights (nDSMs) from spaceborne TomoSAR point clouds.

## 🛠️ Installation

* Clone the repository
```bash
git clone git@github.com:zhu-xlab/tomosar2height.git
cd tomosar2height
```

* All-in-one installation to create a conda environment with all dependencies
```bash
conda env create -f environment.yml && conda activate tomosar2height
```

* If you prefer manual installation, follow these steps
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
# Replace `berlin` to `munich` for Munich data
python train.py dataset=berlin use_cloud=true use_image=false wandb=true run_suffix=_cloud gpu_id=0
```

* Using point clouds and images
```bash
# Replace `berlin` to `munich` for Munich data
python train.py dataset=berlin use_cloud=true use_image=true wandb=true run_suffix=_cloud+image gpu_id=0
```

### 📊 Evaluation

Before evaluation, make sure checkpoints are available at `./outputs/TomoSAR2Height-{dataset}{run_suffix}/check_points/model_best.pt`.

* Evaluate a trained TomoSAR2Height model (point clouds only):
```bash
# Berlin data
python test.py dataset=berlin use_cloud=true use_image=false run_suffix=_cloud gpu_id=0

# Munich data
python test.py dataset=munich use_cloud=true use_image=false run_suffix=_cloud gpu_id=0
```

* Evaluate a trained TomoSAR2Height model (point clouds and images):
```bash
# Berlin data (point cloud & image)
python test.py dataset=berlin use_cloud=true use_image=true run_suffix=_cloud+image gpu_id=0

# Munich data (point cloud & image)
python test.py dataset=munich use_cloud=true use_image=true run_suffix=_cloud+image gpu_id=0
```
Specify `run_suffix={YOUR_SUFFIX}` with your desired suffix if needed. The results will be saved at `./outputs/TomoSAR2Height-{dataset}{run_suffix}/tiff_test`.

### ⚙️ Available configurations
```bash
# check available configurations for training
python train.py --cfg job

# check available configurations for evaluation
python test.py --cfg job
```
Alternatively, review the configuration file: `conf/config.yaml`.
