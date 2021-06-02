# SynSLaG: Synthetic Sign Language Generator

## Contents
* [Tool user manual](https://github.com/TeppeiMIURA/SynSLaG#tool-user-manual)
* [Citation](https://github.com/TeppeiMIURA/SynSLaG#citation)
* [License](https://github.com/TeppeiMIURA/SynSLaG#license)


## Tool user manual
In order to download the SynSLaG source code and body models, you need to agree the license terms.
The links to license terms are available here: [https://github.com/TeppeiMIURA/SynSLaG/LICENSE.md](https://github.com/TeppeiMIURA/SynSLaG/blob/master/LICENSE.md)

### 1. Preparation

#### 1.1. 3D motion data
You need download 3D motion data following:

* Y. Nagashima, "Construction of Multi-purpose Japanese Sign Language Database," in Human Systems Engineering and Design, Cham, 2019.

Please access the database repository https://www.nii.ac.jp/dsc/idr/rdata/KoSign/.

#### 1.2. Synthetic ingredients from SURREAL dataset
You need to download some ingredients from https://github.com/gulvarol/surreal in order to run the synthetic data generation.
Once you read and agree on SURREAL license terms and have access to download.

##### Spherical harmonics and Body-part segmentation
You place the spherical harmonics file and body-part segmentation file into SynSLaG structure as following:
```
surreal/datageneration/spher_harm/sh.osl --> SynSLaG/sphers/
surreal/datageneration/pkl/segm_per_v_overlap.pkl --> SynSLaG/models/
```

##### SMPL data and clothing textures
You firstly need to download SMPL data by `surreal/download/download_smpl_data.sh` in SURREAL.
The downloaded data includes files as following:
* `textures/` : folder containing clothing images.
* `smpl_data.npz` : smpl data containing below parameters.
    * `maleshapes       [1700 x 10] - SMPL shape parameters for 1700 male scans`
    * `femaleshapes     [2103 x 10] - SMPL shape parameters for 2103 female scans` 
    * `regression_verts [232]`
    * `joint_regressor  [24 x 232]`
    * `trans*           [T x 3]     - (T: number of frames in MoCap sequence)`
    * `pose*            [T x 72]    - SMPL pose parameters (T: number of frames in MoCap sequence)`

Then, you place the files into SynSLaG structure as following:
```
textures/* --> SynSLaG/textures/
smpl_data.npz --> SynSLaG/shapes/
```

#### 1.3. Blender
You need to download [Blender](https://www.blender.org/) and install some packages.
The provided code was tested with [Blender v2.79](https://download.blender.org/release/Blender2.79/), and install packages as following:
```
# Install pip in Blender
BLENDER_PATH//2.79/python/bin/python.exe -m ensurepip
BLENDER_PATH//2.79/python/bin/python.exe -m pip install --user --upgrade pip

# Install packages
BLENDER_PATH//2.79/python/bin/python.exe -m pip install --user -U numpy opencv-python scipy
```

### 2. Running
Modify following text file according to your generation:
* `SynSLaG/actions/(fe)male.txt`
* `SynSLaG/textures/(fe)male_all.txt`
* `SynSLaG/backgrounds/(in/out)side.txt`

Modify `config` file according to your generation, then execute batch file for running.
```
.\run.bat
```

### 3. Optional tool
We provide optional tool to easily visualize output's synthetic dataset `SynSLaG/tools/RestSynthData.py`.
Execute the tool to output `.tar.gz` file as following:
```
tools/RestSynthData.py --file output/(filename).tar.gz
```

## Citation
If you use this tool, please cite the following:

```
Be submitting...
```

## License
Please check the [license terms](https://github.com/TeppeiMIURA/SynSLaG/blob/master/LICENSE.md) before downloading and/or using the source code and body models.
