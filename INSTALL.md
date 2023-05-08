## Installation

Most of the requirements of this projects are exactly the same as [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). If you have any problem of your environment, you should check their [issues page](https://github.com/facebookresearch/maskrcnn-benchmark/issues) first. Hope you will find the answer.

### Requirements:



### Step-by-step installation

conda create --name=mask3d python=3.10.6
conda activate mask3d

conda update -n base -c defaults conda
conda install openblas-devel -c anaconda


## this installs the right pip and dependencies for the fresh python
conda install ipython
conda install scipy
conda install h5py

## scene_graph_benchmark and coco api dependencies
pip install ninja yacs cython matplotlib tqdm opencv-python overrides

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116

pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116

export INSTALL_DIR=$PWD

## install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

## install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext

## install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch.git
cd scene-graph-benchmark

## the following will install the lib with
## symbolic links, so that you can modify
## the files if you want and won't need to
## re-build it
python setup.py build develop


unset INSTALL_DIR

