# LEarnable Geometric Scattering (LEGS)
### Krishnaswamy Lab, Yale University

Rewritten and updated code base for the central piece in a series of publications, including:
```
1. "Learnable Filters for Geometric Scattering Modules".
2. "Data-Driven Learning of Geometric Scattering Modules for GNNs".
```

## Dependencies
We developed the codebase in a miniconda environment.
Tested on Python 3.9.13 + PyTorch 1.12.1.
How we created the conda environment:
```
conda create --name $OUR_CONDA_ENV pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda activate $OUR_CONDA_ENV
conda install -c conda-forge pytorch_geometric torch-scatter pytorch-lightning
python -m pip install pysmiles graphein
conda install -c pytorch3d pytorch3d
```

## Usage
```
cd ./src
conda activate $OUR_CONDA_ENV
python main.py --pretrain --config ./config/baseline.yaml
```

## Citation
```
@article{tong2022learnable,
  title={Learnable Filters for Geometric Scattering Modules},
  author={Tong, Alexander and Wenkel, Frederik and Bhaskar, Dhananjay and Macdonald, Kincaid and Grady, Jackson and Perlmutter, Michael and Krishnaswamy, Smita and Wolf, Guy},
  journal={arXiv preprint arXiv:2208.07458},
  year={2022}
}
@inproceedings{tong2021data,
  title={Data-Driven Learning of Geometric Scattering Modules for GNNs},
  author={Tong, Alexander and Wenkel, Frederick and Macdonald, Kincaid and Krishnaswamy, Smita and Wolf, Guy},
  booktitle={2021 IEEE 31st International Workshop on Machine Learning for Signal Processing (MLSP)},
  pages={1--6},
  year={2021},
  organization={IEEE}
}
```