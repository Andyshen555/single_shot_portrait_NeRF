# Single Shot Portrait NeRF
## ReadMe
This is a personal project to generate 3D portrait model from a single RGB image. The project started from NVIDIA's lp3d and made modifications based on the original lp3d.\
Our model has changed the dual branch encoder to learn a common human head shape (cannonical) and add details to the base 3D model.

## Environment setup
1. Install anaconda
2. ```conda env create -f environment.yml```
3. ```conda activate lp3d```

## EG3D weights
1. Go to project folder
2. ```wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/eg3d/1/files?redirect=true&path=ffhq512-128.pkl' -O ffhq512-128.pkl```

## Training
### Train with single node
```bash run_train.sh```
### Train with multiple nodes
1. On the first machine, ```bash run_distributed0.sh```.
2. On the second machine, ```bash run_distributed1.sh```.

# View Result
```python gen_sample.py --outdir=out --trunc=0.9 --shapes=False --seeds=0-5 --network=ffhq512-128.pkl```

# Todo
1. Replace the upsampler from eg3d with GFPGan upsampler
