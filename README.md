# WESG

**Paper**:  Wavelet Transform-assisted Efficient 3D Deep Shape Generation

**Authors**: Kai Xu, Zhihao Liao, Yujuan Lu, Qiegen Liu and Yuhao Wang

Date: 1/2022
The code and the algorithm are for non-commercial use only. 
Copyright 2022, Department of Electronic Information Engineering, Nanchang University. 

​    Unsupervised deep learning has been widely employed to generate high-quality point cloud samples. Although recent researches have demonstrated its ability to obtain remarkable results in point cloud generation task, the training stage is still computationally expensive and has a high utilization of GPU memory. This work develops a novel strategy for improving the performance of the generative model for point clouds by learning the priors using a score-based generative model in the wavelet domain. Taking advantage of the multi-scale representation provided by wavelet transform, the proposed model is more efficient to learn the gradient field of the log density, which indicates the distribution of 3D points. Specifically, in the training phase, multiple groups of tensors consisting of wavelet coefficients are applied as the input to train the network employing denoising score matching. After the model is learned, shape is iteratively updated from coarse to fine via Langevin dynamics. Experiments indicated that the present model achieves competitive performance in point cloud generation and auto-encoding, training at faster speed and lower GPU memory.

## Dependencies
```bash
# Create conda environment with torch 1.2.0 and CUDA 10.0
conda env create -f environment.yml
conda activate WESG

# Compile the evaluation metrics
cd evaluation/pytorch_structural_losses/
make clean
make all
```

## Test

if you want to test the WESG model for auto-encoding performance, please
```bash
python test.py configs/recon/airplane/airplane_recon_add.yaml \
    --pretrained pretrained/recon/airplane.pt
python test.py configs/recon/car/car_recon_add.yaml \
    --pretrained pretrained/recon/car.pt
python test.py configs/recon/chair/chair_recon_add.yaml \
    --pretrained pretrained/recon/chair.pt
python test.py configs/recon/shapenet/shapenet_recon.yaml \
    --pretrained pretrained/recon/shapenet.pt
```

if you want to test the WESG model for generation task, please
```bash
python test.py configs/gen/airplane_gen_add.yaml \
    --pretrained pretrained/gen/airplane.pt
python test.py configs/gen/car/car_gen_add.yaml \
    --pretrained pretrained/gen/car.pt
python test.py configs/gen/chair/chair_gen_add.yaml \
    --pretrained pretrained/gen/chair.pt
```



## Checkpoints

we provide pretrained checkpoints in [Baidu Drive](https://pan.baidu.com/s/10NVtDjMONxQyd9Yj8objgA). key number is "WESG".You can put it under the project root directory.

## Dataset

We choose the dataset following the guidance from PointFlow: [link](https://github.com/stevenygd/PointFlow). 

### Other Related Projects

  * Learning Gradient Fields for Shape Generation  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2008.06520)   [<font size=5>**[Code]**</font>](https://github.com/RuojinCai/ShapeGF)  

  * Wavelet Transform-assisted Adaptive Generative Modeling for Colorization  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2107.04261)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/WACM)

  * PointFlow : 3D Point Cloud Generation with Continuous Normalizing Flows  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/1906.12320)   [<font size=5>**[Code]**</font>](https://github.com/stevenygd/PointFlow)  