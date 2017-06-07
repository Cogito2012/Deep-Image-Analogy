

# Deep Image Analogy

## Introduction

**Deep Image Analogy** is a technique to find semantically-meaningful dense correspondences between two input images. It adapts the notion of image analogy with features extracted from a Deep Convolutional Neural Network.

**Deep Image Analogy** is initially described in a [SIGGRAPH 2017 paper](https://arxiv.org/abs/1705.01088)


![image](https://github.com/msracver/Deep-Image-Analogy/blob/master/windows/deep_image_analogy/example/readme/teaser.png)


## Disclaimer

This is an reimplemention of [Deep Image Analogy](https://github.com/msracver/Deep-Image-Analogy/tree/linux) with C++ combined with CUDA. It is worth noticing that:
- The codes are based on [Caffe](https://github.com/Microsoft/caffe).
- The codes only have been tested on Ubuntu with CUDA 8 or 7.5.
- The codes only support for machine with GPU, and have been tested on Nvidia GeoForce GTX 1080.


## License

© Microsoft, 2017. Licensed under an  BSD 2-Clause license.

## Citation
If you find **Deep Image Analogy** (include deep patchmatch) helpful for your research, please consider citing:
```
  @article{liao2017visual,
    title={Visual Attribute Transfer through Deep Image Analogy},
    author={Liao, Jing and Yao, Yuan and Yuan, Lu and Hua, Gang and Kang, Sing Bing},
    journal={arXiv preprint arXiv:1705.01088},
    year={2017}
  }
```

## Getting Started

### Prerequisite

- Linux and CUDA 8 or 7.5

### Configure & Build

- Install dependencies for building Caffe. Just follow the tutorial from [Caffe](http://caffe.berkeleyvision.org/).
- Use configuration script to download [VGG19 Caffemodel](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel) by typing ```sh scripts/config_deep_image_analogy.sh```.
- Modify the CUDA path in ```Makefile.config.example``` and rename it to ```Makefile.config```.
- Compile Caffe, make sure you installed all the dependencies before. Just type ```make all```.
- Compile deep_image_analogy by ```sh scripts/make_deep_image_analogy.sh```.
- Add libraries built by Caffe into ```LD_LIBRARY_PATH``` by ```export LD_LIBRARY_PATH="./build/lib"```.

### Datastets

To run the codes for multiple images, the input datasets are required to be formed as follow steps:

- Put the content images and style images into ```./deep_image_analogy/images_content``` and ```./deep_image_analogy/images_style``` folders.
- Edit the configuration in ```./script/generate_list.sh``` and then run
```sudo sh ./script/generate_list.sh``` to generate file lists.

- Finally, the datasets will be formed as follows:
```
--- datasets/images_content/*.jpg
--- datasets/images_style/*.jpg
--- datasets/content_list.txt
--- datasets/style_list.txt
```

**Tips:** The size of input image is limited, mostly should not be large than 700x500 if you use 1.0 for parameter **ratio**.

### Demo

To run the demo, just type:
```
./demo deep_image_analogy/models/ deep_image_analogy/demo/content.png deep_image_analogy/demo/style.png deep_image_analogy/demo/output/ 0 0.5 2 0
```

You need to set several parameters which have been mentioned in the paper. To be more specific, you need to set

- **path_model**, where the VGG-19 model is.
- **path_A**, the file list of input content images A.
- **path_BP**, the file list of input style images BP.
- **path_output**, the output path, will be created automatically.
- **GPU Number**, GPU ID you want to run this experiment.
- **Ratio**, the ratio to resize the inputs before sending them into the network.
- **Blend Weight**, the level of weights in blending process.
- **Flag of WLS Filter**, if you are trying to do photo style transfer, we recommend to switch this on to keep the structure of original photo.



### Tips

- We often test images of size 600x400 and 448x448.
- We set ratio to 1.0 by default. Specifically, for face (portrait) cases, we find ratio = 0.5 often make the results better.
- Blend weight controls the result appearance. If you want the result to be more like original content photo, please increase it; if you want the result more faithful to the style, please reduce it.
- For the four applications, our settings are mostly (but not definitely):
  - Photo to Style: blend weight=3, ratio=0.5 for face and ratio=1 for other cases.
  - Style to Style: blend weight=3, ratio=1.
  - Style to Photo: blend weight=2, ratio=0.5.
  - Photo to Photo: blend weight=3, ratio=1.

## Acknowledgments

Our codes acknowledge [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), [PatchMatch](http://gfx.cs.princeton.edu/gfx/pubs/Barnes_2009_PAR/index.php), [CudaLBFGS](https://github.com/jwetzl/CudaLBFGS) and [Caffe](https://github.com/BVLC/caffe). We also acknowledge to the authors of our image and style examples but we do not own the copyrights of them.
