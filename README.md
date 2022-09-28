# cephaAI
![](https://i.ibb.co/b6cG49p/image.png)
## Introduction

CephaAI is an streamlit based app, used for automated 2D cephalomtrical diagnosis using a deep learning model.
  >The cephalometrical analysis in cephaAI, start first detecting anatomical landmarks positions and calculating the angles to get interpretations. CephaAI provide a vue of heatmaps produced by the model, those heatmaps show for each landmark the positions with high probablities to be that landmark.
The model we're using on our app is the implementation of the following paper [Cephalometric Landmark Detection by Attentive Feature Pyramid Fusion and Regression-Voting](https://arxiv.org/pdf/1908.08841.pdf).

## Dataset 
The dataset we used to train the model is the same dataset provided in the ISBI2015 challenge, it contains 400 skull X-ray images labled with ".txt" files containing the landmarks coordinates [You can download the dataset here!](https://figshare.com/s/37ec464af8e81ae6ebbf).
````
path/to/cephalometric
	400_junior
		001.txt
		...
	400_senior
		001.txt
		...
	RawImage
		TrainingData
			001.bmp
			...
		Test1Data
			151.bmp
			...
		Test2Data
			301.bmp
			...
````
## Configuration and requirements
In the cepphaAI we're using the following frameworks:
  * numpy~=1.23.3
  * streamlit~=1.13.0
  * matplotlib~=3.6.0
  * pandas~=1.5.0
  * pillow~=9.2.0
  * torch~=1.12.1
  * torchvision~=0.13.1
  * scikit-image~=0.19.3
## Training and saving the model:
You can see the following [Kaggle Notebook]().
## Demo
![]()
## Reference
```
@inproceedings{chen2019cephalometric,
  title={Cephalometric landmark detection by attentive feature pyramid fusion and regression-voting},
  author={Chen, Runnan and Ma, Yuexin and Chen, Nenglun and Lee, Daniel and Wang, Wenping},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={873--881},
  year={2019},
  organization={Springer}
}
```
