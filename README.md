# Team Name: Unstable Diffusion
# Team Members: 
* Csik Laura - Z09RRY
* Váradi Dominik - QCCH6T
* Zsáli Zsombor - A6HFRW
# Project Description
Image generation with diffusion models
Implementing and training diffusion models, such as DDPM (Denoising Diffusion Probabilistic Model) for generating realistic images. Evaluating the capabilities of the models on two different datasets, such as CelebA and Danbooru.
# Functions of files
DL_hf.ipynb: data preparation
# Related Works
[CelebA Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)   
[Danbooru Faces Kaggle](https://www.kaggle.com/datasets/subinium/highresolution-anime-face-dataset-512x512)   
# Run
### How to aquire the datasets
You have to manually download the datasets to your local machine.   
[CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset/download?datasetVersionNumber=2)   
[Danbooru Faces](https://www.kaggle.com/datasets/subinium/highresolution-anime-face-dataset-512x512/download?datasetVersionNumber=1)   
After downloading and unzipping the data, change the dataset path in [docker-compose.yaml](https://github.com/ZsZs88/DL-hf/blob/f2817fa6fdb07b495011c3d05330e67f76cda19d/docker-compose.yml#L9).
### How to run the Jupyter Lab in a Docker container
docker-compose build  
docker-compose up  
Open the link and change the 8888 port number to 8899.  
