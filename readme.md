# Outfit Transformer: Outfit Representations for Fashion Recommendation

## Introduction

This repository contains the implementation of the Outfit Transformer, inspired by the original paper:

> Rohan Sarkar et al. [Outfit Transformer: Outfit Representations for Fashion Recommendation](https://arxiv.org/abs/2204.04812). CVPR 2023.

Our implementation includes improvements by incorporating 'Style' and 'Event' features.

## Settings

### Environment Setting
```
conda create -n outfit-transformer python=3.12.4
conda activate outfit-transformer
conda env update -f environment.yml
```
### Download Dataset
```
mkdir datasets
cd datasets
gdown --id 1ox8GFHG8iMs64iiwITQhJ47dkQ0Q7SBu
unzip polyvore.zip -d polyvore
cd ../
```
### Download Checkpoint
Pretrained model checkpoints are also available [here](https://drive.google.com/drive/folders/1cMTvmC6vWV9F9j08GX1MppNm6DDnSiZl?usp=drive_link).

## Training
Follow the steps below to train the model:

### Step 1: Compatibility Prediction
Start by training the model for the Compatibility Prediction (CP) task

**Train**
```
python -m src.run.1_train_compatibility \
--wandb_key $YOUR/WANDB/API/KEY
```
**Test**
```
python -m src.run.1-1_test_compatibility \
--checkpoint $PATH/TO/LOAD/MODEL/.PT/FILE
```

<br>

### Step 2: Complementary Item Retrieval

After completing Step 1, use the checkpoint with the best accuracy from the Compatibility Prediction task to train the model for the Complementary Item Retrieval (CIR) task:

**Train**
```
python -m src.run.2_train_complementary \
--wandb_key $YOUR/WANDB/API/KEY \
--checkpoint $PATH/TO/LOAD/MODEL/.PT/FILE
```
**Test**
```
python -m src.run.2-1_test_complemenatry \
--checkpoint $PATH/TO/LOAD/MODEL/.PT/FILE
```
<br>

## New Method: Adding Style and Event to Embeddings

We have introduced a new method to add style and event information to the embeddings. This involves extracting ResNet embeddings for images and then training classifiers to predict style and event labels. The embeddings are then transformed using these classifiers.

### Step 0: Download Fashion4Event dataset
That is existing in: https://drive.google.com/drive/folders/1bgXykJSwWICoZeB8Kx79wnchDxrRbXp7

### Step 1: Tag the Style
```
python -m src.StyleKobi
```

### Step 2: Train Style and Event Classifiers
Run the notebook to extract ResNet embeddings for the images and train classifiers for style and event prediction using the extracted embeddings.
At the end of the notebook, two JSON files will be saved—one for style and one for event.
```
jupyter notebook /home/nogaschw/outfit-transformer-main/Fashion Rec -Style and Event/CreateBasicClassifiers.ipynb
```

### Step 3: Save Transformed Embeddings
Create a new embedding file by concatenating the original embeddings with the output from the JSON file.
```
python /home/nogaschw/outfit-transformer-main/create_emb.ipynb
```
<br>

In addition, we have a folder for generating the dataset as described in the paper.

*The original code is taken from*
```
https://github.com/owj0421/outfit-transformer
```
