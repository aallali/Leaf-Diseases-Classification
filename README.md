# Leaf-Diseases-Classification  🌿 (Leaffliction)

## 📜 Description

This is a computer vision project for plant leaf diseases.
In this project we'll be doing image dataset analysis, data augmentation, image transformations and image classification.

## 📦 Installation

To setup the project, you need to launch the following command:

```bash
git clone https://github.com/aallali/Leaf-Diseases-Classification-CNN \
    && cd Leaf-Diseases-Classification-CNN
bash setup_env.sh
source venv/bin/activate
```

## 📋 Summary

- [Data Analysis](#-data-analysis)
- [Data Augmentation](#-data-augmentation)
- [Image Transformations](#-image-transformations)
- [Classification](#-classification)

## 📊 Data analysis

A program named **000-Distribution.py** is created to extract and analyze the image dataset of plant leaves. Pie charts and Bar charts are generated for each plant type, using images available in the subdirectories of the given input directory.
usage:
```shell
$> python 000-Distribution.py ./dataset/Grape
folder_statistics
	Grape_Esca : 1382
	Grape_spot : 1075
	Grape_healthy : 422
	Grape_Black_rot : 1178
total_augmentation_to_balance : 2954
root_path : ./dataset/Grape

```

![image](https://i.imgur.com/759ey5n.png)
