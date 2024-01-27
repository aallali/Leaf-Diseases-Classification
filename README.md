# Leaf-Diseases-Classification  🌿 (Leaffliction)

## 📜 Description

This project focuses on computer vision applications related to plant leaf diseases. It encompasses tasks such as image dataset analysis, data augmentation, image transformations, and image classification to address various aspects of plant health in the context of leaf diseases.

## 🛠️ Setup Project:

To setup the project, you need to launch the following command:

```bash
$> git clone https://github.com/aallali/Leaf-Diseases-Classification
$> cd Leaf-Diseases-Classification
$> bash ft_setup_env.sh
$> bash ft_setup_dataset.sh
$> source venv/bin/activate
```

## 📋 Table of contents

- [Data Analysis](#-data-analysis)
- [Data Augmentation](#-data-augmentation)
- [Image Transformations](#-image-transformations)
- [Classification](#-classification)
- [Unit Tests](#-unit-tests)

## 📊 Data analysis

The script named **`000-Distribution.py`** is designed for extracting and analyzing an image dataset of plant leaves. It processes images from subdirectories within the provided input directory, generating both pie charts and bar charts for each plant type in the dataset.
usage:
there is 2 options to visualize data distribution either for all the dataset folder or just specific subfolder inside.
```txt
option1: python 000-Distribution.py ./path/to/folder/
option2: python 000-Distribution.py ./path/to/folder/subfolder
```

```shell
$> python 000-Distribution.py ./dataset/
('Apple_healthy', 1640)
('Apple_scab', 629)
('Apple_Black_rot', 620)
('Apple_rust', 275)
('Grape_Esca', 1382)
('Grape_spot', 1075)
('Grape_healthy', 422)
('Grape_Black_rot', 1178)
```
![image](https://i.imgur.com/6rXh3iI.png)

## 🗃️ Data Augmentation:

commin soon...

## 🎞️ Image Transformation:

commin soon...

## 🤖 Classification:

commin soon...


---
## 🔬 Unit Tests:
we tried to run as much unit tests as possible for program especially the helpers functions (`libft`)
- run all tests:
```commandline
python3 -m unittest discover -s ./unittests -t .. -v
```
- run single test file:
```commandline
python3 -m unittest unittests/libft/test_ft_dict.py -v
```
