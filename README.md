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
_note: python version used during the making of this project : `3.10.12`_

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
A second program, named **`001-Augmentation.py`**, has been developed to balance the dataset. It employs data augmentation techniques, including rotation, projection, scaling, blur, etc., to generate six types of augmented images for each original image.
usage:
_op1: augment all images in a given folder to given destination_
_op2: augment a single image to "./augmented_directory" and plot it_
```txt
option1: ./001-Augmentation.py ./path/to/folder/ -f="/path/to/export_location"
option2: ./001-Augmentation.py ./path/to/image
```
##### - Augment single image
```shell
$> ./001-Augmentation.py dataset/Apple/Apple_healthy/image\ \(1337\).JPG
```

![image](https://i.imgur.com/4Pddmb9.png)

##### - Balance all dataset
```shell
$> ./001-Augmentation.py dataset/ -l="augmented_directory" 
&& ./000-Distribution.py ./augmented_directory
```
- _all classes are equivalent now_
![image](https://i.imgur.com/H3Knhwv.png)

## 🎞️ Image Transformation:
In this section, the **`002.Transformation.py`** program is crafted to harness the functionality of the **PlantCV library**. **Transformations**, which are processes that alter the appearance or characteristics of images, play a crucial role in the **leaf classification** domain. These transformations, such as **Gaussian blur**, **ROI (Region of Interest) object identification**, and **object analysis**, are applied directly to **plant leaf images**.

**Transformations** are essential in **leaf classification** as they help enhance the quality of input images and highlight relevant features. For instance, **Gaussian blur** smoothens the image, reducing noise and enhancing structural details. **ROI object identification** focuses the analysis on specific regions of interest, ensuring that only pertinent information is considered. **Object analysis** provides valuable insights into the characteristics of identified objects within the images.

By incorporating these **transformations**, the program aims to preprocess **leaf images** effectively, extracting key features that contribute to accurate and meaningful **leaf classification**. The **[PlantCV library](https://plantcv.danforthcenter.org/)** serves as a powerful tool in this process, offering a range of functions for image analysis and transformation.

#### ==> transform single given image and visualize the output

```shell
$> ./002-Transformation.py dataset/Apple/Apple_healthy/image\ \(1337\).JPG
```
![image](https://i.imgur.com/93cM7F1.png)
![image](https://i.imgur.com/avuJHe7.png)

#### ==> transform all images in given source folder to given destination folder
```shell
$> ./002-Transformation.py augmented_datasets_train -dst="augmented_datasets_train_transformed"
Source directory :  augmented_datasets_train
Destination directory :  augmented_datasets_train_transformed
Bulk transformer is running now, please be patient...
100%|███████████████████████████████████████████████████████████| 11880/11880 [06:30<00:00, 30.42it/s]
```
![image](https://i.imgur.com/s1tZHhR.png)


## 🤖 Classification:
In the last phase, the development process includes the creation of two distinct programs: **`003-train.py`** and **`004-predict.py`**.

### 1- Train:
Within the **`003-train.py`** program, augmented images are employed to discern the distinctive features of designated leaf diseases. This involves leveraging a Convolutional Neural Network (CNN) implemented using the Keras framework. The acquired learning outcomes are then stored and provided in the form of a compressed .zip archive.

`comming soon...`
### 2- Prediction:
On the other hand, the **`004-predict.py`** program is designed to take a leaf image as its input. It not only displays the original image but also showcases its various transformations. Furthermore, the program makes predictions regarding the specific type of disease present in the given leaf.

```shell
./004-predict.py dataset/Apple/Apple_healthy/image\ \(1337\).JPG
```
![predict](https://i.imgur.com/Mt1DhIn.png)

```shell
$ ./004-predict.py -h
usage: 004-predict.py [-h] [-lb LABELS] [-m MODEL] image_path

Predict class of a leaf image or directory

positional arguments:
  image_path            Path to the image

options:
  -h, --help            show this help message and exit
  -lb LABELS, --labels LABELS
                        /path/to/labels.txt (default: models/labels.txt)
  -m MODEL, --model MODEL
                        /path/to/model.h5 (default: models/model.h5)
```
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
