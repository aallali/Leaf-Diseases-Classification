#!/usr/bin/env python3

import cv2
import os
from libft import Options, Transforner
import numpy as np
import shutil
from matplotlib import pyplot as plt
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model  # noqa: E402

MODEL = load_model('models/imageclassifier.h5')

CLASSES = []


def plot_prediction(image, image_masked, class_name_prediction, img_path):
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2,
                                   sharex=True, figsize=(12, 8))
    ax0.imshow(image)
    ax0.set_title(f"Image : {img_path}", fontsize=10)
    ax1.imshow(image_masked)
    ax1.set_title("Masked image", fontsize=10)
    fig.suptitle(f"Class predicted : {class_name_prediction}")
    plt.show()


def predict(img):

    yhat = MODEL.predict(np.expand_dims(img / 255, 0))
    predicted_index = np.argmax(yhat)
    predicted_class = CLASSES[predicted_index]

    return predicted_class


def predict_all_unit_tests():
    all_images = [
        os.path.join(dp, f) for dp, dn, filenames in
        os.walk("./evaluation/test_images") for f in filenames
        if os.path.splitext(f)[1] == '.JPG'
    ]
    for img_path in all_images:
        predict_image(img_path)
    pass


def predict_image(img_path):
    IMG = img_path
    result = []
    predictions = dict()

    options = Options(IMG, dest_path="./.temp_predict")
    transformer = Transforner(options)
    transformer.mask()

    transformer2 = Transforner(options)
    transformer2.load_original(img_raw=transformer.masked2)
    transformer2.run_all()

    trans = {
        'Fig2. Gaussian_Blur': cv2.imread(
            transformer2.getPath("gaussian_blur")
        ),
        'Fig3. Mask': cv2.imread(
            transformer2.getPath("mask")
        ),
        'Fig4. Roi_Objects': cv2.imread(
            transformer2.getPath("roi_objects")
        ),
        'Fig5. Pseudo-LandMarks': cv2.imread(
            transformer2.getPath("pseudolandmarks")
        ),
        'Fig6. Analysis Obj.': cv2.imread(
            transformer2.getPath("analysis_obj")
        )
    }

    for t in trans:
        className = predict(trans[t])
        result.append(t + ", classed as: " + className)
        if className in predictions:
            predictions[className] += 1
        else:
            predictions[className] = 1

    plot_prediction(
        transformer.img,
        transformer.masked2,
        max(predictions, key=predictions.get),
        img_path=IMG
    )

    shutil.rmtree(options.destination)
    return


def main():
    global CLASSES

    with open("./data_classes.txt", 'r') as file:
        CLASSES = [line.strip() for line in file if line.strip()]

    # predict_all_unit_tests()
    parser = argparse.ArgumentParser(
        description="Predict class of a leaf image"
    )
    parser.add_argument("image_path", help="Path to the image")

    args = parser.parse_args()

    predict_image(args.image_path)


if __name__ == "__main__":
    main()
