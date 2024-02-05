#!/usr/bin/env python3
import cv2
import os
from libft import Options, Transforner
import numpy as np
import shutil
from matplotlib import pyplot as plt
import argparse
from tqdm import tqdm
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model  # noqa: E402

MODEL = None  # load_model('models/model_26.h5')
CLASSES = []


def plot_prediction(image, image_masked, class_name_prediction, img_path):
    fig, (ax0, ax1) = plt.subplots(
        nrows=1,
        ncols=2,
        sharex=True,
        figsize=(12, 8)
    )
    ax0.imshow(image)
    ax0.set_title(f"Image : {img_path}", fontsize=11)
    ax1.imshow(image_masked)
    ax1.set_title("Masked image", fontsize=11)
    fig.suptitle(f"Class predicted : {class_name_prediction}", fontsize=20)
    plt.show()


def predict(img):
    image = np.array(img)
    image_resize = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    yhat = MODEL.predict(np.expand_dims(image_resize, axis=0), verbose=0)
    predicted_index = np.argmax(yhat)
    predicted_class = CLASSES[predicted_index]
    return predicted_class


def predict_folder(folder_path):
    all_images = [
        os.path.join(dp, f) for dp, dn, filenames in
        os.walk(folder_path) for f in filenames
        if os.path.splitext(f)[1] == '.JPG'
    ]
    correctPredicts = 0
    malPredicts = []
    for img_path in tqdm(all_images):
        img_class = Options(img_path).class_name
        img_predicted_class = predict_image(img_path=img_path, plot=False)
        if img_class == img_predicted_class:
            correctPredicts += 1
        else:
            malPredicts.append([img_class, img_predicted_class])

    percentage = 100 * float(correctPredicts)/float(len(all_images))
    percentage = round(percentage, 2)
    raw_percentage = f"{correctPredicts}/{len(all_images)}"
    print(f"{raw_percentage} ({percentage}%) predicted correctly")

    pass


def predict_image(img_path, plot=True):
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
        'Gaussian_Blur': cv2.imread(
            transformer2.getPath("gaussian_blur")
        ),
        'Mask': cv2.imread(
            transformer2.getPath("mask")
        ),
        'Roi_Objects': cv2.imread(
            transformer2.getPath("roi_objects")
        ),
        'Pseudo_LandMarks': cv2.imread(
            transformer2.getPath("pseudolandmarks")
        ),
        'Analysis_Obj.': cv2.imread(
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

    predicted_classname = max(predictions, key=predictions.get)
    if plot:
        plot_prediction(
            transformer.img,
            transformer.masked2,
            predicted_classname,
            img_path=IMG
        )

    shutil.rmtree(options.destination)
    return predicted_classname


def main():
    global CLASSES
    global MODEL

    parser = argparse.ArgumentParser(
        description="Predict class of a leaf image or directory"
    )
    parser.add_argument(
        "-lb",
        "--labels",
        default="models/labels.txt",
        help="/path/to/labels.txt (default: models/labels.txt)",
    )

    parser.add_argument(
        "-m",
        "--model",
        default="models/model.h5",
        help="/path/to/model.h5 (default: models/model.h5)",
    )
    parser.add_argument("image_path", help="Path to the image")
    args = parser.parse_args()

    with open(args.labels, 'r') as file:
        CLASSES = [line.strip() for line in file if line.strip()]

    if os.path.isfile(args.model):
        MODEL = load_model(args.model)
    else:
        print("Invalid model path!")
        exit(1)

    if os.path.isfile(args.image_path):
        predict_image(args.image_path)

    elif os.path.isdir(args.image_path):
        predict_folder(args.image_path)

    else:
        print("Invalid input path!")
        exit(1)


if __name__ == "__main__":
    main()
