# write prediction logic here
from tensorflow.keras.models import load_model
import os
import numpy as np
import cv2


def main():
    new_model = load_model('models/imageclassifier3.h5')

    # Directory containing the dataset
    dataset_directory = 'augmented_datasets_train_transformed'

    # testign image
    img = cv2.imread('augmented_datasets_validation/Grape_Esca/image (33).JPG')

    # Get a list of subdirectories (each subdirectory represents a class)
    class_directories = [d for d in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, d))]
    class_directories.reverse()  # Reverse the list of directories

    # Save the class names to the 'classes' attribute
    classes = sorted([d for d in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, d))])

    yhat = new_model.predict(np.expand_dims(img/255, 0))
    predicted_index = np.argmax(yhat)
    predicted_class = classes[predicted_index]

    print(predicted_class)

if __name__ == "__main__":
    # Call the main function if the script is run directly
    main()