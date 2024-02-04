# Training code
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
import cv2
import imghdr
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import argparse
import yaml

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def load_config(config_file):
    with open(config_file, 'r') as file:
        config_dict = yaml.safe_load(file)
    return Config(**config_dict)

def generate_config(config_file):
    """
    Generate a configuration file with default values.
    """
    default_config = {
        'model': "",
        'epochs': 1,
        'model_save_location': "models",
        'training_set': "augmented_datasets_train_transformed",
        'validation_set': "augmented_datasets_validation",
        'testing_set': "augmented_datasets_test"
    }
    with open(config_file, 'w') as file:
        yaml.dump(default_config, file)


class Trainer:
    def __init__(self, data_set_train, data_set_test, model, save_path):
        """
        Initialize the Trainer object.

        Args:
        - data_set_train: Training dataset.
        - data_set_test: Test dataset.
        - model: Path to a pre-trained model, if available.
        """
        self.data_set = data_set_train
        self.save_path = save_path
        self.train = 0
        self.val = 0
        self.test = data_set_test
        if (model):
            self.model = tf.keras.models.load_model(model)
        else:    
            self.model = Sequential()
        self.history = 0
        self.classes = []

    def group_data(self, training, validation):
        """
        Group the training dataset into training and validation sets.

        Args:
        - training: Percentage of data to use for training.
        - validation: Percentage of data to use for validation.
        """
        train_size = int(len(self.data_set)*.7)
        val_size = int(len(self.data_set)*.3)
        
        self.train = self.data_set.take(train_size)
        self.val = self.data_set.skip(train_size).take(val_size)
        
    def save_classes(self, path):
        """
        Save the classes found in the dataset.

        Args:
        - path: Directory containing the dataset.
        """
        dataset_directory = path
        class_directories = [d for d in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, d))]
        self.classes = class_directories
        with open(self.save_path + '/labels', 'w') as file:
            for class_name in class_directories:
                file.write(class_name + '\n')
        print("Classes saved:", self.classes)
        
    def build_neural_network_layers(self, model_choice):
        """
        Build the neural network layers based on the chosen model configuration.

        Args:
        - model_choice: Integer indicating the model configuration to use.
        """
        if model_choice == 1:
            self.model.add(Conv2D(16, 3, padding='same', activation='relu')),
            self.model.add(MaxPooling2D()),
            self.model.add(Conv2D(32, 3, padding='same', activation='relu')),
            self.model.add(MaxPooling2D()),
            self.model.add(Conv2D(64, 3, padding='same', activation='relu')),
            self.model.add(MaxPooling2D()),
            self.model.add(Flatten()),
            self.model.add(Dense(128, activation='relu')),
            self.model.add(Dense(8, activation='softmax')),
        if model_choice == 2:
            self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(64, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Conv2D(128, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            self.model.add(Flatten())
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(8, activation='softmax'))

        if model_choice == 3:
            self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
            self.model.add(MaxPooling2D((2, 2)))
            self.model.add(Conv2D(64, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D((2, 2)))
            self.model.add(Conv2D(128, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D((2, 2)))
            self.model.add(Flatten())
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dense(8, activation='softmax'))
  
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    def start(self, epoch):
        """
        Start training the neural network.

        Args:
        - epoch: Number of epochs for training.
        """
        self.history = self.model.fit(self.train, epochs=epoch, validation_data=self.val)

    def save(self, path):
        """
        Save the trained model.

        Args:
        - path: Directory to save the trained model.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path,'imageclassifier.h5'))
        
    def plot_history(self):
        """
        Plot training history (loss and accuracy).
        """
        fig = plt.figure()
        plt.plot(self.history.history['loss'], color='teal', label='loss')
        plt.plot(self.history.history['val_loss'], color='orange', label='val_loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()

        fig = plt.figure()
        plt.plot(self.history.history['accuracy'], color='teal', label='accuracy')
        plt.plot(self.history.history['val_accuracy'], color='orange', label='val_accuracy')
        fig.suptitle('Accuracy', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()
        
    def testing(self):
        """
        Perform testing on the test dataset and print evaluation metrics.
        """
        pre = Precision()
        re = Recall()
        acc = BinaryAccuracy()
        for batch in self.test.as_numpy_iterator(): 
            X, y = batch
            yhat = self.model.predict(X)
            pre.update_state(y, yhat)
            re.update_state(y, yhat)
            acc.update_state(y, yhat)
        print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')

def main(args):
    """
    Main function to train the convolutional neural network.

    Args:
    - config: Dictionary containing configuration parameters.
    """
    print(args)
    training_data = tf.keras.utils.image_dataset_from_directory(args.training_set)
    testing_data = tf.keras.utils.image_dataset_from_directory(args.validation_set)
    training_data = training_data.map(lambda x, y: (x / 255, tf.one_hot(y, depth=8)))
    testing_data = testing_data.map(lambda x, y: (x / 255, tf.one_hot(y, depth=8)))
    trainer = Trainer(training_data, testing_data, args.model, args.model_save_location)
    trainer.save_classes(args.training_set)
    trainer.group_data(7, 2)
    if (not args.model):
        trainer.build_neural_network_layers(3)
    if (args.epochs):
        trainer.start(args.epochs)
        trainer.plot_history()
        trainer.save(args.model_save_location)
    trainer.testing()
if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Train a convolutional neural network")
    parser.add_argument("--generate-config", action="store_true", help="Generate a default configuration file")
    args = parser.parse_args()

    if args.generate_config:
        generate_config('config.yaml')
        print("Default configuration file generated successfully.")
    else:
        config = load_config('config.yaml')
        main(config)