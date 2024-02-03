#training code 
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


class Trainer:
    def __init__(self, data_set_train, data_set_test, model):
        self.data_set = data_set_train
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
        train_size = int(len(self.data_set)*.7)
        val_size = int(len(self.data_set)*.3)
        
        self.train = self.data_set.take(train_size)
        self.val = self.data_set.skip(train_size).take(val_size)
        
    def save_classes(self, path):
        # Directory containing the dataset
        dataset_directory = path

        # Get a list of subdirectories (each subdirectory represents a class)
        class_directories = [d for d in os.listdir(dataset_directory) if os.path.isdir(os.path.join(dataset_directory, d))]

        # Save the class names to the 'classes' attribute
        self.classes = class_directories

        print("Classes saved:", self.classes)
        
    def build_neural_network_layers(self, model_choice):
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
            # Add Convolutional layers
            self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            self.model.add(Conv2D(64, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            self.model.add(Conv2D(128, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))

            # Flatten the output of the convolutional layers
            self.model.add(Flatten())

            # Add fully connected layers
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
            self.model.add(Dense(8, activation='softmax'))  # Output layer with 8 classes and softmax activation

        if model_choice == 3:
            self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
            self.model.add(MaxPooling2D((2, 2)))
            
            self.model.add(Conv2D(64, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D((2, 2)))
            
            self.model.add(Conv2D(128, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D((2, 2)))
            
            # Flatten layer
            self.model.add(Flatten())
            
            # Dense layers
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dense(8, activation='softmax'))  # Output layer with 8 classes, softmax activation
  
            # Compile the model with adjusted learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(optimizer='adam',
                    loss='categorical_crossentropy',  # Categorical cross-entropy loss for multi-class classification
                    metrics=['accuracy'])

        # self.model.compile(optimizer='adam',
        #       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #       metrics=['accuracy'])

    def start(self, epoch):
        self.history = self.model.fit(self.train, epochs=epoch, validation_data=self.val)
    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path,'imageclassifier.h5'))
        
    def plot_history(self):
        # plot loss
        fig = plt.figure()
        plt.plot(self.history.history['loss'], color='teal', label='loss')
        plt.plot(self.history.history['val_loss'], color='orange', label='val_loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()
        # plot accuracy
        fig = plt.figure()
        plt.plot(self.history.history['accuracy'], color='teal', label='accuracy')
        plt.plot(self.history.history['val_accuracy'], color='orange', label='val_accuracy')
        fig.suptitle('Accuracy', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()
        
    def testing(self):
        pre = Precision()
        re = Recall()
        acc = BinaryAccuracy()
        for batch in self.test.as_numpy_iterator(): 
            X, y = batch
            yhat = self.model.predict(X)
            pre.update_state(y, yhat)
            re.update_state(y, yhat)
            acc.update_state(y, yhat)
        print(pre.result(), re.result(), acc.result())

def main(args):
    # Load datasets
    training_data = tf.keras.utils.image_dataset_from_directory(args.training_set)
    testing_data = tf.keras.utils.image_dataset_from_directory(args.validation_set)

    # Preprocess datasets
    training_data = training_data.map(lambda x, y: (x / 255, tf.one_hot(y, depth=8)))
    testing_data = testing_data.map(lambda x, y: (x / 255, tf.one_hot(y, depth=8)))

    # Create and configure trainer object
    trainer = Trainer(training_data, testing_data, args.model)
    trainer.save_classes(args.training_set)
    trainer.group_data(7, 2)
    if (not args.model):
        trainer.build_neural_network_layers(3)
    
    # Start training
    trainer.start(args.epochs)
    trainer.plot_history()
    
    # Save trained model
    trainer.save(args.model_save_location)


if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Train a neural network model")
    parser.add_argument("--model", type=str, default="", help="Path to the saved model file")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--training_set", type=str, help="Path to the training dataset")
    parser.add_argument("--testing_set", type=str, help="Path to the testing dataset")
    parser.add_argument("--save_model_path", type=str, default="models", help="Path to save the trained model")

    args = parser.parse_args()
    main(args)