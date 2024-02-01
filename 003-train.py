#training code 
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
import cv2
import imghdr
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model


class Trainer:
    def __init__(self, data_set_train, data_set_test):
        self.data_set = data_set_train
        self.train = 0
        self.val = 0
        self.test = data_set_test
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
        
    def build_neural_network_layers(self):
  
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
        
        # Compile the model
        self.model.compile(optimizer='adam',
                    loss='categorical_crossentropy',  # Categorical cross-entropy loss for multi-class classification
                    metrics=['accuracy'])


    def start(self, epoch):
        self.history = self.model.fit(self.train, epochs=epoch, validation_data=self.val)
    def save(self, path):
        self.model.save(os.path.join('models','imageclassifier.h5'))
        
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


def main():
    training_data = tf.keras.utils.image_dataset_from_directory('augmented_datasets_train_transformed')
    testing_data = tf.keras.utils.image_dataset_from_directory('augmented_datasets_validation')
    data_set = training_data.as_numpy_iterator()
    data_set_testing = testing_data.as_numpy_iterator()
    testing_data.as_numpy_iterator()
    d = training_data.as_numpy_iterator()
    batch = d.next()
    fig, ax = plt.subplots(ncols=4, figsize=(20,20))

    # for idx, img in enumerate(batch[0][:4]):
    #     ax[idx].imshow(img.astype(int))
    #     ax[idx].title.set_text(batch[1][idx])

    training_data = training_data.map(lambda x, y: (x / 255, tf.one_hot(y, depth=8)))  # One-hot encoding labels
    testing_data = testing_data.map(lambda x, y: (x / 255, tf.one_hot(y, depth=8)))      # One-hot encoding labels
    trainer = Trainer(training_data, testing_data)
    trainer.save_classes('augmented_datasets_train_transformed')
    trainer.group_data(7, 2)
    trainer.build_neural_network_layers()
    trainer.start(5)
    trainer.plot_history()
    trainer.save("path")

if __name__ == "__main__":
    # Call the main function if the script is run directly
    main()