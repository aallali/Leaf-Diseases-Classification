#!/usr/bin/env python3
import os
import argparse
from matplotlib import pyplot as plt
from libft import generate_config, load_config
from time import sleep
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf  # noqa: E402


# ######## extract Objects from TensorFlow #########
Adam = tf.keras.optimizers.Adam
Sequential = tf.keras.models.Sequential
Conv2D = tf.keras.layers.Conv2D

MaxPooling2D = tf.keras.layers.MaxPool2D
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Dropout = tf.keras.layers.Dropout

Precision = tf.keras.metrics.Precision
Recall = tf.keras.metrics.Recall
BinaryAccuracy = tf.keras.metrics.BinaryAccuracy

Callback = tf.keras.callbacks.Callback
###################################################


class FitCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch != 0 and (epoch % 10) == 0:
            print(f"Saving model at epoch {epoch}")
            self.model.save(
                os.path.join(
                    "./models",
                    f"model_progress_{epoch}.h5"
                )
            )
            print(f"Sleeping for 60 seconds before epoch {epoch + 1}")
            sleep(60)


class Trainer:
    def __init__(self, data_set_train, model, save_path):
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
        if (model):
            self.model = tf.keras.models.load_model(model)
        else:
            self.model = Sequential()
        self.history = 0
        self.classes = []

    def group_data(self):
        """
        Group the training dataset into training and validation sets.

        Args:
        - training: Percentage of data to use for training.
        - validation: Percentage of data to use for validation.
        """
        train_size = int(len(self.data_set)*.8)
        val_size = int(len(self.data_set)*.2)

        self.train = self.data_set.take(train_size)
        self.val = self.data_set.skip(train_size).take(val_size)

    def save_classes(self, path):
        """
        Save the classes found in the dataset.

        Args:
        - path: Directory containing the dataset.
        """
        dataset_directory = path
        class_directories = [d for d
                             in os.listdir(dataset_directory) if
                             os.path.isdir(os.path.join(dataset_directory, d))]
        class_directories = sorted(class_directories)
        self.classes = class_directories
        print("Classes saved:", class_directories)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with open(self.save_path + '/labels.txt', 'w') as file:
            for class_name in class_directories:
                file.write(class_name + '\n')

    def build_neural_network_layers(self):
        """
        Build the neural network layers based on the chosen model\
            configuration.

        Args:
        - model_choice: Integer indicating the model configuration to use.
        """

        self.model.add(
            Conv2D(
                    32, (3, 3),
                    activation='relu',
                    input_shape=(128, 128, 3)
                )
            )
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(8, activation='softmax'))

        # learning_rate = 0.001

        optimizer = Adam()

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def start(self, epoch):
        """
        Start training the neural network.

        Args:
        - epoch: Number of epochs for training.
        """
        self.history = self.model.fit(
            self.train,
            epochs=epoch,
            callbacks=[
                FitCallback()
            ]
        )

    def save(self, path):
        """
        Save the trained model.

        Args:
        - path: Directory to save the trained model.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, 'model.h5'))

    def plot_history(self):
        """
        Plot training history (loss and accuracy).
        """
        fig = plt.figure()
        plt.plot(self.history.history['loss'], color='teal', label='loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()

        fig = plt.figure()
        plt.plot(
            self.history.history['accuracy'],
            color='teal',
            label='accuracy'
        )
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
        print(f'Precision: {pre.result().numpy()},\
              Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')


def main(args):
    """
    Main function to train the convolutional neural network.

    Args:
    - config: Dictionary containing configuration parameters.
    """

    training_data = tf.keras.utils.image_dataset_from_directory(
        args.training_set, image_size=(128, 128))
    training_data = training_data.map(lambda x, y: (x / 255, tf.one_hot(
        y, depth=8)))
    trainer = Trainer(training_data, args.model, args.model_save_location)
    trainer.save_classes(args.training_set)
    trainer.group_data()
    if (not args.model):
        trainer.build_neural_network_layers()
    if (args.epochs):
        trainer.start(args.epochs)
        trainer.plot_history()
        trainer.save(args.model_save_location)


if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Train a convolutional neural network"
    )
    parser.add_argument(
        "-gc",
        "--generate-config",
        action="store_true",
        help="Generate a default configuration file"
    )
    args = parser.parse_args()

    if args.generate_config:
        generate_config('config.yaml')
        print("Default configuration file generated successfully.")
    else:
        config = load_config('config.yaml')
        main(config)
