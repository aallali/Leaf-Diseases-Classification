# CNN Image Classifier Documentation

## Main Steps of the Application:

1. **Data Preparation:**
   - Loading and preprocessing the training and testing datasets.
   - Grouping the data into training and testing sets.
   - One-hot encoding the labels for classification.

2. **Model Building:**
   - Defining the architecture of the neural network model.
   - Compiling the model with appropriate loss function, optimizer, and metrics.

3. **Training:**
   - Training the neural network model on the training dataset.
   - Monitoring the training process and evaluating the model's performance on the validation set.

4. **Evaluation:**
   - Evaluating the trained model on the testing dataset.
   - Calculating and analyzing metrics such as precision, recall, and accuracy.

5. **Visualization and Saving:**
   - Plotting the training history to visualize loss and accuracy trends.
   - Saving the trained model for future use.

## Detailed Explanation of Each Step:

1. **Data Preparation:**
   - **Loading Data:**
     - Using `tf.keras.utils.image_dataset_from_directory` to load image datasets from directories.
   - **Preprocessing Data:**
     - Normalizing pixel values by dividing by 255 to scale the pixel values between 0 and 1.
     - One-hot encoding labels using `tf.one_hot` to represent categorical data in a format that is suitable for machine learning algorithms, especially neural networks. When dealing with classification tasks, where the target variable (labels) represents different categories or classes, one-hot encoding is commonly used. 
   - **Grouping Data:**
     - Splitting the training dataset into training and validation sets using `take` and `skip` methods.

2. **Model Building:**
   - **Defining Model Architecture:**
     - Using `Sequential` API to create a sequential model. The Sequential API in TensorFlow/Keras is a user-friendly way to build neural network models layer by layer. It's called "Sequential" because you can simply stack layers on top of each other in a sequential manner, which is suitable for most deep learning tasks.
     - Adding layers using functions like `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense`.
   - **Compiling the Model:**
     - Specifying optimizer (`Adam`), loss function (`categorical_crossentropy`), and evaluation metrics (`accuracy`).
        **Optimizer (Adam)** : The optimizer is responsible for updating the weights of the neural network during training in order to minimize the loss function.
        **Loss Function (Categorical Crossentropy)** : The loss function measures how well the model performs during training by comparing the predicted output of the model with the actual target output. Categorical crossentropy is a commonly used loss function for classification problems with multiple classes. It quantifies the difference between the predicted probability distribution and the actual distribution of the labels
   - **Model Architecture:**
        - **Convolutional Layer 1:**
            - **Description**: This layer performs a 2D convolution operation on the input data using 16 filters with a kernel size of 3x3. It applies the ReLU activation function to introduce non-linearity into the network.
            - **Parameters**:
            - Filters: 16
            ! Initially, the values of the filters are random, but as the network learns from the training data, the filters adjust their values to become more effective at extracting relevant features from the input images.
            - Kernel Size: 3x3
            - Padding: 'same' (pad the input such that the output has the same height and width as the input)
            - Activation: ReLU
            ![Alt text](./images/convolutional.png)

        - **MaxPooling Layer 2:**
            - **Description**: This layer performs max pooling on the output of the previous convolutional layer. It reduces the spatial dimensions of the input by taking the maximum value within each window of size 2x2.
            - **Parameters**:
            - Pool Size: Default (2x2)
            ![Alt text](./images/Maxpool2.png)

        - **Convolutional Layer 3:**
            - **Description**: Similar to the first convolutional layer, this layer performs a 2D convolution operation on the input data using 32 filters with a kernel size of 3x3. It also applies the ReLU activation function.
            - **Parameters**:
            - Filters: 32
            - Kernel Size: 3x3
            - Padding: 'same'
            - Activation: ReLU

        - **MaxPooling Layer 4:**
            - **Description**: Another max pooling layer that reduces the spatial dimensions of the input by taking the maximum value within each window of size 2x2.
            - **Parameters**:
            - Pool Size: Default (2x2)

        - **Convolutional Layer 5:**
            - **Description**: This layer performs a 2D convolution operation on the input data using 64 filters with a kernel size of 3x3. It applies the ReLU activation function.
            - **Parameters**:
            - Filters: 64
            - Kernel Size: 3x3
            - Padding: 'same'
            - Activation: ReLU

        - **MaxPooling Layer 6:**
            - **Description**: Another max pooling layer that reduces the spatial dimensions of the input by taking the maximum value within each window of size 2x2.
            - **Parameters**:
            - Pool Size: Default (2x2)

        - **Flatten Layer 7:**
            - **Description**: This layer flattens the input tensor into a one-dimensional vector, which is required before feeding it into a fully connected (dense) layer.
            - **Parameters**: None
            ![Alt text](./images/Flatten.png)

        - **Dense Layer 8:**
            - **Description**: Fully connected layer with 128 neurons. It applies the ReLU activation function to introduce non-linearity into the network.
            - **Parameters**:
            - Neurons: 128
            - Activation: ReLU

            ! the Flatten layer converts the multi-dimensional output of the preceding layer into a one-dimensional format, which is then fed into the Dense layer. Each neuron in the Dense layer processes information from all elements of this one-dimensional array, enabling the network to learn complex patterns and relationships in the data.
        - **Dense Layer 9 (Output Layer):**
            - **Description**: The final dense layer with 8 neurons, corresponding to the number of output classes. It uses the softmax activation function to compute the probability distribution over the classes.
            - **Parameters**:
            - Neurons: 8
            - Activation: Softmax

    ![Alt text](https://www.researchgate.net/publication/339892439/figure/fig5/AS:871189681537032@1584719214501/A-schematic-illustration-of-the-convolutional-neural-network-CNN-architecture-The.png)
3. **Training:**
   - **Initiating Training:**
     - Using `fit` method to train the model on the training dataset.
     - Specifying the number of epochs for training.
   - **Monitoring Training:**
     - Recording the training history to track metrics like loss and accuracy during each epoch.
     - Using `validation_data` parameter to monitor the model's performance on the validation set during training.

4. **Evaluation:**
   - **Evaluating Model Performance:**
     - Using the trained model to predict labels for the testing dataset.
     - Calculating performance metrics such as precision, recall, and accuracy using `Precision`, `Recall`, and `BinaryAccuracy` metrics.

5. **Visualization and Saving:**
   - **Visualizing Training History:**
     - Plotting loss and accuracy curves using `matplotlib`.
   - **Saving the Model:**
     - Using the `save` method to save the trained model in the desired file format (e.g., `.h5`).

## Conclusion:
