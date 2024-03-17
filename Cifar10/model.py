from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout, BatchNormalization

# Declaration of some global variables:
PIXELS_HEIGHT = 32
PIXELS_WIDTH = 32
CLASSES = 10


def load_cifar10():
    # Load CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Convert labels to one-hot encoding
    train_labels = to_categorical(train_labels, CLASSES)
    test_labels = to_categorical(test_labels, CLASSES)

    return train_images, train_labels, test_images, test_labels


def build_model():
    # Build the Neural Network
    # Sequential model is use for classification, while a non-Sequential model is for languages processing
    # 3X3 is the avg for this problem while a lot of layers can reach overfitting, and few is not enough
    model = models.Sequential()

    # Convolutional layers with ReLU activation for feature extraction from images
    # Filter is like a feature of the pixels, 32 for time saving and overfitting
    # The filter is a group of 3X3 pixels
    # Relu normalize the values, while negative is 0 and positive remain the same
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    # BatchNormalization is for normalize value by reduce average to 0 and deviation to 1
    # Batch normalization helps normalize activation values, improving training stability.
    model.add(BatchNormalization())
    # Maxpooling is a scanner of 2X2 and maximize only the "relevant" features of pixels
    model.add(layers.MaxPooling2D((2, 2)))
    # Filters are increasing in progression for time, overfitting and rule the complexity of learning rate
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())

    # Dense layers
    # Dense is for the classifications
    # Those layers are fully connected, so they have to be flattened to 1D array
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    # Dropout is for overfitting, its disconnect some random neurons
    model.add(Dropout(0.5))
    # Softmax is for normalize the probability between [0,1]
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam',  # Define the optimization algorithm (Adaptive Moment Estimation)
                  # to learn "with" the process
                  loss='categorical_crossentropy',  # Specify the loss function (categorical_crossentropy for
                  # multi-class classification). the goal is to increase the probabilities of right classifications
                  metrics=['accuracy'])

    return model


def build_model_old():
    # Build the Neural Network
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(PIXELS_HEIGHT, PIXELS_WIDTH, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model