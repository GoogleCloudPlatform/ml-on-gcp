from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
