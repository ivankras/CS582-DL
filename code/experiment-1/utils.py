import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

def to_grayscale_norm(img):
    treating = Image.fromarray(img)
    treating = treating.convert('L')
    return np.array(treating).reshape(32, 32, 1).astype('float32') / 255.0

def get_prepared_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    x_train_gray = np.array([to_grayscale_norm(img) for img in x_train])
    x_test_gray = np.array([to_grayscale_norm(img) for img in x_test])

    assert x_train_gray.shape == (50000, 32, 32, 1)
    assert x_test_gray.shape == (10000, 32, 32, 1)

    oh_encoder = OneHotEncoder(sparse=False)
    y_train_oh = oh_encoder.fit_transform(y_train)
    y_test_oh = oh_encoder.transform(y_test)

    assert y_train_oh.shape == (50000, 10)
    assert y_test_oh.shape == (10000, 10)

    return (x_train_gray, x_test_gray), (y_train_oh, y_test_oh)

def get_prepared_data_color():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    assert x_train.shape == (50000, 32, 32, 3)
    assert x_test.shape == (10000, 32, 32, 3)
    assert y_train.shape == (50000, 1)
    assert y_test.shape == (10000, 1)

    x_train_norm = np.array([img.astype('float32') / 255.0 for img in x_train])
    x_test_norm = np.array([img.astype('float32') / 255.0 for img in x_test])

    oh_encoder = OneHotEncoder(sparse=False)
    y_train_oh = oh_encoder.fit_transform(y_train)
    y_test_oh = oh_encoder.transform(y_test)

    assert y_train_oh.shape == (50000, 10)
    assert y_test_oh.shape == (10000, 10)

    return (x_train_norm, x_test_norm), (y_train_oh, y_test_oh)

def get_category(y, verbose=False):
    i2Cat = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }

    best_est_idx = np.argmax(y)
    if verbose:
        print(f'Best estimation ({best_est_idx}) confidence: {y[best_est_idx]}')
    return i2Cat[best_est_idx]
