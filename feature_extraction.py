# TODO: import Keras layers you need here
import pickle

import numpy as np
import tensorflow as tf
from keras.layers import Flatten, Dense, Activation
from keras.models import Sequential

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', 'vgg-100/vgg_cifar10_100_bottleneck_features_train.p',
                    "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', 'vgg-100/vgg_cifar10_bottleneck_features_validation.p',
                    "Bottleneck features validation file (.p)")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)
    nb_classes = len(np.unique(y_train))

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)
    print("Number of classes = ", nb_classes)
    print("Data shape = ", X_train.shape)

    # TODO: define your model and hyperparams here
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    train_shape = X_train.shape[1:]

    model = Sequential()
    model.add(Flatten(input_shape=train_shape))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    from sklearn.preprocessing import LabelBinarizer
    label_binarizer = LabelBinarizer()

    y_train_one_hot = label_binarizer.fit_transform(y_train)
    y_val_norm = label_binarizer.fit_transform(y_val)

    model.compile('adam', 'categorical_crossentropy', ['accuracy'])
    model.fit(X_train, y_train_one_hot, nb_epoch=50, validation_data=(X_val, y_val_norm),
              shuffle=True)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
