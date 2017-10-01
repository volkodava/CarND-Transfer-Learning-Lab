# TODO: import Keras layers you need here
import pickle
from enum import Enum

import numpy as np
import tensorflow as tf
from keras.layers import Flatten, Dense, Activation
from keras.models import Sequential

flags = tf.app.flags
FLAGS = flags.FLAGS


class ModelType(Enum):
    INCEPTION = 0
    VGG = 1
    RESNET = 2


class DataType(Enum):
    CIFAR = 0
    TRAFFIC = 1


default_model = ModelType.RESNET
default_data = DataType.TRAFFIC

train_file_pattern = "%s-100/%s_%s_100_bottleneck_features_train.p"
valid_file_pattern = "%s-100/%s_%s_bottleneck_features_validation.p"

model_prefix = ""
data_prefix = ""

if default_model == ModelType.INCEPTION:
    model_prefix = "inception"
elif default_model == ModelType.VGG:
    model_prefix = "vgg"
elif default_model == ModelType.RESNET:
    model_prefix = "resnet"
else:
    raise ValueError("Model type not exists %s" % default_model)

if default_data == DataType.CIFAR:
    data_prefix = "cifar10"
elif default_data == DataType.TRAFFIC:
    data_prefix = "traffic"
else:
    raise ValueError("Data type not exists %s" % default_data)

training_file = train_file_pattern % (model_prefix, model_prefix, data_prefix)
validation_file = valid_file_pattern % (model_prefix, model_prefix, data_prefix)

print("training_file: ", training_file)
print("validation_file: ", validation_file)

flags.DEFINE_string('training_file', training_file,
                    "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', validation_file,
                    "Bottleneck features validation file (.p)")
flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")


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
    history = model.fit(X_train, y_train_one_hot, FLAGS.batch_size, FLAGS.epochs, validation_data=(X_val, y_val_norm),
                        shuffle=True, verbose=0)

    print("Model: ", default_model)
    print("Data: ", default_data)
    print("Accuracy: %s%%" % (history.history['acc'][-1] * 100))
    print("Loss: %s%%" % (history.history['loss'][-1]))


# Data shape =  (1000, 1, 1, 2048)
# Model:  ModelType.INCEPTION
# Data:  DataType.CIFAR
# Accuracy: 100.0%
# Loss: 0.0937012120485%
#
# Model:  ModelType.VGG
# Data:  DataType.CIFAR
# Accuracy: 95.0999994755%
# Loss: 0.255422445536%
#
# Model:  ModelType.RESNET
# Data:  DataType.CIFAR
# Accuracy: 100.0%
# Loss: 0.0710589367747%
#
#
# Model:  ModelType.INCEPTION
# Data:  DataType.TRAFFIC
# Accuracy: 100.0%
# Loss: 0.0272921757764%
#
# Model:  ModelType.VGG
# Data:  DataType.TRAFFIC
# Accuracy: 99.558139607%
# Loss: 0.0852844921032%
#
# Model:  ModelType.RESNET
# Data:  DataType.TRAFFIC
# Accuracy: 100.0%
# Loss: 0.0321581754331%


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
