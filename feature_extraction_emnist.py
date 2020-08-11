import numpy as np
import keras

from dataset_emnist import load_emnist
from keras.utils.generic_utils import CustomObjectScope

with CustomObjectScope({'relu6': keras.applications.mobilenet.relu6,'DepthwiseConv2D': keras.applications.mobilenet.DepthwiseConv2D}):
    model = keras.models.load_model('pre-trained.h5')
    model.summary()

_, private_dataset = load_emnist()

(x_train_images, y_train), (x_test_images, y_test) = private_dataset
x_train_features = model.predict(x_train_images)
x_test_features  = model.predict(x_test_images)

#save extracted features and labels
np.save('x_train_features.npy', x_train_features)
np.save('y_train.npy', y_train)

np.save('x_test_features.npy', x_test_features)
np.save('y_test.npy', y_test)