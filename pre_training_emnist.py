import numpy as np
import keras

#pip install emnist

from dataset_emnist import load_emnist
from emnist import list_datasets, extract_training_samples, extract_test_samples

input_shape = (28, 28, 1)

#model in keras

#model = keras.models.Sequential()
#model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
#model.add(AvgPooling2D(pool_size=(2, 2)))
#model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
#model.add(Dense(128, activation=tf.nn.relu))
#model.add(Dropout(0.2))
#model.add(Dense(5,activation=tf.nn.softmax))

feature_layers = [
    keras.layers.Conv2D(28, (3, 3), padding='same', input_shape=(28, 28, 1)),
   #keras.layers.Activation('sigmoid'),
    keras.layers.AveragePooling2D(pool_size=(2,2)),
    keras.layers.Flatten()
    #keras.layers.Activation('relu'),
    #keras.layers.Conv2D(32, (3, 3), padding='same'),
    #keras.layers.Dropout(.25),
    
]

classification_layers = [
    keras.layers.Dense(128),
    keras.layers.Activation('relu'),
    keras.layers.Dropout(.50),
    keras.layers.Dense(27),
    keras.layers.Activation('softmax')
]

model = keras.models.Sequential(feature_layers + classification_layers)

public_dataset, _ = load_emnist()
(x_train, y_train), (x_test, y_test) = public_dataset
#images, labels = extract_training_samples('digits')
#(x_train, y_train) = extract_training_samples('letters')
#(x_test, y_test) = extract_test_samples('letters')
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    verbose=1,
    validation_data=(x_test, y_test))
model.save('pre-trained.h5')