"""
Trains and stores some MNIST keras dummy models
Those models are just used as dummy models to show how the k2arm and X-Cube-AI framework
work and not to achive high acuracy.

@author: Raphael Zingg zing@zhaw.ch
@copyright: 2019 ZHAW / Institute of Embedded Systems
"""

import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28*28)
x_test = x_test.reshape(x_test.shape[0], 28*28)

# minimal model
modelMin = tf.keras.models.Sequential()
modelMin.add(tf.keras.layers.Dense(10))
modelMin.add(tf.keras.layers.Activation('softmax'))
modelMin.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
modelMin.fit(x_train, y_train, epochs=5)
modelMin.evaluate(x_test, y_test)
modelMin.summary()
modelMin.save('output/modelMin/model.keras')

# middle sized model
modelMid = tf.keras.models.Sequential()
modelMid.add(tf.keras.layers.Dense(32))
modelMid.add(tf.keras.layers.Activation('relu'))
modelMid.add(tf.keras.layers.Dense(32))
modelMid.add(tf.keras.layers.Activation('relu'))
modelMid.add(tf.keras.layers.Dense(32))
modelMid.add(tf.keras.layers.Activation('relu'))
modelMid.add(tf.keras.layers.Dense(10))
modelMid.add(tf.keras.layers.Activation('softmax'))
modelMid.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
modelMid.fit(x_train, y_train, epochs=5)
modelMid.evaluate(x_test, y_test)
modelMid.summary()
modelMid.save('output/modelMid/model.keras')

# large model
modelLar = tf.keras.models.Sequential()
modelLar.add(tf.keras.layers.Dense(64))
modelLar.add(tf.keras.layers.Activation('relu'))
modelLar.add(tf.keras.layers.Dense(64))
modelLar.add(tf.keras.layers.Activation('relu'))
modelLar.add(tf.keras.layers.Dense(64))
modelLar.add(tf.keras.layers.Activation('relu'))
modelLar.add(tf.keras.layers.Dense(10))
modelLar.add(tf.keras.layers.Activation('softmax'))
modelLar.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
modelLar.fit(x_train, y_train, epochs=5)
modelLar.evaluate(x_test, y_test)
modelLar.summary()
modelLar.save('output/modelLar/model.keras')