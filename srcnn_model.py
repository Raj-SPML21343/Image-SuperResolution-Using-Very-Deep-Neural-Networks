import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from data_set import data_maker



class PSNRCallback(Callback):
    def __init__(self, validation_data):
        super(PSNRCallback, self).__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        # Predict the validation set using the model
        y_pred = self.model.predict(self.validation_data[0])
        # Calculate PSNR using NumPy
        psnr = tf.image.psnr(self.validation_data[1], y_pred, max_val=1.5)
        # Add PSNR to the logs
        logs = logs or {}
        logs['val_psnr'] = np.mean(psnr)
        print("PSNR: {:.4f}".format(np.mean(psnr)))

# Define the model architecture of SRCNN
model_srcnn = tf.keras.Sequential()
model_srcnn.add(Conv2D(64, (9, 9), padding='same', input_shape=(None, None, 3)))
model_srcnn.add(Activation('relu'))
model_srcnn.add(Conv2D(32, (1, 1), padding='same'))
model_srcnn.add(Activation('relu'))
model_srcnn.add(Conv2D(3, (5, 5), padding='same'))

# Compile the model
model_srcnn.compile(loss='mse', optimizer=SGD(learning_rate=0.001, momentum=0.9, nesterov=True), metrics=["accuracy"])

# Train the model with PSNRCallback
checkpoint = ModelCheckpoint('model_srcnn.h5', monitor='val_loss', save_best_only=True)

train_x, train_y, test_x, test_y = data_maker()
psnr_callback = PSNRCallback(validation_data=(test_x, test_y))
hist_srcnn = model_srcnn.fit(train_x, train_y, batch_size=16, epochs=100, validation_data=(test_x, test_y), callbacks=[checkpoint, psnr_callback])
